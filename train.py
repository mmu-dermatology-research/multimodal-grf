# +
import torch
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data as data
import torch.nn.functional as F
from torch.cuda import amp

import os
import argparse
import math
import random
import time
import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tqdm import tqdm
from utils.utils import AvgMeter, build_model
from utils.dataloader import create_dataset
from utils.ema import ModelEMA
from utils.loss import Lossncriterion, structure_loss
from utils.optim import set_optimizer
from utils.metrics import iou_score
from utils.dataloader import test_dataset

from collections import OrderedDict

# -
def arg_parser():
    parser = argparse.ArgumentParser()
    # for training
    parser.add_argument('--epoch', type=int, default=60, help='# epoch')
    parser.add_argument('--batchsize', type=int, default=5, help='batch size')
    parser.add_argument('--kfold', type=int, default=5, help='# fold')
    parser.add_argument('--k', type=int, default=-1, help='specific # fold')
    parser.add_argument('--seed', type=int, help='random seed for split data')
    parser.add_argument('--lr', type=float, default=1e-5, help='learning rate')
    parser.add_argument('--name', type=str, default='exp', help='exp name to annotate this training')
    parser.add_argument('--optimizer', type=str, default='AdamW', help='choose optimizer')

    # for data
    parser.add_argument('--dataratio', type=float,default=0.8, help='ratio of data for training/val')
    parser.add_argument('--data_path', nargs='+', type=str, default='dataset/train/', help='path to training data')
    parser.add_argument('--augmentation', action='store_true', help='activate data augmentation')

    # for model
    parser.add_argument('--class-num', type=int, default=1, help='output class')
    parser.add_argument('--arch', type=int, default=53, help='backbone version')
    parser.add_argument('--trainsize', type=int, default=512, help='img size')
    parser.add_argument('--weight', type=str, default='', help='path to model weight')
    parser.add_argument('--modelname', type=str, default='lawinloss4', help='choose model')
    parser.add_argument('--decoder', type=str, default='lawin', help='choose decoder')
    parser.add_argument('--rect', action='store_true', help='padding the image into rectangle')

    return parser.parse_args()

def trainingplot(rec, name):
    fig, ax = plt.subplots()
    ax2 = ax.twinx()
            
    x = np.arange(0, opt.epoch, 1)
    line1, = ax.plot(x, rec[0, :], label='Loss')
    line2, = ax.plot(x, rec[1, :], label='Deep loss 1')
    line3, = ax.plot(x, rec[2, :], label='Deep loss 2')
    line4, = ax2.plot(x, rec[3, :], label='Dice', color='r')
    line5, = ax2.plot(x, rec[4, :], label='Best dice', color='y')
            
    ax.set_xlabel('Epoch')
    ax.set_title(name, fontsize=16)
            
    ax.set_ylim([0, 1.5])
    ax2.set_ylim([0.6, 0.85])
            
    ax.set_ylabel('Loss', color='g')
    ax2.set_ylabel('Dice', color='b')
            
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')
    
    plt.savefig(name, dpi=300)

def test(model, criterion, test_loader):
    model.eval()
    mdice, mwbce, mwiou, omax, omin, miou = AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter()
    pbar = enumerate(test_loader)
    print(('\n' + '%10s' * 6) % ('Dice', 'gpu_mem', 'wbce', 'wiou', 'max', 'min'))
    pbar = tqdm(pbar, total=len(test_loader))
    
    for i, (image, gt, name) in pbar:
        gt = gt.cuda()
        image = image.cuda()
        with torch.no_grad():
            output = model(image)
            
        wbce, wiou = structure_loss(output, gt)
        dice = criterion.dice_coefficient(output, gt)
        iou = iou_score(output, gt)
        
        mdice.update(dice.item(), 1)
        mwbce.update(wbce.item(), 1)
        mwiou.update(wiou.item(), 1)
        omax.update(output.max().item(), 1)
        omin.update(output.min().item(), 1)
        miou.update(iou.item(), 1)
        
        mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)
        s = ('%10.4g' + '%10s' + '%10.4g' * 4) % (mdice.avg, mem, mwbce.avg, mwiou.avg, omax.avg, omin.avg)
        pbar.set_description(s)
        
    return mdice.avg, mwbce.avg+mwiou.avg, miou.avg

def train(train_loader, model, optimizer, epoch, opt, scaler, ema, criterion):
    model.train()
    loss_record, deep1, deep2, boundary, iou_record, dice_record = AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter()
    pbar = enumerate(train_loader)
    print(('\n' + '%10s' * 6) % ('Epoch', 'gpu_mem', 'loss', 'deep1', 'deep2', 'bound'))
    if opt.global_rank in [-1, 0]:
        pbar = tqdm(pbar, total=len(train_loader))
    
    for i, (images, gts, name) in pbar:
        images = images.cuda()
        gts = gts.cuda()

        multiscale = 0.25
        trainsize = random.randrange(int(opt.trainsize * (1 - multiscale)), int(opt.trainsize * (1 + multiscale))) // 64 * 64
        images = F.interpolate(images, size=(trainsize, trainsize), mode='bilinear', align_corners=False)
        gts = F.interpolate(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=False)
        
        # ---- forward ----
        with amp.autocast():
            output = model(images)
            loss = criterion(output[0], gts)
            deep_loss = criterion(output[1], gts)
            deep_loss2 = criterion(output[2], gts)
            boundary_loss = criterion.boundary_forward(output[3], gts)
            iou = iou_score(output[0], gts)
            dice = criterion.dice_coefficient(output[0], gts)
            
            scaler.scale(loss).backward(retain_graph=True)
            scaler.scale(deep_loss).backward(retain_graph = True)
            scaler.scale(deep_loss2).backward(retain_graph = True)
            scaler.scale(boundary_loss).backward()
                
            loss_record.update(loss.item(), opt.batchsize)
            deep1.update(deep_loss.item(), opt.batchsize)
            deep2.update(deep_loss2.item(), opt.batchsize)
            boundary.update(boundary_loss.item(), opt.batchsize)
            iou_record.update(iou.item(), opt.batchsize)
            dice_record.update(dice.item(), opt.batchsize)

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        ema.update(model)
        
        if opt.global_rank in [-1, 0]:
            mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
            s = ('%10s' * 2 + '%10.4g' * 4) % ('%g/%g' % (epoch, opt.epoch - 1), mem, loss_record.avg, deep1.avg, deep2.avg, boundary.avg)
            pbar.set_description(s)
            ema.update_attr(model)
    
    return loss_record.avg, deep1.avg, deep2.avg, iou_record.avg, dice_record.avg



if __name__ == '__main__':
    opt = arg_parser()
    if opt.seed == None:
        opt.seed = np.random.randint(2147483647)
        print('You chose seed %g in this training.'%opt.seed)
    opt.world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    opt.global_rank = int(os.environ['RANK']) if 'RANK' in os.environ else -1   
    
    logname = opt.name + '_' + time.strftime("%Y%m%d%H%M%S", time.localtime()) + '.log'
    logging.basicConfig(filename=logname, format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
    logging.info(opt)
    print('logging at ', logname)
    
    save_path = os.path.join('weights', opt.name)
    os.makedirs(save_path, exist_ok=True)
    
    for k in range(opt.kfold):
        if opt.kfold > 1:
            print('%g/%g-fold'%(k+1, opt.kfold))
            if opt.k != -1:
                k = opt.k
                
        train_dataset = create_dataset(opt.data_path, opt.trainsize, opt.augmentation, True, opt.dataratio, opt.rect, k=k, k_fold=opt.kfold, seed=opt.seed)
        train_loader = data.DataLoader(dataset=train_dataset, batch_size=opt.batchsize, shuffle=True, num_workers=4, pin_memory=True)
        del train_dataset

        test_dataset = create_dataset(opt.data_path, opt.trainsize, False, False, opt.dataratio, opt.rect, k=k, k_fold=opt.kfold, seed=opt.seed)
        test_loader = data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=True, num_workers=4, pin_memory=True)
        del test_dataset
        
        model = build_model() #opt.modelname, opt.class_num, opt.arch)
        logging.info(model)
        
        optimizer = set_optimizer(model, opt.optimizer, opt.lr)
        logging.info(optimizer)
        lf = lambda x: ((1.001 + math.cos(x * math.pi / opt.epoch))) #* (1 - 0.1) + 0.1  # cosine
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
        scaler = amp.GradScaler()
        ema = ModelEMA(model) if opt.global_rank in [-1, 0] else None
        criterion = Lossncriterion().cuda()
        
        if opt.weight != '':
            model.load_state_dict(torch.load(opt.weight))#, map_location=device))

        log = OrderedDict([
            ('epoch', []),
            ('lr', []),
            ('loss', []),
            ('deep1', []),
            ('deep2', []),
            ('dice', []),
            ('iou', []),
            ('val_loss', []),
            ('val_dice', []),
            ('val_iou', [])
        ])
        
        print('Start training at rank: ', opt.global_rank)
        best = 0
        rec = np.zeros((6, opt.epoch))
        for epoch in range(opt.epoch):
            optimizer.zero_grad()
            loss, deep1, deep2, iou, dice = train(train_loader, model, optimizer, epoch, opt, scaler, ema, criterion)
            scheduler.step()

            val_dice, val_loss, val_iou = test(model, criterion, test_loader)
            if val_iou > best:
                best = val_iou
                # if best > 0.8:
                #    pthname = os.path.join(save_path, '%s_%g,%g_best_%g.pth'%(opt.name, k+1, opt.kfold, int(best*10000)))
                #    torch.save(model.state_dict(), pthname)
                #    print('[Saving Best Weight]', pthname)
            rec[0, epoch] = loss
            rec[1, epoch] = deep1
            rec[2, epoch] = deep2
            rec[3, epoch] = val_dice
            rec[4, epoch] = best
            rec[5, epoch] = val_loss
            logging.info('Epoch: %g,mDice: %g,Best mDice: %g,loss: %g,loss2: %g,loss3: %g,lr: %g'%(epoch, val_dice, best, loss, deep1, deep2, scheduler.get_last_lr()[0]))
            print("best val_iou: ", best)

            # save csv log for current epoch
            log['epoch'].append(epoch)
            log['lr'].append(scheduler.get_last_lr()[0])
            log['loss'].append(loss)
            log['deep1'].append(deep1)
            log['deep2'].append(deep2)
            log['dice'].append(dice)
            log['iou'].append(iou)
            log['val_loss'].append(val_loss)
            log['val_dice'].append(val_dice)
            log['val_iou'].append(val_iou)

            pd.DataFrame(log).to_csv(opt.name + '_fold-' + str(k) + '_log.csv', index=False)

            torch.save(model.state_dict(), os.path.join(save_path, 'fold-' + str(k) + '_' + '%s_%g,%g_val_iou_0.%g_epoch_%g.pth'%(opt.name, k+1, opt.kfold, int(val_iou*10000), epoch)))
            # trainingplot(rec, os.path.join(save_path, '%s_%g,%g_val_iou_0.%g_epoch_%g.pdf'%(opt.name, k+1, opt.kfold, int(val_iou*10000), epoch)))

        # torch.save(model.state_dict(), os.path.join(save_path, '%s_%g,%g_final_%g.pth'%(opt.name, k+1, opt.kfold, int(best*10000))))
        # trainingplot(rec, os.path.join(save_path, '%s_%g,%g_final_%g.pdf'%(opt.name, k+1, opt.kfold, int(best*10000))))
        
        if opt.k != -1:
            break