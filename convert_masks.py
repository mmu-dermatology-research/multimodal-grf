import glob
import cv2
import numpy as np

from PIL import Image

masks = glob.glob('pred_masks/*.png')

print('converting masks to 8-bit black & white...')

for mask in masks:
  img = Image.open(mask)

  # convert to black & white
  thresh = 120 # 30
  fn = lambda x : 255 if x > thresh else 0
  img = img.convert('L').point(fn, mode='1')
  img.save(mask)

  img = cv2.imread(mask)

  # convert to 8-bit (required by get_metrics.py)
  img = Image.fromarray(img.astype(np.uint8))
  img = img.convert("L", palette=Image.ADAPTIVE, colors=8)
  
  img.save(mask)

print('complete')
