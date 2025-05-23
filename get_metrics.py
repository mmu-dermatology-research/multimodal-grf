import os
import json
import csv
import glob
import shutil
import SimpleITK
from evalutils import ClassificationEvaluation
from evalutils.io import SimpleITKLoader
from evalutils.io import SimpleITKLoader

gt_path = 'dataset/test/masks/'
pred_path = 'pred_mask/'

if not os.path.exists('metrics'):
    os.makedirs('metrics')
	
file_loader = SimpleITKLoader()
gt_images = os.listdir(gt_path)
FNE_T = []
FPE_T = []
MO_T = []
UO_T = []
VS_T = []
JC_T = []
DC_T = []

csv_items = []

def read_mask(path):
    x = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
    # x = cv2.resize(x, (256,256))
    x = x/255.0
    x = x > 0.5
    x = x.astype(np.int32)
    return x  

for image in gt_images:
    
    # Load the images for this case
    gt = file_loader.load_image(gt_path+image)
    pred = file_loader.load_image(pred_path+image)
    
    # Cast to the same type
    caster = SimpleITK.CastImageFilter()
    caster.SetOutputPixelType(SimpleITK.sitkUInt8)
    caster.SetNumberOfThreads(1)
    gt = caster.Execute(gt)
    pred = caster.Execute(pred)
    
    # get our ground truth image 
    imgGT = SimpleITK.GetArrayFromImage(gt).max()
    img = SimpleITK.GetArrayFromImage(gt) 
    img[img > 0] == 1
    gt = SimpleITK.GetImageFromArray(img)
    # get our prediction image 
    imgPD = SimpleITK.GetArrayFromImage(pred).max()
    #print(imgGT)
    #print(imgPD)
    # Score the case
    FNE = 1
    FPE = 1
    MO = 0
    UO = 0
    VS = 1
    JC = 0
    DC = 0
    
    if imgGT == 0 and imgPD == 0:
        FNE = 0
        FPE = 0
        MO = 1
        UO = 1
        VS = 0
        JC = 1
        DC = 1
    elif (imgGT == 0 and imgPD != 0) or (imgGT != 0 and imgPD == 0):
        FNE = 1
        FPE = 1
        MO = 0
        UO = 0
        VS = 1
        JC = 0
        DC = 0
    else:
        # simpleITK uses multi class allowing blur to come through, whereas sklearn allows binary to be processed
        overlap_measures = SimpleITK.LabelOverlapMeasuresImageFilter()
        overlap_measures.SetNumberOfThreads(1)
        try:
            overlap_measures.Execute(gt, pred)
            # FNE : False Negative Error
            FNE = overlap_measures.GetFalseNegativeError()
            # FPE : False Positive Error
            FPE = overlap_measures.GetFalsePositiveError()
            # MO : Mean Overlap
            MO = overlap_measures.GetMeanOverlap()
            # UO : Union Overlap
            UO = overlap_measures.GetUnionOverlap()
            # VS : Volume Similarity 
            VS = overlap_measures.GetVolumeSimilarity() 
            # JC : Jaccard cofficent 
            JC = overlap_measures.GetJaccardCoefficient()
            # DC : Dice cofficent
            DC = overlap_measures.GetDiceCoefficient()
        except: 
            print("Segmentations failed to overlap: " + image)
        
    FNE_T.append(FNE)
    FPE_T.append(FPE)
    MO_T.append(MO)
    UO_T.append(UO)
    VS_T.append(VS)
    JC_T.append(JC)
    DC_T.append(DC)

def mean(lst): 
    return sum(lst)/len(lst)

my_dictionary = {
'case': {},
'aggregates':{
    "FalseNegativeError":mean(FNE_T),
	"FalsePositiveError":mean(FPE_T),
	"MeanOverlap":mean(MO_T),
    "UnionOverlap":mean(UO_T),
	"VolumeSimilarity":mean(VS_T),
	"JaccardCoefficient":mean(JC_T),
    "DiceCoefficient":mean(DC_T),
    }
}

jsonFileName = 'metrics/metrics.json'
fields = ["name","FNE","FPE","MO","UO","VS","JC","DC"]
rows =[]

# save  metrics results into json file
# print("name,FNE,FPE,MO,UO,VS,JC,DC")
for i in range(len(gt_images)):
    rows.append([gt_images[i], FNE_T[i], FPE_T[i], MO_T[i], UO_T[i],VS_T[i],JC_T[i],DC_T[i]])
    #print(gt_images[i]+","+str(FNE_T[i])+","+str(FPE_T[i])+","+str(MO_T[i])+","+str(UO_T[i])+","+str(VS_T[i])+","+str(JC_T[i])+","+str(DC_T[i]))

fileObj= open(jsonFileName, "w+")
json.dump(my_dictionary, fileObj)
fileObj.close()
