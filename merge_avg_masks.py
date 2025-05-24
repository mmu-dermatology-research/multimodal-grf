import cv2
import numpy as np
import glob
import ntpath

from PIL import Image

mask1_path = '/dob_grf_i2/pred_mask/'
mask2_path = '/gender_grf_i2/pred_mask/'
mask3_path = '/hdd_grf_i2/pred_mask/'

mask4_path = '/dob_grf_i5/pred_mask/'
mask5_path = '/gender_grf_i5/pred_mask/'
mask6_path = '/hdd_grf_i5/pred_mask/'

mask_filenames = glob.glob(mask1_path + '*.png')

for msk in mask_filenames:
  path, filename = ntpath.split(msk)
  mask1 = cv2.imread(mask1_path + filename, cv2.IMREAD_GRAYSCALE)
  mask2 = cv2.imread(mask2_path + filename, cv2.IMREAD_GRAYSCALE)
  mask3 = cv2.imread(mask3_path + filename, cv2.IMREAD_GRAYSCALE)

  mask4 = cv2.imread(mask4_path + filename, cv2.IMREAD_GRAYSCALE)
  mask5 = cv2.imread(mask5_path + filename, cv2.IMREAD_GRAYSCALE)
  mask6 = cv2.imread(mask6_path + filename, cv2.IMREAD_GRAYSCALE)

  d1 = cv2.distanceTransform(mask1, cv2.DIST_LABEL_PIXEL, 3) - cv2.distanceTransform(~mask1, cv2.DIST_LABEL_PIXEL, 3)
  d2 = cv2.distanceTransform(mask2, cv2.DIST_LABEL_PIXEL, 3) - cv2.distanceTransform(~mask2, cv2.DIST_LABEL_PIXEL, 3)
  d3 = cv2.distanceTransform(mask3, cv2.DIST_LABEL_PIXEL, 3) - cv2.distanceTransform(~mask3, cv2.DIST_LABEL_PIXEL, 3)

  d4 = cv2.distanceTransform(mask4, cv2.DIST_LABEL_PIXEL, 3) - cv2.distanceTransform(~mask4, cv2.DIST_LABEL_PIXEL, 3)
  d5 = cv2.distanceTransform(mask5, cv2.DIST_LABEL_PIXEL, 3) - cv2.distanceTransform(~mask5, cv2.DIST_LABEL_PIXEL, 3)
  d6 = cv2.distanceTransform(mask6, cv2.DIST_LABEL_PIXEL, 3) - cv2.distanceTransform(~mask6, cv2.DIST_LABEL_PIXEL, 3)

  mask = 255*(d1 + d2 + d3 + d4+ d5 + d6)
  mask = mask.clip(0, 255).astype("uint8")

  output = Image.fromarray(mask.astype(np.uint8))
  output = output.convert("P", palette=Image.ADAPTIVE, colors=8)
  output.save('merged_masks/' + filename)
