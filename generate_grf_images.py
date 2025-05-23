import numpy as np
import matplotlib.pyplot as plt
import csv
import math

from PIL import Image

# uses code from here: https://andrewwalker.github.io/statefultransitions/post/gaussian-fields/

def fftIndgen(n):
    a = list(range(0, n//2+1))
    b = list(range(1, n//2))
    b = reversed(b)
    b = [-i for i in b]
    return a + b

def gaussian_random_field(Pk = lambda k : k**-3.0, size = (100,100)):
    def Pk2(kx, ky):
        if kx == 0 and ky == 0:
            return 0.0
        return np.sqrt(Pk(np.sqrt(kx**2 + ky**2)))
    noise = np.fft.fft2(np.random.normal(size = (size[0], size[1])))
    amplitude = np.zeros((size[0], size[1]))
    for i, kx in enumerate(fftIndgen(size[0])):
        for j, ky in enumerate(fftIndgen(size[1])):
            amplitude[i, j] = Pk2(kx, ky)
    return np.fft.ifft2(noise * amplitude)

# specify path to your metadata csv file
with open('/dataset/patient_data.csv', mode='r') as csv_file:
  csv_reader = csv.DictReader(csv_file)
  for row in csv_reader:
    filename = row['image']
    filename = filename.replace('.jpg', '.png')
    val = float(row['dob_norm']) # set to value from your csv

    # set a fixed random seed for each metadata type
    # dob seed    = 76539635
    # gender seed = 88118546
    # hdd seed    = 41094303
    np.random.seed(76539635)
	
    # change this to either 2 or 5 to replicate the types of GRFs from the paper
    pi = 5.0
    
    # use this for variables that only have 2 possible values
    # comment this line out if variables have more than 2 possible values
    p_spec = -abs(pi + val)

    # use this for variables that have more than 2 possible values (e.g. dob)
    # comment these 2 lines out if variables have only 2 possible values
    # pd = math.modf(val)[0]
    # p_spec = -abs(pi + pd)

    print('processing: ' + filename)
    print('p_spec: ' + str(p_spec))

    for alpha in [p_spec]:
      out = gaussian_random_field(Pk = lambda k: k**alpha, size=(480,640))

      # save colour gaussian random field image
      save_dir = '/dataset/grf/'
      plt.imsave(save_dir + filename, out.real)
      
      # reload image and convert to 8-bit greyscale
      im = Image.open(save_dir + filename).convert('L')
      im.save(save_dir + filename)
