import matplotlib.pyplot as plt
import numpy as np
from skimage import io,color,morphology,measure,filters
from scipy import ndimage
from scipy.cluster.vq import vq, kmeans2
from skimage.segmentation import random_walker
#from structure_tensor import eig_special_2d, structure_tensor_2d

fnames=[]
fnames.append( '2x/Rb/20221209_165815.jpg')
fnames.append( '2x/Na/20221209_170719.jpg')
fnames.append( '2x/Cs/20221209_165247.jpg')
fnames.append( '2x/K/20221209_170131.jpg')
fnames.append( '2x/Li/20221209_171852.jpg')

images = []
for fname in fnames:
    newimage = plt.imread(fname)
    images.append(newimage)
    
fig,axs = plt.subplots(5,5,sharex=True,sharey=True)

for i in range(5):
    print(f'Image {i+1}')        
    axs[0][i].imshow(images[i])
    gray=color.rgb2gray(images[i])
    filtered = filters.farid(gray)
    axs[1][i].imshow(filtered,cmap='gray')
    ths = [filters.threshold_isodata(filtered),filters.threshold_otsu(filtered)]
    print(ths)
    axs[2][i].imshow(filtered>ths[0],cmap='binary')
    axs[3][i].imshow(filtered>ths[1],cmap='binary')
    axs[4][i].imshow(filtered>filters.threshold_local(filtered))
        
plt.show()
