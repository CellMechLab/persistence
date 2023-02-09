
import matplotlib.pyplot as plt
import numpy as np
from skimage import morphology,measure,io,color,filters
import functions
import glob

from scipy.cluster.vq import vq, kmeans2

images = glob.glob('./2x/Na/*.jpg')
nasty = './2x/Na/20221209_170814.jpg'

cleaning = True
magic = False
stop = True

for image in images:
    print(image)
    if stop is True:
        image = nasty
    src = io.imread(image)

    mask = functions.segment(src)
    firstguess = functions.select(mask)
    isolated = functions.isolate(firstguess,src)
    filament = functions.skinny(isolated)

    colorised = color.label2rgb(filament, bg_label=0,image=src)
    
    fig,ax=plt.subplots() #1,2,sharex=True,sharey=True
    fig.suptitle(image)
    ax.imshow(colorised)

    if stop is True:
        break


plt.show()