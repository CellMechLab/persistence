import matplotlib.pyplot as plt
import numpy as np
from skimage import io,color

from functions import *

#fname = '2x/Rb/20221209_165815.jpg'
#fname = '2x/Na/20221209_170719.jpg'
#fname = '2x/Cs/20221209_165247.jpg'
#fname = '2x/K/20221209_170131.jpg'
fname = '2x/Li/20221209_171852.jpg'

source = io.imread(fname)

print('getting the filament')

skinny = getSkinny(source,debug=False)

fig,axs = plt.subplots(1,3,sharex=True,sharey=True)
axs[0].imshow(source)
axs[1].imshow(color.rgb2gray(source),cmap='gray')
axs[2].imshow(skinny,cmap='binary')

print('crawling over it')

#start/end points
step = 20
radius = 2
try:
    L= int(skinny.shape[0]/2)
    M = np.where(skinny[L,:])[0][0]
    start = [L, M]
    x1 = [M]
    y1 = [L]
    x1,y1 = crawl(x1,y1,-1,-1,skinny,step,radius)
    x2 = [M]
    y2 = [L]
    x2,y2 = crawl(x2,y2,1,1,skinny,step,radius)
    x = np.append(x2[::-1] ,x1[1:])
    y = np.append(y2[::-1] ,y1[1:])
    axs[1].plot(x,y, 'r-')
except:
    print('Please re-run, I think the seeds were not good!')
    pass
plt.show()