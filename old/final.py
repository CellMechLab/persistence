import matplotlib.pyplot as plt
import numpy as np
from skimage import io,morphology,color,measure
from scipy.cluster.vq import vq, kmeans2
from scipy.interpolate import splprep, splev, interp1d, splrep
from scipy.signal import savgol_filter,medfilt

source = io.imread('2x/Rb/20221209_165815.jpg')

filament = np.load('fil.npy')
skinny = morphology.skeletonize(filament)
    
fig,axs = plt.subplots(1,2,sharex=True)

wx,wy=np.where(skinny)
axs[0].plot(wx,'.')
axs[1].plot(wy,'.')
x = np.arange(len(wx))

wwx = medfilt(wx,101)
wwy = medfilt(wy,101)

xspl = interp1d(x, wwx,kind='cubic')
yspl = interp1d(x, wwy,kind='cubic')

#x2 = np.linspace(min(x), max(x), 1000)
axs[0].plot(x,xspl(x),'r-')
axs[1].plot(x,yspl(x),'r-')

fig,axs = plt.subplots(1,3,sharex=True,sharey=True)
axs[0].imshow(source)
axs[1].imshow(filament,cmap='binary')
axs[1].imshow(skinny,alpha=0.5)
axs[2].imshow(color.rgb2gray(source),cmap='gray')
axs[2].plot(yspl(x),xspl(x), 'r-')

plt.show()