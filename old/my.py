import matplotlib.pyplot as plt
import numpy as np
from skimage import io,morphology,color,measure
from scipy.cluster.vq import vq, kmeans2
from scipy.interpolate import splprep, splev, interp1d, splrep
from scipy.signal import savgol_filter,medfilt

source = io.imread('2x/Cs/20221209_165815.jpg')
filament = np.load('fil.npy')
skinny = morphology.skeletonize(filament)
    
fig,axs = plt.subplots(1,3,sharex=True,sharey=True)
axs[0].imshow(source)
axs[1].imshow(filament,cmap='binary')
axs[1].imshow(skinny,alpha=0.5)
axs[2].imshow(color.rgb2gray(source),cmap='gray')

start = [1420,3694]
stop = [1600,1067]

def dtest(ix,iy,ar,dist):
    try:
        if dist == 0:
            return ar[ix,iy]==1
        if dist ==1:
            tot = ar[ix,iy]+ar[ix+1,iy]+ar[ix,iy+1]+ar[ix-1,iy]+ar[ix,iy-1]
            tot += ar[ix-1,iy-1]
            tot += ar[ix-1,iy+1]
            tot += ar[ix+1,iy-1]
            tot += ar[ix+1,iy+1]
            return tot>0
    except IndexError:
        return False 

def intercept(p1,p2,map,ax):
    xm = int((p1[0]+p2[0])/2)
    ym = int((p1[1]+p2[1])/2)
        
    dx = p2[0]-p1[0]
    dy = p2[1]-p1[1]
    
    if dtest(ym,xm,map,1) is True:
        return xm,ym
        
    if np.abs(dy)>np.abs(dx):    
        if dy==0:
            return xm,ym
        mort = -dx/dy
        dist = 0
        i = 0
        while True:
            err = 0
            if (xm+i)<map.shape[1]:
                px = xm+i
                py = int(ym+mort*i)     
                #ax.plot(px,py,'go')       
                if dtest(py,px,map,dist):
                    break
            else:
                err+=1
            if(xm-i)>0:
                px = xm-i
                py = int(ym-mort*i)
                #ax.plot(px,py,'go')
                if dtest(py,px,map,dist):
                    break
            else:
                err += 1
            if err == 2:
                px = xm
                py = ym
                if dist==1:
                    break
                dist+=1
                i=0
            i+=1
    else:
        if dx==0:
            return xm,ym
        mort = -dy/dx
        i = 0
        dist = 0
        while True:
            err = 0
            if (ym+i)<map.shape[0]:
                py = ym+i
                px = int(xm+mort*i)
                #ax.plot(px,py,'yo')
                if dtest(py,px,map,dist):
                    break
            else:
                err += 1
            if (ym-i)>0:
                py = ym-i
                px = int(xm-mort*i)
                #ax.plot(px,py,'yo')
                if dtest(py,px,map,dist):
                    break
            else:
                err += 1
            if err == 2:
                px = xm
                py = ym
                if dist==1:
                    break
                dist+=1
                i=0
            i+=1
    return px,py

x = [start[0],stop[0]]
y = [start[1],stop[1]]

N = 5
for i in range(N-1):
    newx = []
    newy = []
    for j in range(len(x)-1):    
        newx.append(x[j])
        newy.append(y[j])
        newpoint = intercept([x[j],y[j]],(x[j+1],y[j+1]),skinny,axs[2])
        newx.append(newpoint[0])            
        newy.append(newpoint[1])            
                
    newx.append(x[-1])
    newy.append(y[-1])
    x=newx
    y=newy
    
axs[2].plot(x,y, 'r-')
plt.show()