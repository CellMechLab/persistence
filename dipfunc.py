import numpy as np
from skimage import measure

def findnext(x,y,dx,dy,skinny,delta,R,limit=4,ax=None):
    xc = x + int(delta*dx/np.sqrt(dx**2+dy**2))
    yc = y + int(delta*dy/np.sqrt(dx**2+dy**2))
    if ax is not None:
        ax.plot(yc,xc, 'yo')     
    for i in range(limit):
        square = skinny[yc-R:yc+R,xc-R:xc+R]
        if np.sum(square)>0:
            break
        R *= 2
    else:
        raise ValueError
    dy,dx = measure.centroid(square)    
    return xc+int(dx-R),yc+int(dy-R)

def crawl(x,y,dx,dy,skinny,step,radius):
    while True:
        try:
            xnew,ynew = findnext(x[-1],y[-1],dx,dy,skinny,step,radius)           
        except ValueError:
            break
        dx = xnew-x[-1]
        dy = ynew-y[-1] 
        x.append(xnew)
        y.append(ynew)  
    return x,y