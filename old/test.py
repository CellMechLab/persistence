
import matplotlib.pyplot as plt
import numpy as np
from skimage import morphology,measure,io,color,filters
import functions
import glob

from scipy.cluster.vq import vq, kmeans2

images = glob.glob('./2x/Cs/*.jpg')

cleaning = True
magic = False
stop = False

for image in images:
    print(image)
    src = io.imread(image)

    mask = functions.segment(src)
    firstguess = functions.select(mask)
    isolated = functions.isolate(firstguess,src)
    filament = functions.skinny(isolated)
    colorised = color.label2rgb(isolated, bg_label=0,image=src)
    
    step = 20
    radius = 2
    x = y = []
    try:
        L= int(filament.shape[0]/2)
        M = np.where(filament[L,:])[0][0]
        start = [L, M]
        x1 = [M]
        y1 = [L]
        x1,y1 = functions.crawl(x1,y1,-1,-1,filament,step,radius)
        x2 = [M]
        y2 = [L]
        x2,y2 = functions.crawl(x2,y2,1,1,filament,step,radius)
        x = np.append(x2[::-1] ,x1[1:])
        y = np.append(y2[::-1] ,y1[1:])        
    except:
        print('Please re-run, I think the seeds were not good!')
        pass

    fig,axs=plt.subplots(1,2,sharex=True,sharey=True) #1,2,sharex=True,sharey=True
    fig.suptitle(image)
    axs[0].imshow(colorised)
    axs[1].imshow(src)
    axs[1].plot(x,y, 'r-')

    if stop is True:
        break


plt.show()