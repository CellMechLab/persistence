import numpy as np
from skimage import morphology,measure,filters
from scipy.cluster.vq import vq, kmeans2

def segment(image):
    channel = image[:,:,1] #using the green channel
    #threshold masking
    mask = channel>filters.threshold_mean(channel) #mean filter does not miss too much
    #cleaning small structures and filling holes
    cleaned = morphology.remove_small_holes(mask,500)
    cleaned = morphology.remove_small_objects(cleaned,5000)
    return cleaned

def select(mask):
    label_image = measure.label(mask)
    obj=[]
    for i in range(np.max(label_image)+1):
        skel = label_image==i
        area = np.sum(skel)
        x,y = np.where(skel)
        length = np.sqrt( (np.min(x)-np.max(x))**2 * (np.min(y)-np.max(y))**2)
        obj.append(int(length/area))    
        iFil = np.argmax(obj)
    return label_image==iFil

def isolate(filament,src):
    pointer = np.where(filament)
    features = src[pointer[0],pointer[1],:]*1.0
    nclusters = 2
    codebook,distorsion = kmeans2(features,nclusters)
    codes,backdistorsion = vq(features, codebook)    
    islands = np.zeros((src.shape[0],src.shape[1]))
    islands[pointer[0],pointer[1]]=codes+1
    if np.sum(islands==1)>np.sum(islands==2):
        return islands==1
    else:
        return islands==2
    
def skinny(islands):
    cleaned = morphology.remove_small_holes(islands,500)
    cleaned = morphology.binary_closing(cleaned,footprint = morphology.star(10))
    cleaned = morphology.remove_small_objects(cleaned,5000)
    return morphology.skeletonize(cleaned)


