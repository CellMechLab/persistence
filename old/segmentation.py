import matplotlib.pyplot as plt
import numpy as np
from skimage import io,color,morphology,measure,filters
from scipy import ndimage
from scipy.cluster.vq import vq, kmeans2
from sklearn.cluster import DBSCAN
#from structure_tensor import eig_special_2d, structure_tensor_2d

blank = plt.imread('2x/Blank/20221209_172045.jpg')

fnames=[]
fnames.append( '2x/Rb/20221209_165815.jpg')
fnames.append( '2x/Na/20221209_170719.jpg')
fnames.append( '2x/Cs/20221209_165247.jpg')
fnames.append( '2x/K/20221209_170131.jpg')
fnames.append( '2x/Li/20221209_171852.jpg')

start = [300,300]
stop = [2900,4200]

start = [0,0]
stop = [-1,-1]

def getIslands(source,nclusters = 2):    
    features = source.reshape(source.shape[0]*source.shape[1],source.shape[2])/255
    codebook,distorsion = kmeans2(features,nclusters)
    codes,backdistorsion = vq(features, codebook)
    islands = codes.reshape(source.shape[0],source.shape[1])
    return islands

def getSpectral(mask):
    features = np.array(np.where(mask))
    db = DBSCAN(eps=150,min_samples=20).fit(features)
    labels = db.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    newmap = np.zeros_like(mask)
    print("Estimated number of clusters: %d" % n_clusters_)
    print("Estimated number of noise points: %d" % n_noise_)
    newmap[features]=clus
    return mask

def clean(segmented):
    cleaned = morphology.binary_dilation(segmented,morphology.disk(3))
    cleaned = morphology.remove_small_holes(cleaned,500)
    cleaned = morphology.remove_small_objects(cleaned,5000)
    frame = 300
    cleaned[:frame,:]=False
    cleaned[:,:frame]=False
    cleaned[:,-frame:]=False
    cleaned[-frame:,:]=False    
    return cleaned

def skinny(label_image):
    obj = []
    num=[]
    for i in range(np.max(label_image)+1):
        skel = label_image==i
        x,y = np.where(skel)
        area = np.sum(skel)
        length = np.sqrt( (np.min(x)-np.max(x))**2 * (np.min(y)-np.max(y))**2)
        obj.append(int(length/area))    
        num.append(i)
    iFil = num[np.argmax(obj)]
    filament = label_image==iFil
    return filament

def prepare(image):
    return image

images = []
for fname in fnames:
    im = plt.imread(fname)
    #im=filters.butterworth(color.rgb2gray(im))
    images.append(im)
    
#fig,axs = plt.subplots(clusters+1,5,sharex=True,sharey=True)
fig,axs = plt.subplots(5,5,sharex=True,sharey=True)

#fig,axs = plt.subplots(2,3,sharex=True,sharey=True)
#axs[0][0].imshow(images[2])
#axs[0][1].imshow(prepare(images[2]))

#axs[1][0].imshow(images[2][:,:,0],cmap='gray')
#axs[1][1].imshow(images[2][:,:,1],cmap='gray')
#axs[1][2].imshow(images[2][:,:,2],cmap='gray')

clustering = 'spectral'

for i in range(5):
    print(f'Image {i+1}')        
    axs[0][i].imshow(images[i])
    if len(images[i].shape)==3:
        #if clustering == 'kmeans':
        islands = getIslands(images[i],2)
        area=[]
        for j in range(2):
            area.append(np.sum(islands==j))
            #axs[j+1][i].imshow(islands==j,cmap='binary')
        jok = np.argmin(area)
        filament = islands==jok
        #islands = getSpectral(images[i])
            #axs[4][i].imshow(islands)
            #filament =   islands==1      
        axs[1][i].imshow(filament,cmap='binary')
        cleaned = clean(filament)
        axs[2][i].imshow(cleaned,cmap='binary')
        labels = measure.label(cleaned)
        skin = skinny(labels)
        axs[3][i].imshow(skin,cmap='binary')
        islands2 = getSpectral(filament)
        axs[4][i].imshow(islands2)
        
plt.show()
