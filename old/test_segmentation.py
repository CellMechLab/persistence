
import matplotlib.pyplot as plt
import numpy as np
from skimage import morphology,measure,io,color,filters

import glob

images = glob.glob('./2x/Na/*.jpg')
nasty = './2x/Na/20221209_170814.jpg'
cleaning = True
magic = False
stop = False

for image in images:
    print(image)
    if stop is True:
        image = nasty
    src = io.imread(image)

    if magic is True:
        thresholds = filters.threshold_multiotsu(src[:,:,1])
        label_image = np.digitize(src[:,:,1], bins=thresholds)

    else:
        mask = src[:,:,1]>filters.threshold_mean(src[:,:,1])

        if cleaning is True:
            #cleaned = morphology.binary_dilation(mask,morphology.disk(3))
            cleaned = morphology.remove_small_holes(mask,500)
            cleaned = morphology.remove_small_objects(cleaned,5000)
        else:
            cleaned = mask

        label_image = measure.label(cleaned)
    colorised = color.label2rgb(label_image, bg_label=0)
    fig,axs=plt.subplots(1,2,sharex=True,sharey=True)
    fig.suptitle(image)
    axs[0].imshow(src)
    axs[1].imshow(colorised)

    if stop is True:
        break


plt.show()