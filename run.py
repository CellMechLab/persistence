from magicgui import magicgui
from magicgui.widgets import FileEdit,FloatSpinBox, Container, PushButton,Label
from skimage import io,color,filters,morphology,measure
import matplotlib.pyplot as plt
import numpy as np

label = Label()
line_edit = FileEdit('r')
spin_box = FloatSpinBox(value=0,min=0.0,max=1.0,step=0.01)
btn = PushButton(text='View Mask') #,'Mask',bind=calcmask)

image = None
channel = None

plt.ion()
fig,axs = plt.subplots(2,2)
plt.show()

def calcmask():
  print('ciao')
  print (line_edit.value)
btn.changed.connect(calcmask)

def openimage():
  global image
  global channel
  name = line_edit.value
  try:
    image = io.imread(name)
    channel = color.rgb2gray(image)
    axs[0][0].imshow(image,cmap='gray')
    label.value = 'File opened'
    spin_box.max = np.max(channel)
    spin_box.min = np.min(channel)
    spin_box.value = filters.threshold_isodata(channel)
  except:
    label.value = 'Error opening the file'

def plotmask():
  axs[0][1].imshow(channel,cmap='gray')
  mask = channel>spin_box.value
  axs[0][1].imshow(mask,alpha=0.5)
  plotregions()

def plotregions():
  mask = channel>spin_box.value
  cleaned = morphology.remove_small_holes(mask,500)
  cleaned = morphology.remove_small_objects(cleaned,5000)
  label_image = measure.label(cleaned)
  axs[1][0].imshow(color.label2rgb(label_image))

line_edit.changed.connect(openimage)
spin_box.changed.connect(plotmask)

container = Container(widgets=[label,line_edit, spin_box,btn])
container.show(run=True)
