import sys
import numpy as np
import pyqtgraph as pg
import prepare_view as view
from PyQt5 import QtCore, QtGui, QtWidgets
from skimage import io,color,transform,morphology,measure,filters
from scipy.cluster.vq import vq, kmeans2

pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')


class NanoWindow(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        QtWidgets.QMainWindow.__init__(self, parent)

        self.ui = view.Ui_radius()
        self.ui.setupUi(self)

        # set plots style
        #self.curve_raw = pg.PlotCurveItem(clickable=False)
        #self.curve_raw.setPen(
        #    pg.mkPen(pg.QtGui.QColor(0, 255, 0, 255), width=1))
        #self.ui.g_single.plotItem.showGrid(True, True)
        #self.ui.g_single.plotItem.addItem(self.curve_raw)
        #self.curve_single = pg.PlotCurveItem(clickable=False)
        

        #self.ui.save.clicked.connect(self.saveJSON)

        self.workingdir = './'        

        self.ui.button_mask.clicked.connect(self.mask)
        self.ui.button_open.clicked.connect(self.open_folder)
        self.ui.button_kmeans.clicked.connect(self.machine)
        self.ui.button_skel.clicked.connect(self.skeleton)

        for radio in [self.ui.ch_red,self.ui.ch_green,self.ui.ch_blue,self.ui.ch_gray]:
            radio.clicked.connect(self.selectChannel)
        
        for imagebox in [self.ui.im_mask,self.ui.im_orig,self.ui.im_skinny]:
            imagebox.ui.histogram.hide()
            imagebox.ui.roiBtn.hide()
            imagebox.ui.menuBtn.hide()

        # connect load and open, other connections after load/open
        QtCore.QMetaObject.connectSlotsByName(self)

    def open_folder(self):
        fname = QtWidgets.QFileDialog.getOpenFileName(self,'Select the image to evaluate',self.workingdir)
        if fname == '' or fname is None or fname[0] == '':
            return        
        QtWidgets.QApplication.setOverrideCursor(QtGui.QCursor(QtCore.Qt.WaitCursor))
        image = io.imread(fname[0])
        self.image = image
        self.ui.im_orig.setImage(self.image)
        QtWidgets.QApplication.restoreOverrideCursor()
        self.selectChannel()
        
    def selectChannel(self):
        QtWidgets.QApplication.setOverrideCursor(QtGui.QCursor(QtCore.Qt.WaitCursor))
        if self.ui.ch_red.isChecked():
            self.channel = self.image[:,:,0]
        if self.ui.ch_green.isChecked():
            self.channel = self.image[:,:,1]
        if self.ui.ch_blue.isChecked():
            self.channel = self.image[:,:,2]
        else:
            self.channel = color.rgb2gray(self.image)
        self.ui.im_channel.setImage(self.channel)
        val = np.min(self.channel)+(np.max(self.channel)-np.min(self.channel))/2
        self.ui.threshold.setMaximum(int(np.max(self.channel)))
        self.ui.threshold.setMinimum(int(np.min(self.channel)))
        self.ui.threshold.setValue( int(filters.threshold_otsu(self.channel)))
        QtWidgets.QApplication.restoreOverrideCursor()

    def machine(self):
        QtWidgets.QApplication.setOverrideCursor(QtGui.QCursor(QtCore.Qt.WaitCursor))
        source = self.image
        nclusters= int(self.ui.n_clusters.value())        
        features = source.reshape(source.shape[0]*source.shape[1],source.shape[2])
        codebook,distorsion = kmeans2(features,nclusters)
        codes,backdistorsion = vq(features, codebook)
        self.islands = codes.reshape(source.shape[0],source.shape[1])
        self.ui.im_mask.setImage(color.label2rgb(self.islands))
        QtWidgets.QApplication.restoreOverrideCursor()
    
    def mask(self):
        QtWidgets.QApplication.setOverrideCursor(QtGui.QCursor(QtCore.Qt.WaitCursor))
        th = float(self.ui.threshold.value())
        mask = self.channel>th
        cleaned = morphology.remove_small_objects(mask,5000)
        cleaned = morphology.remove_small_holes(cleaned,500)        
        self.islands =  measure.label(cleaned)        
        self.ui.islands.setMaximum(int(np.max(self.islands)))
        self.ui.im_mask.setImage(color.label2rgb(self.islands))
        obj=[]
        for i in range(np.max(self.islands)+1):
            skel = self.islands==i
            area = np.sum(skel)
            x,y = np.where(skel)
            length = np.sqrt( (np.min(x)-np.max(x))**2 * (np.min(y)-np.max(y))**2)
            obj.append(int(length/area))    
            iFil = np.argmax(obj)
        self.ui.islands.setValue(iFil)
        QtWidgets.QApplication.restoreOverrideCursor()

    def skeleton(self):
        QtWidgets.QApplication.setOverrideCursor(QtGui.QCursor(QtCore.Qt.WaitCursor))
        iFil = self.ui.islands.value()
        cleaned = self.islands==iFil    
        np.savez('file.npz',cleaned)
        cleaned = morphology.binary_closing(cleaned,footprint = morphology.star(10))
        cleaned = morphology.remove_small_objects(cleaned,5000)
        self.skel = morphology.skeletonize(cleaned)
        label_image = measure.label(self.skel)
        self.ui.im_skinny.setImage(color.label2rgb(label_image))
        QtWidgets.QApplication.restoreOverrideCursor()
        

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName('Nano2021')
    app.setStyle('Fusion')
    chiaro = NanoWindow()
    chiaro.show()
    # QtCore.QObject.connect( app, QtCore.SIGNAL( 'lastWindowClosed()' ), app, QtCore.SLOT( 'quit()' ) )
    sys.exit(app.exec_())
