# -*- coding: utf-8 -*-
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import sys
import numpy as np
import pyqtgraph as pg
from pyqtgraph import PlotWidget
from pyqtshow.Finger import Ui_Form
from data_set.Fetch_data import *

# Global setting
pg.setConfigOption('background', (255, 255, 255))
pg.setConfigOption('foreground', 'k')

class MyFigure(QWidget):
    def __init__(self, ds):
        super().__init__()
        self.ds = ds
        # self.resize(650, 640)

        self.win = pg.GraphicsLayoutWidget(self, size=(640, 630))
        self.plot1 = self.win.addPlot(row=0, col=0)
        self.plot2 = self.win.addPlot(row=1, col=0)
        self.plot2.setYRange(0.1, 0.7)
        self.plot3 = self.win.addPlot(row=2, col=0)
        self.plot4 = self.win.addPlot(row=3, col=0)
        self.plot5 = self.win.addPlot(row=4, col=0)

        # Adding a PlotWidget Control
        # self.plotWidget_ted = PlotWidget(self)

        # Set the size and relative position of the control
        # self.plotWidget_ted.setGeometry(QRect(5, 10, 620, 100))

        # generate 0
        self.data1 = np.zeros(160)
        self.data2 = np.zeros(160)
        self.data3 = np.zeros(160)
        self.data4 = np.zeros(160)
        self.data5 = np.zeros(160)
        # self.curve1 = self.plotWidget_ted.plot(self.data1, pen='d')

        self.curve1 = self.plot1.plot(self.data1, pen=pg.mkPen(color=(220,20,60), width=2))
        self.curve2 = self.plot2.plot(self.data1, pen=pg.mkPen(color=(154,205,50), width=2))
        self.curve3 = self.plot3.plot(self.data1, pen=pg.mkPen(color=(30,144,255), width=2))
        self.curve4 = self.plot4.plot(self.data1, pen=pg.mkPen(color=(0,206,209), width=2))
        self.curve5 = self.plot5.plot(self.data1, pen=pg.mkPen(color=(138,43,226), width=2))


        # Setting Timer
        self.timer = QTimer()
        # Timer signal binding update_data function
        self.timer.timeout.connect(self.update_data)
        # The timer interval is 50ms, which can be interpreted as refreshing the data once in 50ms.
        self.timer.start(50)

    # Data left shift
    def update_data(self):
        # self.data1[:-1] = self.data1[1:]
        self.data2[:-1] = self.data2[1:]
        # self.data3[:-1] = self.data3[1:]
        # self.data4[:-1] = self.data4[1:]
        # self.data5[:-1] = self.data5[1:]


        # self.data1[-1] = self.ds.getnodata()[0]
        self.data2[-1] = self.ds.getdata()
        # self.data3[-1] = self.ds.getnodata()[2]
        # self.data4[-1] = self.ds.getnodata()[3]
        # self.data5[-1] = self.ds.getnodata()[4]


        # Data Filling into Plotted Curves
        # self.curve1.setData(self.data1)
        self.curve2.setData(self.data2)
        # self.curve3.setData(self.data3)
        # self.curve4.setData(self.data4)
        # self.curve5.setData(self.data5)


class Finger_Main(QWidget, Ui_Form):
    def __init__(self, ds):
        super(Finger_Main, self).__init__()
        self.setupUi(self)

        self.F = MyFigure(ds)
        # Inherited from container groupBox
        self.gridlayout = QGridLayout(self.groupBox)
        self.gridlayout.addWidget(self.F)

def Wave_run(ds):
    app = QApplication(sys.argv)
    my_pyqt_form = Finger_Main(ds)
    my_pyqt_form.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    ds = DataStore()
    Wave_run(ds)

