# -*- coding: utf-8 -*-
import numpy as np
from pyqtshow.Finger_Main import *
import multiprocessing

class DataStore():
    def __init__(self):
        self.data = multiprocessing.Value('f', 0.0)
        self.rawdata = np.array([])
        self.noldata = np.array([])

    def set_data(self, data):
        self.data.value = data
        # self.rawdata = np.array(self.data)
        # self.noldata = self.normalization(self.data)

    def normalization(self, data):
        min = np.min(data)
        range = np.max(data) - min
        return (data - min) / range

    def getdata(self):
        return self.data.value

    def getrawdata(self):
        return self.rawdata

    def getnodata(self):
        return self.noldata

if __name__ == '__main__':
    ds = DataStore()
    Wave_run(ds)