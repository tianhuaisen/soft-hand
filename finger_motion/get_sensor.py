# -*- coding: utf-8 -*-

import serial
import time
import serial.tools.list_ports

def openSer():
    # Configure the basic parameters of the serial port and establish communication
    ser = serial.Serial("com6", 115200, timeout=0.01) # stopbits=1
    return ser


# Determine if a number is inside a string
def is_number(s):
    n = len(s)
    for i in range(n):
        try:
            float(s[i])
        except ValueError:
            return False
    return True

# Getting data printed directly from the serial port
def getSerdata(ser):
    time.sleep(0.05)
    count = ser.inWaiting()  # Returns the number of bytes in the receive cache

    # Receipt of data
    if count > 0:
        data = ser.read_all()
        data = str(data)[2:17]  # Range of data to be obtained by number of ADCs
        data = data.split(',')
        if is_number(data):
            return data
        else:
            return []

if __name__ == '__main__':
    """Create a buffer to remove the none value"""
    # # get data array
    # ser = openSer()
    # alldata = []
    # while True:
    #     data = getSerdata(ser)
    #     if data != None:
    #         alldata.extend(data)
    #         if len(alldata) == 32:
    #             alldata = list(map(float, alldata))
    #             print(alldata)
    #             alldata = []


    # get one data
    ser = openSer()
    while True:
        data = getSerdata(ser)
        if data != None and len(data):
            data = list(map(float, data))
            data = [data[0], data[1]]
            print(data)