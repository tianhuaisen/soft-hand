# -*- coding: utf-8 -*-
import torch
import numpy as np
import multiprocessing
import keyboard
from data_set.get_sensor import *
from data_set.Fetch_data import *
from pyqtshow.Finger_Main import *
import time

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
save_model_path = '../model_train/saved_model/motive 90.28 batch=100.pkl'
net = torch.load(save_model_path)
board = ['0', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j',
         'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
         'u', 'v', 'w', 'x', 'y', 'z', 'down', 'left', 'right', 'up']
ds = DataStore()

def test_single(data):

    with torch.no_grad():
        net.eval()
        y_pre, _, _, _, _, _ = net(data, DEVICE)
        pred = torch.argmax(y_pre).item()
        if pred==0:
            keyboard.press('enter')
            print('result: ' + '0')
            commands()
        elif pred==1:
            keyboard.press(board[1])
            print('result: ' + 'a')
            commands()
        elif pred==2:
            keyboard.press(board[2])
            print('result: ' + 'b')
            commands()
        elif pred==3:
            keyboard.press(board[3])
            print('result: ' + 'c')
            commands()
        elif pred==4:
            keyboard.press(board[4])
            print('result: ' + 'd')
            commands()
        elif pred==5:
            keyboard.press(board[5])
            print('result: ' + 'e')
            commands()
        elif pred==6:
            keyboard.press(board[6])
            print('result: ' + 'f')
            commands()
        elif pred==7:
            keyboard.press(board[7])
            print('result: ' + 'g')
            commands()
        elif pred==8:
            keyboard.press(board[8])
            print('result: ' + 'h')
            commands()
        elif pred==9:
            keyboard.press(board[9])
            print('result: ' + 'i')
            commands()
        elif pred==10:
            keyboard.press(board[10])
            print('result: ' + 'j')
            commands()
        elif pred==11:
            keyboard.press(board[11])
            print('result: ' + 'k')
            commands()
        elif pred==12:
            keyboard.press(board[12])
            print('result: ' + 'l')
            commands()
        elif pred==13:
            keyboard.press(board[13])
            print('result: ' + 'm')
            commands()
        elif pred==14:
            keyboard.press(board[14])
            print('result: ' + 'n')
            commands()
        elif pred==15:
            keyboard.press(board[15])
            print('result: ' + 'o')
            commands()
        elif pred==16:
            keyboard.press(board[16])
            print('result: ' + 'p')
            commands()
        elif pred==17:
            keyboard.press(board[17])
            print('result: ' + 'q')
            commands()
        elif pred==18:
            keyboard.press(board[18])
            print('result: ' + 'r')
            commands()
        elif pred==19:
            keyboard.press(board[19])
            print('result: ' + 's')
            commands()
        elif pred==20:
            keyboard.press(board[20])
            print('result: ' + 't')
            commands()
        elif pred==21:
            keyboard.press(board[21])
            print('result: ' + 'u')
            commands()
        elif pred==22:
            keyboard.press(board[22])
            print('result: ' + 'v')
            commands()
        elif pred==23:
            keyboard.press(board[23])
            print('result: ' + 'w')
            commands()
        elif pred==24:
            keyboard.press(board[24])
            print('result: ' + 'x')
            commands()
        elif pred==25:
            keyboard.press(board[25])
            print('result: ' + 'y')
            commands()
        elif pred==26:
            keyboard.press(board[26])
            print('result: ' + 'z')
            commands()
        elif pred==27:
            keyboard.press('down')
            print('result: ' + 'down')
            commands()
        elif pred==28:
            keyboard.press('esc')
            print('result: ' + 'left')
            commands()
        elif pred==29:
            keyboard.press('shift')
            keyboard.press('f5')
            print('result: ' + 'right')
            commands()
        else:
            keyboard.press('up')
            print('result: ' + 'up')
            commands()

def commands():
    time.sleep(0.5)
    print('========Start to acquire hand signals, please write')


def normalization(resistance):
    resistance = np.array(resistance)
    min_res = np.amin(resistance)
    max_res = np.amax(resistance)
    resistance = (resistance - min_res) / (
                max_res - min_res)
    return resistance


def rawNormalization(resistance_raw):
    resistance_raw = np.array(resistance_raw)
    min_res = np.amin(resistance_raw)
    resistance_raw = (resistance_raw - min_res)
    return resistance_raw


def read_from_sensor():
    global ds
    new_p1 = multiprocessing.Process(target=Wave_run, args=(ds,))
    new_p1.start()

    all_data = np.array([])
    ser = openSer()
    print('========Start to acquire hand signals, please write')
    while True:
        data = getSerdata(ser)
        if data != None and len(data):
            data = float(data[0])
            ds.set_data(data)
            all_data = np.append(all_data, data)
            if len(all_data) == 32:
                sensor = torch.tensor(normalization(all_data)).reshape(1,32,1)
                rawsensor = torch.tensor(rawNormalization(all_data)).reshape(1,32,1)
                new_sensor = torch.cat([sensor, rawsensor], dim=-1)
                new_sensor = new_sensor.float().to(DEVICE)
                test_single(new_sensor)
                all_data = np.array([])

# sichuan university
if __name__ == '__main__':
    read_from_sensor()


