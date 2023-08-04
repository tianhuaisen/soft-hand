# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import os

np.set_printoptions(threshold=1000000000000000)

def normalization(data):
	min = np.min(data)
	range = np.max(data) - min
	return (data - min) / range

# Data reduction
def shorten_data(data):
	new_data = []
	for i in range(len(data)):
		if i % 2 == 0:
			new_data.append(data[i])
	return np.array(new_data)

# Label data
def label_data(data, number):
	label = []
	new_data = normalization(data)
	mean = np.mean(new_data)
	n = len(data)
	for i in range(2):
		label.append(0)
	for i in np.arange(2, n-2, 1):
		if abs(new_data[i-2]-new_data[i+2])>0.1 and abs(new_data[i]-mean)>0.1:
			label.append(number)
		else:
			label.append(0)
	for i in range(2):
		label.append(0)

	for j in range(5):
		for i in np.arange(1, n - 1, 1):
			if label[i-1]==number and label[i+1]==number:
				label[i] = number
	for j in range(3):
		for i in np.arange(0, n - 1, 1):
			if label[i+1]==number:
				label[i] = number
	for j in range(3):
		for i in np.arange(n-1, 1, -1):
			if label[i-1]==number:
				label[i] = number

	return label

def csv_to_pkl():
	file_name = []
	file_label = []
	save_data = {'sensor': [], 'label': []}
	path = 'E:/my_git/all_finger_data/test_data_csv/'

	files = os.listdir(path)
	for file in files:
		file_path = os.path.join(path, file)
		file_name.append(file_path)
		file_label.append(file[:-4])

	plt.figure(figsize=(18, 2))
	i = 29
	print(file_label)
	print(file_label[i])
	data = pd.read_csv(file_name[i])
	data = np.array(data['TBS1202C'][130:2010].values.tolist()).astype(np.float32)

	# to pkl
	data = shorten_data(data)
	save_data['sensor'] = data
	save_data['label'] = label_data(data, i+1)
	save_path = 'E:/my_git/all_finger_data/test_data_pkl/' + file_label[i] + '.pkl'
	with open(save_path, 'wb') as f:
		pickle.dump(save_data, f)
	x = np.arange(len(data))
	plt.plot(x, save_data['sensor'])
	plt.plot(x, save_data['label'])
	plt.show()


def view_data(path):
	# data = normalization(pd.read_pickle(path))
	# data = normalization(pd.read_pickle(path)['sensor'])

	data_all = pd.read_pickle(path)
	print(data_all)
	print(len(data_all['sensor']))
	print(len(data_all['label']))
	data = pd.read_pickle(path)['sensor']

	x = np.arange(len(data))
	plt.plot(x, data)
	plt.show()


if __name__ == '__main__':

	path = 'E:/my_git/all_finger_data/test_data_pkl/z_test.pkl'
	# data = pd.read_pickle(path)['sensor']
	# print(data)
	view_data(path)

	# csv_to_pkl()
