import numpy as np
import torch
from torch.utils.data import Dataset
import pickle
from model_train.params import Params
import os
import dill

params = Params()
np.set_printoptions(threshold=1000000000000000)

class SensorDataset(Dataset):
	def __init__(self, params):
		self.resistance, self.resistance_raw, self.label = [], [], []
		self.params = params

	def addFile(self, file):
		with open(file, 'rb') as f:
			data = pickle.load(f)

		#  Values and labels
		cur_resistance = data['sensor']
		cur_label = data['label']

		self.resistance.extend(cur_resistance)
		self.resistance_raw.extend(cur_resistance)
		self.label.extend(cur_label)

	def parseData(self):
		self.normalization()
		self.rawNormalization()
		self.slidingTimeWindow(self.params.window_size)


	def normalization(self):
		self.resistance = np.array(self.resistance)
		self.min_res = np.amin(self.resistance)
		self.max_res = np.amax(self.resistance)
		self.resistance = (self.resistance-self.min_res)/(self.max_res - self.min_res)


	def rawNormalization(self):
		self.resistance_raw = np.array(self.resistance_raw)
		self.min_res = np.amin(self.resistance_raw)
		self.resistance_raw = (self.resistance_raw-self.min_res)


	def slidingTimeWindow(self, window_size):

		total_data = {'sensor': [], 'label': []}
		sensor, rawsensor = [], []

		for i in range(int((len(self.resistance) - window_size)/2)):
			sensor.append(self.resistance[2*i: 2*i+window_size])
			rawsensor.append(self.resistance_raw[2*i: 2*i+window_size])
			total_data['label'].append(self.label[2*i: 2*i+window_size])

		sensor = torch.tensor(np.array(sensor)).unsqueeze(-1)
		rawsensor = torch.tensor(np.array(rawsensor)).unsqueeze(-1)
		new_sensor = torch.cat([sensor, rawsensor], dim = -1)

		# This code is the input Size needed for resnet.
		# sensor = torch.tensor(np.array(sensor)).repeat(1,32)
		# print(sensor.shape)
		# sensor = sensor.reshape(sensor.shape[0],32,32).unsqueeze(-1).reshape(sensor.shape[0], 1, 32, 32)
		# rawsensor = torch.tensor(np.array(rawsensor)).repeat(1,32)
		# print(rawsensor.shape)
		# rawsensor = rawsensor.reshape(rawsensor.shape[0], 32, 32).unsqueeze(-1).reshape(rawsensor.shape[0], 1, 32, 32)
		# new_sensor = torch.cat([sensor, rawsensor, sensor], dim = -3)    # 原为dim = -1

		total_data['sensor'] = new_sensor.tolist()
		print('Data is ready')

		for key in total_data.keys():
			total_data[key] = np.array(total_data[key])

		self.total_data = total_data

	def __len__(self):
		return len(self.total_data['sensor'])

	def __getitem__(self, idx):
		item = {'sensor': self.total_data['sensor'][idx], 'label': self.total_data['label'][idx]}
		return item


# Labeling a piece of timing data
def reLabel(sensordata):

	all_data = {'sensor': [], 'label': []}
	all_data['sensor'] = np.array(sensordata.total_data['sensor'])

	for i in range(len(sensordata.total_data['label'])):
		max = np.max(sensordata.total_data['label'][i])
		if np.sum(sensordata.total_data['label'][i] == max) >= params.window_size/4:
			all_data['label'].append(max)
		else:
			all_data['label'].append(0)
	all_data['label'] = np.array(all_data['label'])
	sensordata.total_data = all_data

	return sensordata

# External interface function
def myDataset(stage):
	sensorDataset = SensorDataset(params)

	if stage == 'train':
		for file_name in os.listdir(params.train_data_dir):
			file_path = os.path.join(params.train_data_dir, file_name)
			if os.path.isfile(file_path):
				sensorDataset.addFile(file_path)
		sensorDataset.parseData()

	elif stage == 'test':
		for file_name in os.listdir(params.test_data_dir):
			file_path = os.path.join(params.test_data_dir, file_name)
			if os.path.isfile(file_path):
				sensorDataset.addFile(file_path)
		sensorDataset.parseData()

	elif stage == 'data':
		for file_name in os.listdir(params.data_dir):
			file_path = os.path.join(params.data_dir, file_name)
			if os.path.isfile(file_path):
				sensorDataset.addFile(file_path)
		sensorDataset.parseData()

	else:
		print("Please choose 'train' data or 'test' data!")
		return False

	sensorDataset = reLabel(sensorDataset)
	return sensorDataset

if __name__ == '__main__':

	print("==========================================")
	dataset = myDataset('data')
	train_size = int(0.9 * len(dataset))
	test_size = len(dataset) - train_size
	train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size,test_size])
	print(len(train_dataset))
	print(len(test_dataset))
	print(len(train_dataset[0]['sensor']))
	print(test_dataset[0])

	# with open('../data_set/train_data/train_data.pkl', 'wb') as f:
	# 	dill.dump(train_dataset, f)
	#
	# with open('../data_set/test_data/test_data.pkl', 'wb') as f:
	# 	dill.dump(test_dataset, f)