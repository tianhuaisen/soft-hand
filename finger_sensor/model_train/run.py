# -*- coding: utf-8 -*-

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from time import time
from tqdm import tqdm
import os
from model_train.params import Params
from model_train.model.loss import loss
from utils.visualization import result_visualization
from model_train.model.Transformer import Transformer
import dill


# File Path Setting
reslut_figure_path = '../model_train/result_figure'
file_name = 'motive'
save_model_path = '../model_train/saved_model/motive 90.28 batch=100.pkl'


# Hyperparametric configuration
params = Params()
test_interval = params.test_interval
draw_key = params.draw_key
EPOCH = params.epoch
BATCH_SIZE = params.batch_size
LR = params.lr
d_model = params.d_model
d_hidden = params.d_hidden
q = params.q
v = params.v
h = params.h
N = params.n
d_channel = params.d_channel  # Time series dimension
d_output = params.d_output  # Number of out classes
dropout = params.dropout
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'use device: {DEVICE}')


# Load training dataset and test dataset
with open('../data_set/train_data/train_data.pkl','rb') as f:
    train_dataset = dill.load(f)
with open('../data_set/test_data/test_data.pkl','rb') as f:
    test_dataset = dill.load(f)
train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE,
                             shuffle=True)
DATA_LEN = len(train_dataset) # Number of samples in the training set
d_input = len(train_dataset[0]['sensor'])  # Number of time series dimension

print(f'Train data size: [{DATA_LEN, d_input, d_channel}]')
print(f'Number of classes: {d_output}')

# Creating a Dual_Transformer Model
net = Transformer(d_model=d_model, d_input=d_input, d_channel=d_channel,
                  d_output=d_output, d_hidden=d_hidden,
                  q=q, v=v, h=h, N=N, dropout=dropout).to(DEVICE)

# Loading existing models
# net = torch.load(save_model_path)

# Create a loss function. Cross entropy loss is used here.
loss_function = loss()
# Optimizer Selection
optimizer_name = 'Adam'
if optimizer_name == 'Adagrad':
    optimizer = optim.Adagrad(net.parameters(), lr=LR)
elif optimizer_name == 'Adam':
    optimizer = optim.Adam(net.parameters(), lr=LR)

# For recording changes in accuracy
correct_on_train = []
correct_on_test = []
# For recording changes in losses
loss_list = []
time_cost = 0



def test(dataloader, flag='test_set'):
    correct = 0
    total = 0
    with torch.no_grad():
        net.eval()
        for _, s in enumerate(dataloader):
            x, y = s['sensor'].float().to(DEVICE), s['label'].float().to(DEVICE)
            y_pre, _, _, _, _, _ = net(x, DEVICE)
            _, label_index = torch.max(y_pre.data, dim=-1)
            total += label_index.shape[0]
            correct += (label_index == y.long()).sum().item()
        if flag == 'test_set':
            correct_on_test.append(round((100 * correct / total), 2))
        elif flag == 'train_set':
            correct_on_train.append(round((100 * correct / total), 2))
        print(f'Accuracy on {flag}: %.2f %%' % (100 * correct / total))

        return round((100 * correct / total), 2)



def train():
    net.train()
    max_accuracy = 0
    pbar = tqdm(total=EPOCH)
    begin = time()
    for index in range(EPOCH):
        for i, s in enumerate(train_dataloader):
            optimizer.zero_grad()
            x, y = s['sensor'].float().to(DEVICE), s['label'].float().to(DEVICE)
            y_pre, _, _, _, _, _ = net(x, DEVICE)

            loss = loss_function(y_pre, y)

            print(f'Epoch:{index + 1}:\t\tloss:{loss.item()}')
            loss_list.append(loss.item())

            loss.backward()
            optimizer.step()

        if ((index + 1) % test_interval) == 0:
            current_accuracy = test(test_dataloader)
            test(train_dataloader, 'train_set')
            print(f'当前最大准确率\t测试集:{max(correct_on_test)}%\t 训练集:{max(correct_on_train)}%')

            if current_accuracy > max_accuracy:
                max_accuracy = current_accuracy
                torch.save(net, f'../model_train/saved_model/{file_name} batch={BATCH_SIZE}.pkl')

        pbar.update()

    os.rename(f'../model_train/saved_model/{file_name} batch={BATCH_SIZE}.pkl',
              f'../model_train/saved_model/{file_name} {max_accuracy} batch={BATCH_SIZE}.pkl')

    end = time()
    time_cost = round((end - begin) / 60, 2)

    # Result
    result_visualization(loss_list=loss_list, correct_on_test=correct_on_test, correct_on_train=correct_on_train,
                         test_interval=test_interval,
                         d_model=d_model, q=q, v=v, h=h, N=N, dropout=dropout, DATA_LEN=DATA_LEN, BATCH_SIZE=BATCH_SIZE,
                         time_cost=time_cost, EPOCH=EPOCH, draw_key=draw_key, reslut_figure_path=reslut_figure_path,
                         file_name=file_name, optimizer_name=optimizer_name, LR=LR)



if __name__ == '__main__':
    train()

