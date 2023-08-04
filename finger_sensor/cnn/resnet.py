# -*- coding: utf-8 -*-

"""
@Time        : 2023/7/23
@Author      : PC
@File        : resnet
@Description : 
"""

from torchvision import models
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from time import time
from tqdm import tqdm
import os
from model_train.params import Params
from model_train.model.loss import loss
from utils.visualization import result_visualization
import dill


reslut_figure_path = '../model_train/result_figure'
file_name = 'resnet'
save_model_path = '../model_train/saved_model/motive 90.28 batch=100.pkl'



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
d_channel = params.d_channel
d_output = params.d_output
dropout = params.dropout
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'use device: {DEVICE}')


with open('../data_set/train_data/train_data.pkl','rb') as f:
    train_dataset = dill.load(f)
with open('../data_set/test_data/test_data.pkl','rb') as f:
    test_dataset = dill.load(f)
train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE,
                             shuffle=True)
DATA_LEN = len(train_dataset)
d_input = len(train_dataset[0]['sensor'])



net = models.resnet101(weights=None)
load_model = torch.load('../cnn/pre_model/resnet101.pth')
net.load_state_dict(load_model)
net.fc = torch.nn.Linear(2048, 31)
net = net.to(DEVICE)



loss_function = loss()
optimizer_name = 'Adagrad'
if optimizer_name == 'Adagrad':
    optimizer = optim.Adagrad(net.parameters(), lr=LR)
elif optimizer_name == 'Adam':
    optimizer = optim.Adam(net.parameters(), lr=LR)



correct_on_train = []
correct_on_test = []
loss_list = []
time_cost = 0


def test(dataloader, flag='test_set'):
    correct = 0
    total = 0
    with torch.no_grad():
        net.eval()
        for _, s in enumerate(dataloader):
            x, y = s['sensor'].float().to(DEVICE), s['label'].float().to(DEVICE)
            y_pre = net(x)
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
            y_pre = net(x)

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