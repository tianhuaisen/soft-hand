# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    # n_position is the input_size, d_hid is input_length
    def __init__(self, n_position=32, d_hid=512, DEVICE='cuda:0'):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(n_position, d_hid).to(DEVICE)
        position = torch.arange(0, n_position, dtype=torch.float).unsqueeze(-1).to(DEVICE)
        div_term = torch.exp(torch.arange(0, d_hid, 2).float() * (-math.log(10000.0)/d_hid)).unsqueeze(0).to(DEVICE)
        temp = torch.matmul(position, div_term)# shape:[n_position, d_hid/2]

        pe[:, 0::2] = torch.sin(temp)
        pe[:, 1::2] = torch.cos(temp)# shape:[n_position, d_hid]

        self.pe = pe.to(DEVICE)

    def forward(self, x):

        return x + self.pe[:, :x.size(-1)]

