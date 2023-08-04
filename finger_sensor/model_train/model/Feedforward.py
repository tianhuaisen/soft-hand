# -*- coding: utf-8 -*-
from torch.nn import Module
import torch
import torch.nn.functional as F

class Feedforward(Module):
    def __init__(self, d_model, d_hidden=512, dropout=0.1):
        super().__init__()

        self.linear_1 = torch.nn.Linear(d_model, d_hidden)
        self.linear_2 = torch.nn.Linear(d_hidden, d_model)
        self.dropout = torch.nn.Dropout(p=dropout)
        self.layer_norm = torch.nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, x):

        residual = x

        x = self.linear_2(F.relu(self.linear_1(x)))
        x = self.dropout(x)
        x = self.layer_norm(x + residual)

        return x


class CWforward(Module):
    def __init__(self, d_model, d_hidden=512, dropout=0.1):
        super().__init__()

        self.linear_1 = torch.nn.Linear(d_model, d_hidden)
        self.linear_2 = torch.nn.Linear(d_hidden, d_model)
        # self.linear_3 = torch.nn.Linear(d_hidden*2, d_model)
        self.dropout = torch.nn.Dropout(p=dropout)
        self.layer_norm = torch.nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, x):

        residual = x

        x = self.linear_2(F.relu(self.linear_1(x)))
        x = torch.sigmoid(x)
        x = self.dropout(x)
        x = self.layer_norm(x + residual)

        return x