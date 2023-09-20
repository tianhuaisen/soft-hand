# -*- coding: utf-8 -*-

from torch.nn import Module
from model_train.model.Feedforward import Feedforward
from model_train.model.MultiHeadAttention import MultiHeadAttention

class Encoder(Module):
    def __init__(self, q, v, h, d_model, d_hidden, dropout=0.1):
        super().__init__()

        self.mha = MultiHeadAttention(d_model=d_model, q=q, v=v, h=h, dropout=dropout)
        self.feedforward = Feedforward(d_model=d_model, d_hidden=d_hidden, dropout=dropout)

    def forward(self, x):

        x, score = self.mha(x)
        x = self.feedforward(x)

        return x,  score