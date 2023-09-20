# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):

    def __init__(self, q, v, h, d_model, dropout=0.1):
        super().__init__()

        self.d_model = d_model

        self.w_q = nn.Linear(d_model, q * h)
        self.w_k = nn.Linear(d_model, q * h)
        self.w_v = nn.Linear(d_model, v * h)
        self.w_out = torch.nn.Linear(v * h, d_model)

        self.q = q
        self.h = h

        self.dropout = torch.nn.Dropout(p=dropout)
        self.layer_norm = torch.nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, x):

        residual = x

        q = torch.cat(self.w_q(x).chunk(self.h, dim=-1), dim=0)
        k = torch.cat(self.w_k(x).chunk(self.h, dim=-1), dim=0)
        v = torch.cat(self.w_v(x).chunk(self.h, dim=-1), dim=0)

        score = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.q)
        score = F.softmax(score, dim=-1)
        attention = torch.matmul(score, v)
        attention_heads = torch.cat(attention.chunk(self.h, dim=0), dim=-1)
        self_attention = self.w_out(attention_heads)

        x = self.dropout(self_attention)
        x = self.layer_norm(x + residual)

        return x, score