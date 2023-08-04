# -*- coding: utf-8 -*-
import torch
from torch.nn import Module
from torch.nn import ModuleList
import torch.nn.functional as F
from model_train.model.Encoder import Encoder
from model_train.model.Feedforward import CWforward
from model_train.model.PositionEncoding import PositionalEncoding

class Transformer(Module):
    def __init__(self, d_model, d_input, d_channel, d_output, d_hidden, q, v, h, N, dropout=0.1):
        super().__init__()

        self.encoder_list = ModuleList([Encoder(d_model=d_model,
                                                  d_hidden=d_hidden,
                                                  q=q,
                                                  v=v,
                                                  h=h,
                                                  dropout=dropout) for _ in range(N)])

        self.encoder_list2 = ModuleList([Encoder(d_model=d_model,
                                                  d_hidden=d_hidden,
                                                  q=q,
                                                  v=v,
                                                  h=h,
                                                  dropout=dropout) for _ in range(N)])

        self.channel_list = ModuleList([CWforward(d_model=d_model,
                                                  d_hidden=d_hidden,
                                                  dropout=dropout) for _ in range(N)])

        self.embedding_channel = torch.nn.Linear(d_channel, d_model)
        self.embedding_input = torch.nn.Linear(d_input, d_model)
        self.embedding_input2 = torch.nn.Linear(d_input, d_model)

        self.weight = torch.nn.Linear(d_model * d_input + d_model * d_channel + d_model * d_channel, 3)
        self.output_linear = torch.nn.Linear(d_model * d_input + d_model * d_channel + d_model * d_channel, d_output)

        self.d_input = d_input
        self.d_model = d_model

    def forward(self, x, DEVICE):
        # x = x.unsqueeze(-1) # Add a dimension
        x = x.transpose(-1, -2)

        channelcoding = self.embedding_channel(x.transpose(-1, -2))
        channel_to_gather = channelcoding
        # self.channel_pe = PositionalEncoding(channelcoding.shape[1], self.d_model, DEVICE=DEVICE)
        # channelcoding = self.channel_pe(channelcoding)
        for channelcoder in self.channel_list:
            channelcoding = channelcoder(channelcoding)


        encoding= self.embedding_input(x)
        input_to_gather = encoding
        self.input_pe = PositionalEncoding(encoding.shape[1], self.d_model, DEVICE=DEVICE)
        encoding = self.input_pe(encoding)
        for encoder in self.encoder_list:
            encoding, score_channel = encoder(encoding)


        encoding2= self.embedding_input2(x)
        input_to_gather2 = encoding2
        for encoder in self.encoder_list2:
            encoding2, score_channel2 = encoder(encoding2)


        # 3 dim to 2 dim
        channelcoding = channelcoding.reshape(channelcoding.shape[0], -1)
        encoding = encoding.reshape(encoding.shape[0], -1)
        encoding2 = encoding2.reshape(encoding2.shape[0], -1)

        # get weight
        weight = F.softmax(self.weight(torch.cat([channelcoding, encoding, encoding2], dim=-1)), dim=-1)
        encoding_all = torch.cat([channelcoding * weight[:, 0:1], encoding * weight[:, 1:2], encoding2 * weight[:, 2:3]], dim=-1)

        # out
        output = self.output_linear(encoding_all)

        return output, encoding_all, score_channel, input_to_gather, channel_to_gather, weight


if __name__ == '__main__':
    d_model = 512
    d_hidden = 1024
    q = 8
    v = 8
    h = 8
    n = 8
    dropout = 0.2
    d_output = 31
    d_channel = 2
    d_input = 32
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = Transformer(d_model=d_model, d_input=d_input, d_channel=d_channel,
                      d_output=d_output, d_hidden=d_hidden,
                      q=q, v=v, h=h, N=n, dropout=dropout).to(DEVICE)
    x = torch.randn(50, 32, 2).to(DEVICE)
    y_pre, a, b, c, d, e = net(x, DEVICE)
    print(y_pre.shape)
    print(a.shape)
    print(b.shape)
    print(c.shape)
    print(d.shape)
    print(e.shape)