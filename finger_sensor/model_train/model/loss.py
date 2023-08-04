# -*- coding: utf-8 -*-

from torch.nn import Module
from torch.nn import CrossEntropyLoss

class loss(Module):
    def __init__(self):
        super(loss, self).__init__()
        self.loss_function = CrossEntropyLoss()

    def forward(self, y_pre, y_true):
        y_true = y_true.long()
        loss = self.loss_function(y_pre, y_true)

        return loss