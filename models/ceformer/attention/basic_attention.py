import torch
from torch import nn


class BasicAttention(nn.Module):
    def __init__(self):
        super(BasicAttention, self).__init__()

    def forward(self, q, k, v, **kwargs):
        raise NotImplementedError()
