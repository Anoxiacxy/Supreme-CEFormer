import torch
from torch import nn


class TopSampler(nn.Module):
    def __init__(self, m, r):
        super(TopSampler, self).__init__()
        self.r = r
        self.m = m

    def forward(self, q, k):
        """
        :param q: [..., SeqLen, Dims]
        :param k: [..., SeqLen, Dims]
        :return:
        """
        a = torch.matmul(q[..., :1, :], k[..., 1:, :].transpose(-2, -1))[..., 0, :]
        _, index = torch.sort(a)
        index = torch.cat([torch.zeros_like(index[..., :1]), index + 1], dim=-1)
        return index <= self.r
