import math

import torch
from torch import nn


class NonNegativeF(nn.Module):
    def forward(self, x: torch.Tensor):
        return torch.exp(x) * (x < 0) + (x + 1) * (x >= 0)


class CosEmbedG(nn.Module):
    def __init__(self):
        super(CosEmbedG, self).__init__()

    def forward(self, x):
        n = x.shape[-2]
        return x * torch.cos(torch.pi / 2.0 / n * torch.arange(n).type_as(x)).unsqueeze(1)


class SinEmbedG(nn.Module):
    def __init__(self):
        super(SinEmbedG, self).__init__()

    def forward(self, x):
        n = x.shape[-2]
        return x * torch.sin(torch.pi / 2.0 / n * torch.arange(n).type_as(x)).unsqueeze(1)


class TopSample(nn.Module):
    def __init__(self, r):
        super(TopSample, self).__init__()
        self.r = r

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


class RandomSample(nn.Module):
    def __init__(self, r):
        super(RandomSample, self).__init__()
        self.r = r

    def forward(self, q, k):
        """
        :param q: [..., SeqLen, Dims]
        :param k: [..., SeqLen, Dims]
        :return:
        """
        pass


class DistributionSample(nn.Module):
    def __init__(self, r, replacement=False):
        super(DistributionSample, self).__init__()
        self.r = r
        self.replacement = replacement

    def forward(self, q, k):
        """
        :param q: [..., SeqLen, Dims]
        :param k: [..., SeqLen, Dims]
        :return: i: [..., SeqLen]
        """
        a = torch.matmul(q[..., :1, :], k[..., 1:, :].transpose(-2, -1))
        a = a / math.sqrt(q.size()[-1])
        a = nn.functional.softmax(a, dim=-1)[..., 0, :]
        a_shape = a.shape
        a = a.reshape(-1, a_shape[-1])
        with torch.no_grad():
            index = torch.multinomial(a, num_samples=self.r, replacement=self.replacement)
        index = index.reshape(*a_shape[:-1], self.r)
        index = torch.cat([torch.zeros(*a_shape[:-1], 1).type_as(index).int(), index + 1], dim=-1)
        logic = torch.zeros(*q.shape[:-1]).type_as(index).bool()
        logic.scatter_(dim=-1, index=index, src=torch.ones_like(index).bool())
        return logic


class ThinConvModule(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ThinConvModule, self).__init__()
        self.conv_7x7 = nn.Conv2d(in_channel, out_channel, kernel_size=(7, 7), stride=3, padding=1)
        self.batch_norm = nn.BatchNorm2d(out_channel)
        self.max_pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

    def forward(self, img: torch.Tensor):
        img = self.conv_7x7(img)
        img = self.batch_norm(img)
        img = self.max_pool(img)
        img = img.permute(0, 2, 3, 1)
        return img


