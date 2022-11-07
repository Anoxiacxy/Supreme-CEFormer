from torch import nn

from .basic_sampler import BasicSampler


class RandomSampler(BasicSampler):
    def __init__(self, m, r):
        super(RandomSampler, self).__init__()
        self.r = r
        self.m = m

    def forward(self, q, k, **kwargs):
        """
        :param q: [..., SeqLen, Dims]
        :param k: [..., SeqLen, Dims]
        :return:
        """
        pass
