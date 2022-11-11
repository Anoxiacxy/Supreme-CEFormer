import numpy as np
import torch
from torch import nn
from numpy import random
from .basic_sampler import BasicSampler


class RandomSampler(BasicSampler):
    def __init__(self, num_sampled, **kwargs):
        super(RandomSampler, self).__init__()
        self.num_sampled = num_sampled

    def forward(self, q, *args, **kwargs):
        batch_size, num_head, seq_length, dims = q.size()

        permutation = np.random.permutation(np.arange(0, seq_length - 1))
        permutation = torch.from_numpy(permutation).type_as(q)

        index = torch.cat([torch.zeros_like(permutation[..., :1]), permutation + 1], dim=-1)
        token_mask = (index <= (self.num_sampled + 1)).unsqueeze(0).repeat(batch_size, 1)

        return token_mask
