import torch
from torch import nn


class BasicSampler(nn.Module):
    def __init__(self):
        super(BasicSampler, self).__init__()

    def forward(
        self,
        q: torch.FloatTensor,
        k: torch.FloatTensor,
        v: torch.FloatTensor,
        token_mask: torch.BoolTensor
    ) -> torch.BoolTensor:
        """
        Sampler new tokens based on the last round and current qkv
        :param q: [Batch, Head, SeqLen, KDim]
        :param k: [Batch, Head, SeqLen, KDim]
        :param v: [Batch, Head, SeqLen, VDim]
        :param token_mask: [Batch, SeqLen]
        :return: new token mask [Batch, SeqLen]
        """
        raise NotImplementedError()
