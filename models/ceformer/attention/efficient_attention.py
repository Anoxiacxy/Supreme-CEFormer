from typing import Optional

import torch
from torch import nn
from torch.nn.init import constant_, xavier_uniform_

__author__ = "Xue-Yang Chen"

import math

import torch
from torch import nn


class NonNegativeF(nn.Module):
    def forward(self, x: torch.Tensor):
        return torch.exp(x) * (x < 0) + (x + 1) * (x >= 0)


class CosEmbedG(nn.Module):
    def __init__(self, n):
        super(CosEmbedG, self).__init__()
        self.n = n
        self.vec = torch.cos(torch.pi / 2.0 / n * torch.arange(n))

    def forward(self, x):
        self.vec = self.vec.type_as(x)
        return x * self.vec.unsqueeze(1)


class SinEmbedG(nn.Module):
    def __init__(self, n):
        super(SinEmbedG, self).__init__()
        self.n = n
        self.vec = torch.sin(torch.pi / 2.0 / n * torch.arange(n))

    def forward(self, x):
        self.vec = self.vec.type_as(x)
        return x * self.vec.unsqueeze(1)


class TopSample(nn.Module):
    def __init__(self, m, r):
        super(TopSample, self).__init__()
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


class RandomSample(nn.Module):
    def __init__(self, m, r):
        super(RandomSample, self).__init__()
        self.r = r
        self.m = m

    def forward(self, q, k):
        """
        :param q: [..., SeqLen, Dims]
        :param k: [..., SeqLen, Dims]
        :return:
        """
        pass


class DistributionSample(nn.Module):
    def __init__(self, m, r, replacement=False):
        super(DistributionSample, self).__init__()
        self.m = m
        self.r = r
        self.replacement = replacement

    def forward(self, q, k):
        """
        :param num_heads:
        :param q: [Batch * Head, SeqLen, Dims]
        :param k: [Batch * Head, SeqLen, Dims]
        :return: i: [Batch * Head, SeqLen]
        """
        a = torch.matmul(q[..., :1, :], k[..., 1:, :].transpose(-2, -1))
        a = a / math.sqrt(q.size()[-1])
        a = nn.functional.softmax(a, dim=-1)[..., 0, :]
        with torch.no_grad():
            index = torch.multinomial(a, num_samples=self.r, replacement=self.replacement)
        index = torch.cat([torch.zeros_like(index[..., :1]).type_as(index).int(), index + 1], dim=-1)
        logic = torch.zeros_like(q[..., 0]).type_as(index).bool()
        logic.scatter_(dim=-1, index=index, src=torch.ones_like(index).bool())
        return logic


class LinearAttention(nn.Module):
    def __init__(self, n):
        super(LinearAttention, self).__init__()
        self.non_neg_f = NonNegativeF()
        self.cos_emb_g = CosEmbedG(n)
        self.sin_emb_g = SinEmbedG(n)

    def forward(self, q, k, v, index=None):
        """
        :param q: [..., SeqLen, KDim]
        :param k: [..., SeqLen, KDim]
        :param v: [..., SeqLen, VDim]
        :param index: [..., SeqLen] bool
        :return: [..., NewSeqLen,
        """
        k_cos = self.cos_emb_g(self.non_neg_f(k)).transpose(-1, -2)
        k_sin = self.sin_emb_g(self.non_neg_f(k)).transpose(-1, -2)
        q_cos = self.cos_emb_g(self.non_neg_f(q))
        q_sin = self.sin_emb_g(self.non_neg_f(q))

        if index is not None:
            q_cos = q_cos[index].reshape(q.shape[0], -1, q.shape[-1])
            q_sin = q_sin[index].reshape(q.shape[0], -1, q.shape[-1])

        k_sin_v = torch.matmul(k_sin, v)  # [..., KDim, VDim]
        k_cos_v = torch.matmul(k_cos, v)

        values = torch.matmul(q_cos, k_cos_v) + torch.matmul(q_sin, k_sin_v)

        return values


class TokenSampler(nn.Module):
    """

    """
    sample_strategy_map = {
        'top_sample': TopSample,
        'distribution_sample': DistributionSample,
        'random_sample': RandomSample,
    }

    def __init__(self, strategy, total_number, sample_number=384):
        super(TokenSampler, self).__init__()
        self.sample_strategy = self.sample_strategy_map[strategy](total_number, sample_number)

    def forward(self, q, k, *args):
        return self.sample_strategy(q, k, *args)


class EfficientAttention(nn.Module):
    """
    """

    def __init__(self, embed_dim, num_heads, patch_shape, k_head_dim=None, v_head_dim=None,
                 dropout=0.1, sample_strategy='distribution_sample', prune_rate=0.7, **kwargs) -> None:
        super(EfficientAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout

        if k_head_dim is None or v_head_dim is None:
            assert self.embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.k_dim = k_head_dim or self.embed_dim // num_heads
        self.v_dim = v_head_dim or self.embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, num_heads * self.k_dim)
        self.k_proj = nn.Linear(embed_dim, num_heads * self.k_dim)
        self.v_proj = nn.Linear(embed_dim, num_heads * self.v_dim)
        if self.v_dim * num_heads != embed_dim:
            self.o_proj = nn.Linear(num_heads * self.v_dim, embed_dim)
        else:
            self.o_proj = nn.Identity()
        self.seq_length = patch_shape[0] * patch_shape[1] + 1
        self.linear_attention = LinearAttention(self.seq_length)
        self.dropout = nn.Dropout(dropout)
        self.sampler = TokenSampler(
            total_number=self.seq_length,
            sample_number=int(self.seq_length * prune_rate),
            strategy=sample_strategy)

        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self.q_proj.weight)
        xavier_uniform_(self.k_proj.weight)
        xavier_uniform_(self.v_proj.weight)
        # xavier_uniform_(self.o_proj.weight)
        constant_(self.q_proj.bias, 0.0)
        constant_(self.k_proj.bias, 0.0)
        constant_(self.v_proj.bias, 0.0)
        # constant_(self.o_proj.bias, 0.0)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        batch_size, seq_length, embed_dim = q.size()
        # Separate Q, K, V from linear output [Batch, Head, SeqLen, Dims]
        q = self.q_proj(q)\
            .reshape(batch_size, seq_length, self.num_heads, self.k_dim)\
            .permute(0, 2, 1, 3)\
            .reshape(batch_size * self.num_heads, seq_length, self.k_dim)
        k = self.k_proj(k)\
            .reshape(batch_size, seq_length, self.num_heads, self.k_dim)\
            .permute(0, 2, 1, 3)\
            .reshape(batch_size * self.num_heads, seq_length, self.k_dim)
        v = self.v_proj(v)\
            .reshape(batch_size, seq_length, self.num_heads, self.v_dim)\
            .permute(0, 2, 1, 3)\
            .reshape(batch_size * self.num_heads, seq_length, self.v_dim)

        index = self.sampler(q, k)  # sampled token index [Batch * Head, SeqLen]
        sampled_values = self.linear_attention(q, k, v, index)  # [Batch * Head, SeqLen, EmbDims]

        values = torch.zeros(batch_size * self.num_heads, self.seq_length, self.v_dim).type_as(sampled_values)
        values[index] += sampled_values.reshape(index.sum(), self.v_dim)

        values = values\
            .reshape(batch_size, self.num_heads, self.seq_length, self.v_dim)\
            .permute(0, 2, 1, 3)\
            .reshape(batch_size, self.seq_length, self.num_heads * self.v_dim)

        o = self.o_proj(values)
        o = self.dropout(o)
        return o, None
