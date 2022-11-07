from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.nn.init import constant_, xavier_uniform_
from ..sampler import DistributionSampler, TopSampler, AdaptiveSampler, RandomSampler

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
        if self.vec.device != x.device:
            self.vec = self.vec.type_as(x)
        return x * self.vec.unsqueeze(1)


class SinEmbedG(nn.Module):
    def __init__(self, n):
        super(SinEmbedG, self).__init__()
        self.n = n
        self.vec = torch.sin(torch.pi / 2.0 / n * torch.arange(n))

    def forward(self, x):
        if self.vec.device != x.device:
            self.vec = self.vec.type_as(x)
        return x * self.vec.unsqueeze(1)


class LinearAttention(nn.Module):
    def __init__(self, n, temperature):
        super(LinearAttention, self).__init__()
        self.non_neg_f = NonNegativeF()
        self.cos_emb_g = CosEmbedG(n)
        self.sin_emb_g = SinEmbedG(n)

        self.temperature = temperature

    def forward(self, q, k, v, token_mask: Optional[torch.Tensor] = None):
        """
        :param q: [Batch, Head, SeqLen, KDim]
        :param k: [Batch, Head, SeqLen, KDim]
        :param v: [Batch, Head, SeqLen, VDim]
        :param token_mask: [Batch, SeqLen] bool
        :return: values [Batch, Head, SeqLen, VDim]
        """
        batch_size, num_head, seq_length, k_dim = q.size()

        k_cos = self.cos_emb_g(self.non_neg_f(k)).transpose(-1, -2)  # [Batch, Head, KDim, SeqLen]
        k_sin = self.sin_emb_g(self.non_neg_f(k)).transpose(-1, -2)

        q_cos = self.cos_emb_g(self.non_neg_f(q))  # [Batch, Head, SeqLen, KDim]
        q_sin = self.sin_emb_g(self.non_neg_f(q))

        k_sin_v = torch.matmul(k_sin, v)  # [Batch, Head, KDim, VDim]
        k_cos_v = torch.matmul(k_cos, v)

        if token_mask is not None:
            # token_count = torch.sum(token_mask, dim=-1)  # [Batch]
            # q_cos = q_cos.transpose(1, 2)
            # q_sin = q_sin.transpose(1, 2)
            # [batch seq dim, Head, KDim)
            # q_cos = q_cos[token_mask]
            # q_sin = q_sin[token_mask]

            # values = torch.cat([
            #     torch.einsum("shk, hkv->shv", q_cos[i][token_mask[i]], k_cos_v[i]) +
            #     torch.einsum("shk, hkv->shv", q_sin[i][token_mask[i]], k_sin_v[i])
            #     for i in range(batch_size)
            # ])
            # # values = (torch.matmul(q_cos, k_cos_v) + torch.matmul(q_sin, k_sin_v))
            # values = values / self.temperature
            #
            # zeros = torch.zeros_like(v).transpose(1, 2)
            # # print(zeros.shape, token_mask.shape, values.shape)
            # zeros[token_mask] = values
            # values = zeros.transpose(1, 2)
            expand_token_mask = token_mask.view(batch_size, 1, seq_length, 1).expand_as(q)
            q_cos *= expand_token_mask
            q_sin *= expand_token_mask

            values = (torch.matmul(q_cos, k_cos_v) + torch.matmul(q_sin, k_sin_v))
            values = values / self.temperature

        else:
            values = (torch.matmul(q_cos, k_cos_v) + torch.matmul(q_sin, k_sin_v))
            values = values / self.temperature

        return values


class EfficientAttention(nn.Module):

    sampler_strategy_map = {
        'top': TopSampler,
        'random': RandomSampler,
        'distribution': DistributionSampler,
        'adaptive': AdaptiveSampler,
    }

    def __init__(
            self,
            embed_dim,
            num_heads,
            patch_shape,
            k_head_dim=None,
            v_head_dim=None,
            qkv_bias=False,
            attn_dropout=0.1,
            proj_dropout=0.1,
            drop_tokens=False,
            sampler_strategy='adaptive',
            prune_rate=1,
            **kwargs,
    ) -> None:
        super(EfficientAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.qkv_bias = qkv_bias

        if k_head_dim is None or v_head_dim is None:
            assert self.embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.k_dim = k_head_dim or self.embed_dim // num_heads
        self.v_dim = v_head_dim or self.embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, num_heads * self.k_dim, bias=qkv_bias)
        self.k_proj = nn.Linear(embed_dim, num_heads * self.k_dim, bias=qkv_bias)
        self.v_proj = nn.Linear(embed_dim, num_heads * self.v_dim, bias=qkv_bias)
        self.o_proj = nn.Linear(num_heads * self.v_dim, embed_dim)

        self.seq_length = np.product(patch_shape) + 1
        self.linear_attention = LinearAttention(self.seq_length, temperature=self.k_dim ** 0.5)

        self.attn_dropout = nn.Dropout(attn_dropout)
        self.proj_dropout = nn.Dropout(proj_dropout)

        if drop_tokens:
            self.sampler = self.sampler_strategy_map[sampler_strategy](
                temperature=self.k_dim ** 0.5,
                num_tokens=self.seq_length,
                num_sampled=int(self.seq_length * prune_rate),
            )
        else:
            self.sampler = None

        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self.q_proj.weight)
        xavier_uniform_(self.k_proj.weight)
        xavier_uniform_(self.v_proj.weight)
        xavier_uniform_(self.o_proj.weight)
        if self.qkv_bias:
            constant_(self.q_proj.bias, 0.0)
            constant_(self.k_proj.bias, 0.0)
            constant_(self.v_proj.bias, 0.0)
        constant_(self.o_proj.bias, 0.0)

    def forward(
        self,
        q: torch.FloatTensor,
        k: torch.FloatTensor,
        v: torch.FloatTensor,
        token_mask: Optional[torch.BoolTensor]
    ) -> (torch.FloatTensor, torch.FloatTensor, torch.BoolTensor):

        batch_size, seq_length, embed_dim = q.size()
        # Separate Q, K, V from linear output [Batch, Head, SeqLen, Dims]
        q = self.q_proj(q) \
            .view(batch_size, seq_length, self.num_heads, self.k_dim) \
            .transpose(1, 2)

        k = self.k_proj(k) \
            .view(batch_size, seq_length, self.num_heads, self.k_dim) \
            .transpose(1, 2)

        v = self.v_proj(v) \
            .view(batch_size, seq_length, self.num_heads, self.v_dim) \
            .transpose(1, 2)

        if self.sampler is not None:
            token_mask = self.sampler(q, k, v, token_mask)

        values = self.linear_attention(q, k, v, token_mask)  # [*, EmbDims]
        values = values.transpose(1, 2) \
            .reshape(batch_size, -1, self.num_heads * self.v_dim)

        o = self.o_proj(values)
        o = self.proj_dropout(o)

        return o, None, token_mask
