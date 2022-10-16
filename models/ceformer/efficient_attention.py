from typing import Optional

import torch
from torch import nn
from torch.nn.init import constant_, xavier_normal_, xavier_uniform_
from .layers import NonNegativeF, SinEmbedG, CosEmbedG, TopSample, DistributionSample, RandomSample

__author__ = "Xue-Yang Chen"


class LinearAttention(nn.Module):
    def __init__(self):
        super(LinearAttention, self).__init__()
        self.non_neg_f = NonNegativeF()
        self.cos_emb_g = CosEmbedG()
        self.sin_emb_g = SinEmbedG()

    def forward(self, q, k, v, index=None):
        """
        :param q: [..., SeqLen, KDim]
        :param k: [..., SeqLen, KDim]
        :param v: [..., SeqLen, VDim]
        :param index: [..., SeqLen]
        :return: [..., NewSeqLen,
        """
        k_cos = self.cos_emb_g(self.non_neg_f(k)).transpose(-1, -2)
        k_sin = self.sin_emb_g(self.non_neg_f(k)).transpose(-1, -2)
        q_cos = self.cos_emb_g(self.non_neg_f(q))
        q_sin = self.sin_emb_g(self.non_neg_f(q))

        if index is not None:
            q_cos = q_cos[index].reshape(*q.shape[:-2], -1, q.shape[-1])
            q_sin = q_sin[index].reshape(*q.shape[:-2], -1, q.shape[-1])

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

    def __init__(self, strategy, sample_number=384):
        super(TokenSampler, self).__init__()
        self.sample_strategy = self.sample_strategy_map[strategy](sample_number)

    def forward(self, q, k):
        return self.sample_strategy(q, k)


class EfficientAttention(nn.Module):
    """
    """

    def __init__(self, embed_dim, num_heads=1, k_head_dim=None, v_head_dim=None,
                 dropout=0.1, sample_strategy='distribution_sample', sample_number=384, **kwargs) -> None:
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

        self.linear_attention = LinearAttention()
        self.dropout = nn.Dropout(dropout)
        self.sampler = TokenSampler(sample_number=sample_number, strategy=sample_strategy)

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

    def forward(self, x: torch.Tensor):
        batch_size, seq_length, embed_dim = x.size()
        # Separate Q, K, V from linear output [Batch, Head, SeqLen, Dims]
        q = self.q_proj(x).reshape(batch_size, seq_length, self.num_heads, self.k_dim).permute(0, 2, 1, 3)
        k = self.k_proj(x).reshape(batch_size, seq_length, self.num_heads, self.k_dim).permute(0, 2, 1, 3)
        v = self.v_proj(x).reshape(batch_size, seq_length, self.num_heads, self.v_dim).permute(0, 2, 1, 3)

        index = self.sampler(q, k)  # sampled token index [Batch, Head, SeqLen]
        sampled_values = self.linear_attention(q, k, v, index)  # [Batch, Head, NewSeqLen, EmbDims]

        values = torch.zeros(batch_size, self.num_heads, seq_length, self.v_dim).type_as(sampled_values)
        values[index] += sampled_values.reshape(index.sum(), self.v_dim)
        values = values.permute(0, 2, 1, 3).reshape(batch_size, seq_length, self.num_heads * self.v_dim)

        o = self.o_proj(values)
        o = self.dropout(o)
        return o

