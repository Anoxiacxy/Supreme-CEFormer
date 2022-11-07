import math

import torch
from torch import nn


class DistributionSampler(nn.Module):
    def __init__(self, temperature, num_tokens, num_sampled, eps=1e-6, replacement=False):
        super(DistributionSampler, self).__init__()
        self.num_tokens = num_tokens
        self.num_sampled = num_sampled
        self.temperature = temperature
        self.replacement = replacement
        self.eps = eps

    def get_attention(self, q: torch.Tensor, k: torch.Tensor, token_mask: torch.Tensor):
        """
        Calculate the contribution of other tokens to the class token
        :param q: [Batch, Head, SeqLen, Dims]
        :param k: [Batch, Head, SeqLen, Dims]
        :param token_mask: [Batch, SeqLen]
        :return: [Batch, Head, 1, SeqLen]
        """

        # attention: [Batch, Head, 1, SeqLen]
        attn = torch.matmul(q[..., :1, :], k.transpose(-2, -1)) / self.temperature
        attn = attn - torch.max(attn, dim=-1, keepdim=True)[0]

        batch_size, seq_length = token_mask.size()
        attn_mask = token_mask.view(batch_size, 1, 1, seq_length)
        attn = torch.exp(attn) * attn_mask

        attn = (attn + self.eps / seq_length) / (attn.sum(dim=-1, keepdim=True) + self.eps)
        return attn

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, token_mask: torch.Tensor):
        """
        :param token_mask: [Batch, SeqLen]
        :param q: [Batch, Head, SeqLen, Dims]
        :param k: [Batch, Head, SeqLen, Dims]
        :param v: [Batch, Head, SeqLen, Dims]
        :return:
        """
        attn = self.get_attention(q, k, token_mask)

        with torch.no_grad():
            index = torch.multinomial(attn, num_samples=self.r, replacement=self.replacement)
        index = torch.cat([torch.zeros_like(index[..., :1]).type_as(index).int(), index + 1], dim=-1)
        logic = torch.zeros_like(q[..., 0]).type_as(index).bool()
        logic.scatter_(dim=-1, index=index, src=torch.ones_like(index).bool())
        return logic
