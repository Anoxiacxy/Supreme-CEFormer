import torch
from torch import nn


class TopSampler(nn.Module):
    def __init__(self, temperature, num_tokens, num_sampled, eps=1e-6):
        super(TopSampler, self).__init__()
        self.num_tokens = num_tokens
        self.num_sampled = num_sampled
        self.temperature = temperature
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
        batch_size, num_head, seq_length, dims = v.size()
        attn = self.get_attention(q, k, token_mask)
        # Batch, SeqLen
        significance_score = torch.sum(attn[:, :, 0], dim=1)

        significance_score = significance_score[:, 1:]  # [B x N-1]
        significance_score = significance_score / significance_score.sum(dim=1, keepdim=True)  # [B x N-1]

        _, index = torch.sort(significance_score)
        index = torch.cat([torch.zeros_like(index[..., :1]), index + 1], dim=-1)
        token_mask = index <= (self.num_sampled + 1)
        return token_mask
