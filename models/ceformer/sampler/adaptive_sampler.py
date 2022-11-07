import torch
from torch import nn


class AdaptiveSampler(nn.Module):
    def __init__(self, temperature, num_tokens, num_sampled, eps=1e-6):
        super(AdaptiveSampler, self).__init__()
        self.num_tokens = num_tokens
        self.num_sampled = num_sampled
        self.temperature = temperature
        self.eps = eps

    @staticmethod
    def cdf_inverse_function_average_sampling(
        normalized_cdf: torch.Tensor, sample_number: int
    ) -> torch.BoolTensor:
        """

        :param normalized_cdf: [Batch, SeqLen]
        :param sample_number: int
        :return:

        sample points: [1 / 2k, 3 / 2k, 5/2k, ..., 2k-1 / 2k]
        """

        cdf = normalized_cdf * sample_number - 0.5
        cdf_int = cdf.floor().int()

        cdf_snap = torch.not_equal(cdf_int[:, 1:], cdf_int[:, :-1])
        sample = torch.nn.functional.pad(cdf_snap, (1, 0), value=True) & (cdf_int >= 0)

        return sample

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
        v_norm = torch.linalg.norm(
            v.transpose(1, 2).reshape(batch_size, seq_length, num_head * dims), ord=2, dim=2)

        significance_score = significance_score * v_norm
        significance_score = significance_score[:, 1:]  # [B x N-1]
        significance_score = significance_score / significance_score.sum(dim=1, keepdim=True)  # [B x N-1]

        cdf = torch.cumsum(significance_score, dim=1)  # [B x T-1]
        normalized_cdf = (  # normalized cdf
            cdf - cdf.min(dim=1)[0].unsqueeze(dim=1)
        ) / ((cdf.max(dim=1)[0] - cdf.min(dim=1)[0]) / 1.0).unsqueeze(dim=1)

        token_mask = self.cdf_inverse_function_average_sampling(normalized_cdf, self.num_sampled)
        token_mask = torch.nn.functional.pad(token_mask, (1, 0), value=True)

        return token_mask
