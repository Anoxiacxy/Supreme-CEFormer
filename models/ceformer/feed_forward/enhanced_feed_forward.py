import torch
from torch import nn

__author__ = "Xue-Yang Chen"


class EnhancedFeedForward(nn.Module):
    activation_class_map = {
        'gelu': nn.GELU,
        'identity': nn.Identity,
    }

    def __init__(self, embed_dim, hidden_dim, patch_shape,
                 dilation: tuple = (1, 2, 4), activation='gelu', dropout=0.1):
        super(EnhancedFeedForward, self).__init__()
        self.embed_dim = embed_dim
        self.patch_shape = patch_shape

        self.conv_1x1_dot_1 = nn.Conv2d(embed_dim, hidden_dim, (1, 1))
        self.conv_3x3_depth = nn.Conv2d(hidden_dim, hidden_dim, (3, 3), 1, 1, groups=hidden_dim)
        self.conv_3x3_dilation = nn.ModuleList([
            nn.Conv2d(hidden_dim, hidden_dim, (3, 3), padding=d, dilation=d, groups=hidden_dim) for d in dilation])
        self.conv_1x1_dot_2 = nn.Conv2d(hidden_dim * len(dilation), embed_dim, (1, 1))

        self.activation = self.activation_class_map[activation]()
        self.dropout = nn.Dropout(dropout)

    def forward(self, seq: torch.Tensor):
        img, class_token = self.seq2img(seq)

        img = self.conv_1x1_dot_1(img)
        img = self.conv_3x3_depth(img)
        img = self.activation(img)
        img = self.dropout(img)

        img_multi_scales = [conv(img) for conv in self.conv_3x3_dilation]
        img = torch.cat(img_multi_scales, dim=-3)
        img = self.conv_1x1_dot_2(img)
        img = self.dropout(img)

        seq = self.img2seq(img, class_token)
        return seq

    def seq2img(self, seq: torch.Tensor):
        class_token, seq = seq[..., :1, :], seq[..., 1:, :]
        # assert seq.shape[-2] == self.patch_shape[0] * self.patch_shape[1]
        img = seq.reshape(-1, *self.patch_shape, self.embed_dim)
        img = img.permute(0, 3, 1, 2)
        return img, class_token

    def img2seq(self, img: torch.Tensor, class_token: torch.Tensor):
        img = img.permute(0, 2, 3, 1)
        seq = img.reshape(-1, self.patch_shape[0] * self.patch_shape[1], self.embed_dim)
        seq = torch.cat([class_token, seq], dim=-2)
        return seq



