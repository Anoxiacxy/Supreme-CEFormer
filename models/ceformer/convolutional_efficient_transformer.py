from typing import Optional, Container

import torch
from torch import nn
from .attention import EfficientAttention, MultiHeadAttention
from .feed_forward import EnhancedFeedForward, PositionwiseFeedForward
from .stem import ThinConvModule, PatchifyStem
from .stem import ConvolutionalStem, ConvolutionalSStem, ConvolutionalLStem, ConvolutionalMStem
from .stem import Alternative1Stem, Alternative2Stem, Alternative3Stem, Alternative4Stem


class EncoderLayer(nn.Module):

    attention_cls_map = {
        "e-attention": EfficientAttention,
        "basic": MultiHeadAttention,
    }

    feedforward_cls_map = {
        "enhanced": EnhancedFeedForward,
        "basic": PositionwiseFeedForward,
    }

    def __init__(self, embed_dim, hidden_dim, num_heads,
                 patch_shape, prune_rate, activation, dropout, attention, feedforward,
                 drop_tokens: bool, dilation: tuple = (1, 2, 4), sampler_strategy: str = "adaptive",
                 **kwargs):
        super(EncoderLayer, self).__init__()
        self.attention = self.attention_cls_map[attention](
            embed_dim, num_heads,
            prune_rate=prune_rate,
            patch_shape=patch_shape,
            attn_dropout=dropout,
            proj_dropout=dropout,
            drop_tokens=drop_tokens,
            sampler_strategy=sampler_strategy,
            **kwargs)
        self.feedforward = self.feedforward_cls_map[feedforward](
            embed_dim, hidden_dim,
            patch_shape=patch_shape,
            activation=activation,
            dropout=dropout,
            dilation=dilation,
            **kwargs)
        self.layer_norm_1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.layer_norm_2 = nn.LayerNorm(embed_dim, eps=1e-6)

    def forward(self, seq: torch.Tensor, token_mask: Optional[torch.Tensor]):
        """
        :param seq: [Batch, SeqLen, Embed]
        :param token_mask: [Batch, SeqLen]
        :return:
        """
        seq = self.layer_norm_1(seq)
        old_seq = seq
        seq, attn, token_mask = self.attention(seq, seq, seq, token_mask=token_mask)
        seq = seq + old_seq

        seq = self.layer_norm_2(seq)
        old_seq = seq
        seq = self.feedforward(seq, token_mask=token_mask)
        seq = seq + old_seq
        return seq, token_mask


class ConvolutionalEfficientTransformer(nn.Module):
    activation_class_map = {
        "relu": nn.ReLU,
        'gelu': nn.GELU,
        'identity': nn.Identity,
    }

    stem_class_map = {
        "thin_conv": ThinConvModule,
        "conv": ConvolutionalStem,
        "conv_s": ConvolutionalSStem,
        "conv_m": ConvolutionalMStem,
        "conv_l": ConvolutionalLStem,
        "patchify": PatchifyStem,
        "alter1": Alternative1Stem,
        "alter2": Alternative2Stem,
        "alter3": Alternative3Stem,
        "alter4": Alternative4Stem,
    }

    def __init__(
            self, img_height=224, img_width=224, img_channel=3, num_classes=1000, softmax=False,
            embed_dim=128, hidden_dim=384, num_layers=12, num_heads=8,
            dropout=0.1, prune_rate=0.7, drop_token_layers: Container = range(2, 24), dilation=(1, 2, 4),
            activation='gelu', stem="conv", attention="e-attention",
            feedforward="enhanced", sampler_strategy="adaptive",
            **kwargs
    ):
        super(ConvolutionalEfficientTransformer, self).__init__()
        self.embed_dim = embed_dim
        self.thin_conv = self.stem_class_map[stem](img_channel, embed_dim)

        example_input = torch.randn(1, img_channel, img_height, img_width)
        example_embed = self.thin_conv(example_input)
        self.patch_shape = example_embed.shape[-3:-1]
        # print(f"patch shape = {self.patch_shape}")

        self.class_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.token_mask = torch.ones(1, self.patch_shape[0] * self.patch_shape[1] + 1).bool()
        self.pos_embed = nn.Parameter(torch.randn(1, 1 + self.patch_shape[0] * self.patch_shape[1], embed_dim))
        self.dropout = nn.Dropout(dropout)

        self.encoder_layers = nn.ModuleList([
            EncoderLayer(
                embed_dim, hidden_dim, num_heads, self.patch_shape, prune_rate,
                activation, dropout, attention, feedforward, _ in drop_token_layers,
                dilation, sampler_strategy,
            ) for _ in range(num_layers)
        ])

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes),
        )
        if softmax:
            self.mlp_head.append(nn.Softmax(dim=-1))

    def forward(self, images: torch.Tensor, log_token_count: bool = False):
        single_image = False
        if images.dim == 3:
            images = images.unsqueeze(0)
            single_image = True
        batch_size, channel, height, width = images.shape
        class_token = self.class_token.expand((batch_size, -1, -1))
        embed_token = self.thin_conv(images).reshape(batch_size, -1, self.embed_dim)
        seq = torch.cat([class_token, embed_token], dim=-2)
        seq += self.pos_embed
        seq = self.dropout(seq)

        token_mask = self.token_mask.to(images.device).repeat(batch_size, 1)
        token_count = []
        for encoder_layer in self.encoder_layers:
            seq, token_mask = encoder_layer(seq, token_mask)
            token_count.append(token_mask[0].sum().item())

        class_token = seq[:, 0]
        cls = self.mlp_head(class_token)

        if single_image:
            cls = cls[0]

        if log_token_count:
            out = cls, token_count
        else:
            out = cls

        return out

