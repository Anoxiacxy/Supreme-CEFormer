import torch
from torch import nn
from torch.nn import MultiheadAttention
from .efficient_attention import EfficientAttention
from .enhanced_feed_forward import EnhancedFeedForward
from .layers import ThinConvModule


class EncoderLayer(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_heads,
                 patch_shape, sample_number, activation, dropout, **kwargs):
        super(EncoderLayer, self).__init__()
        self.attention = EfficientAttention(
            embed_dim, num_heads,
            sample_number=sample_number,
            dropout=dropout, **kwargs)
        self.feedforward = EnhancedFeedForward(
            embed_dim, hidden_dim, patch_shape,
            activation=activation, dropout=dropout, **kwargs)
        self.layer_norm_1 = nn.LayerNorm(embed_dim)
        self.layer_norm_2 = nn.LayerNorm(embed_dim)

    def forward(self, seq: torch.Tensor):
        seq = self.layer_norm_1(seq)
        seq = self.attention(seq) + seq
        seq = self.layer_norm_2(seq)
        seq = self.feedforward(seq) + seq
        return seq


class ConvolutionalEfficientTransformer(nn.Module):
    activation_class_map = {
        'gelu': nn.GELU,
        'identity': nn.Identity,
    }

    def __init__(self, img_height=224, img_width=224, img_channel=3, output_dim=1000, softmax=False,
                 embed_dim=128, hidden_dim=384, num_layers=12, num_heads=8, activation='gelu',
                 dropout=0.1, prune_rate=0.7, **kwargs):
        super(ConvolutionalEfficientTransformer, self).__init__()
        self.embed_dim = embed_dim
        self.thin_conv = ThinConvModule(img_channel, embed_dim)

        example_input = torch.randn(1, img_channel, img_height, img_width)
        example_embed = self.thin_conv(example_input)
        self.patch_shape = example_embed.shape[-3:-1]
        self.sample_number = int(self.patch_shape[0] * self.patch_shape[1] * prune_rate)

        self.class_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, 1 + self.patch_shape[0] * self.patch_shape[1], embed_dim))
        self.dropout = nn.Dropout(dropout)

        self.encoder_layers = nn.ModuleList([
            EncoderLayer(
                embed_dim, hidden_dim, num_heads, self.patch_shape, self.sample_number, activation, dropout
            ) for _ in range(num_layers)
        ])

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, output_dim),
        )
        if softmax:
            self.mlp_head.append(nn.Softmax(dim=-1))

    def forward(self, images: torch.Tensor):
        single_input = False
        if images.dim() == 3:
            single_input = True
            images = images.unsqueeze(0)
        batch_size, channel, height, width = images.shape
        class_token = self.class_token.expand((batch_size, -1, -1))
        embed_token = self.thin_conv(images).reshape(batch_size, -1, self.embed_dim)
        seq = torch.cat([class_token, embed_token], dim=-2)
        seq += self.pos_embed
        seq = self.dropout(seq)

        for encoder_layer in self.encoder_layers:
            seq = encoder_layer(seq)

        cls = self.mlp_head(seq[:, 0])

        if single_input:
            return cls[0]
        else:
            return cls
