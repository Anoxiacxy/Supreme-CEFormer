import numpy as np
import torch.nn as nn
import torch.nn.functional as F

__author__ = "Yu-Hsiang Huang"


class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, embed_dim, hidden_dim, dropout=0.1, **kwargs):
        super().__init__()
        self.w_1 = nn.Linear(embed_dim, hidden_dim) # position-wise
        self.w_2 = nn.Linear(hidden_dim, embed_dim) # position-wise
        self.layer_norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, **kwargs):

        residual = x

        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual

        x = self.layer_norm(x)

        return x
