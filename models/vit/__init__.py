import math

import torch
import torchmetrics
from torch import nn, optim
import pytorch_lightning as pl
import torch.functional as F
from torchvision.models import VisionTransformer

def img_to_patch(x, patch_size, flatten_channels=True):
    """
    Inputs:
        x - Tensor representing the image of shape [B, C, H, W]
        patch_size - Number of pixels per dimension of the patches (integer)
        flatten_channels - If True, the patches will be returned in a flattened format
                           as a feature vector instead of a image grid.
    """
    B, C, H, W = x.shape
    x = x.reshape(B, C, H // patch_size, patch_size, W // patch_size, patch_size)
    x = x.permute(0, 2, 4, 1, 3, 5)  # [B, H', W', C, p_H, p_W]
    x = x.flatten(1, 2)  # [B, H'*W', C, p_H, p_W]
    if flatten_channels:
        x = x.flatten(2, 4)  # [B, H'*W', C*p_H*p_W]
    return x


class AttentionBlock(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_heads, dropout=0.0):
        """
        Inputs:
            embed_dim - Dimensionality of input and attention feature vectors
            hidden_dim - Dimensionality of hidden layer in feed-forward network
                         (usually 2-4x larger than embed_dim)
            num_heads - Number of heads to use in the Multi-Head Attention block
            dropout - Amount of dropout to apply in the feed-forward network
        """
        super().__init__()

        self.layer_norm_1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.layer_norm_2 = nn.LayerNorm(embed_dim)
        self.linear = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        inp_x = self.layer_norm_1(x)
        x = x + self.attn(inp_x, inp_x, inp_x)[0]
        x = x + self.linear(self.layer_norm_2(x))
        return x


class MyVisionTransformer(nn.Module):
    def __init__(
        self,
        embed_dim,
        hidden_dim,
        num_channels,
        num_heads,
        num_layers,
        num_classes,
        patch_size,
        num_patches,
        dropout=0.0,
    ):
        """
        Inputs:
            embed_dim - Dimensionality of the input feature vectors to the Transformer
            hidden_dim - Dimensionality of the hidden layer in the feed-forward networks
                         within the Transformer
            num_channels - Number of channels of the input (3 for RGB)
            num_heads - Number of heads to use in the Multi-Head Attention block
            num_layers - Number of layers to use in the Transformer
            num_classes - Number of classes to predict
            patch_size - Number of pixels that the patches have per dimension
            num_patches - Maximum number of patches an image can have
            dropout - Amount of dropout to apply in the feed-forward network and
                      on the input encoding
        """
        super().__init__()

        self.patch_size = patch_size

        # Layers/Networks
        self.input_layer = nn.Linear(num_channels * (patch_size**2), embed_dim)
        self.transformer = nn.Sequential(
            *(AttentionBlock(embed_dim, hidden_dim, num_heads, dropout=dropout) for _ in range(num_layers))
        )
        self.mlp_head = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, num_classes))
        self.dropout = nn.Dropout(dropout)

        # Parameters/Embeddings
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, 1 + num_patches, embed_dim))

    def forward(self, x):
        # Preprocess input
        x = img_to_patch(x, self.patch_size)
        B, T, _ = x.shape
        x = self.input_layer(x)

        # Add CLS token and positional encoding
        cls_token = self.cls_token.repeat(B, 1, 1)
        x = torch.cat([cls_token, x], dim=1)
        x = x + self.pos_embedding[:, : T + 1]

        # Apply Transforrmer
        x = self.dropout(x)
        x = x.transpose(0, 1)
        x = self.transformer(x)

        # Perform classification prediction
        cls = x[0]
        out = self.mlp_head(cls)
        return out


class LitViT(pl.LightningModule):
    optimizer_class_map = {
        'adamw': optim.AdamW,
        'sgd': optim.SGD
    }

    def __init__(
            self,
            img_height: int = 224, img_width: int = 224, img_channel: int = 3,
            nun_classes: int = 1000,
            embed_dim: int = 128,
            hidden_dim: int = 384,
            num_layers: int = 12,
            num_heads: int = 8,
            patch_size: int = 16,
            dropout: float = 0.1,
            # optimizer parameters
            optimizer='adamw',
            lr: float = 0.0005,
            weight_decay: float = 0.05,
            batch_size: int = 1024,
            warmup_epochs: int = 5,
            max_epochs: int = 300,
    ):
        super().__init__()
        self.save_hyperparameters()
        assert img_width == img_height
        self.network = VisionTransformer(
            image_size=img_width,
            patch_size=patch_size,
            num_layers=num_layers,
            num_heads=num_heads,
            hidden_dim=embed_dim,
            mlp_dim=hidden_dim,
            dropout=dropout,
            num_classes=nun_classes,
        )
        self.loss = nn.CrossEntropyLoss()
        self.train_acc = torchmetrics.Accuracy(top_k=1)
        self.valid_acc = torchmetrics.Accuracy(top_k=1)
        self.example_input_array = torch.randn(1, 3, img_width, img_height)

    def forward(self, x):
        return self.network(x)

    def configure_optimizers(self):
        optimizer = self.optimizer_class_map[self.hparams.optimizer](
            self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)

        def warm_up_with_cosine_lr(epoch):
            if epoch < self.hparams.warmup_epochs:
                return (epoch + 1) / self.hparams.warmup_epochs
            else:
                return 0.5 * (math.cos((epoch - self.hparams.warmup_epochs) /
                                       (self.hparams.max_epochs - self.hparams.warmup_epochs) * math.pi) + 1)

        lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warm_up_with_cosine_lr)
        return [optimizer], [lr_scheduler]
        # lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     optimizer, mode='min', patience=5, factor=0.9, verbose=True)
        # return {"optimizer": optimizer, "lr_scheduler": lr_scheduler, "monitor": "valid_loss"}

    def _calculate_loss(self, batch, mode="train"):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)

        if mode == 'train':
            self.train_acc(y_hat, y)
            self.log('train_acc', self.train_acc, on_step=True, on_epoch=False, prog_bar=True)
        elif mode == 'valid':
            self.valid_acc(y_hat, y)
            self.log('valid_acc', self.valid_acc, on_step=True, on_epoch=False, prog_bar=True)
        self.log(f'{mode}_loss', loss, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._calculate_loss(batch, mode="train")
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._calculate_loss(batch, mode="valid")
        return loss

    def test_step(self, batch, batch_idx):
        loss = self._calculate_loss(batch, mode="test")
        return loss
