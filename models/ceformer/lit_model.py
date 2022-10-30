import math
import random
from typing import Optional
import torch
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import nn, optim
import pytorch_lightning as pl
from .convolutional_efficient_transformer import ConvolutionalEfficientTransformer
from .utils import CosineWarmupScheduler
import torchmetrics


class LitCEFormer(pl.LightningModule):
    optimizer_class_map = {
        'adamw': optim.AdamW,
        'sgd': optim.SGD
    }

    def __init__(
            self,
            # model parameters
            img_height: int = 224, img_width: int = 224, img_channel: int = 3,
            img_dims: tuple = None,
            nun_classes: int = 1000,
            embed_dim: int = 128, hidden_dim: int = 384,
            num_layers: int = 12, num_heads=8,
            activation='gelu', softmax=True,
            dropout=0.1, prune_rate=0.7, stem="conv",
            # optimizer parameters
            optimizer='adamw',
            lr: float = 0.0005,
            weight_decay: float = 0.05,
            batch_size: int = 1024,
            warmup_epochs: int = 5,
            max_epochs: int = 300,
            **kwargs):
        super().__init__(**kwargs)
        if img_dims is not None:
            img_channel, img_height, img_width = img_dims
        self.save_hyperparameters()
        self.network = ConvolutionalEfficientTransformer(
            img_height, img_width, img_channel, nun_classes, softmax, embed_dim, hidden_dim,
            num_layers, num_heads, activation, dropout, prune_rate, stem
        )
        self.loss = nn.CrossEntropyLoss()
        self.train_acc = torchmetrics.Accuracy(top_k=1)
        self.valid_acc = torchmetrics.Accuracy(top_k=1)
        self.example_input_array = torch.randn(1, 3, img_width, img_height)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.network(images)

    def _calculate_loss(self, batch, batch_idx, mode):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)

        if mode == 'train':
            self.train_acc(y_hat, y)
            self.log('train_acc', self.train_acc, on_step=True, on_epoch=False, prog_bar=True)
        elif mode == 'valid':
            self.valid_acc(y_hat, y)
            self.log('valid_acc', self.valid_acc, on_step=True, on_epoch=False, prog_bar=True)
        self.log(f'{mode}_loss', loss)
        return loss

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        loss = self._calculate_loss(batch, batch_idx, 'train')
        return loss

    def validation_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        loss = self._calculate_loss(batch, batch_idx, 'valid')
        return loss

    def test_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        loss = self._calculate_loss(batch, batch_idx, 'test')
        return loss

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
