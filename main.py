import os
import torch
import pytorch_lightning as pl
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from models import LitCEFormer, LitViT
from pl_bolts.datamodules import ImagenetDataModule
from pl_bolts.datamodules import CIFAR10DataModule
from pl_bolts.datamodules import MNISTDataModule
from pl_bolts.models.vision import UNet

if __name__ == '__main__':
    cifar10 = CIFAR10DataModule(
        data_dir='data',
        batch_size=128,
        normalize=True,
    )

    minist = MNISTDataModule(
        data_dir='data',
        normalize=True,
    )

    trainer = pl.Trainer(
        precision=16,
        default_root_dir='tf-logs',
        max_epochs=600,
        accelerator='gpu',
        callbacks=[
            pl.callbacks.LearningRateMonitor(
                logging_interval='step'
            ),
            pl.callbacks.RichProgressBar(),
        ]
    )

    unet = UNet(num_classes=cifar10.num_classes)

    cefomer = LitCEFormer(
        embed_dim=32,
        # img_channel=cifar10.dims[0],
        # img_height=cifar10.dims[1],
        # img_width=cifar10.dims[2],
        # img_dims=cifar10.dims,
        nun_classes=cifar10.num_classes,
        num_layers=6,
        # output_dim=cifar10.num_classes,
        prune_rate=0.7,
        lr=0.00005,
        softmax=True)

    # # 创建输入网络的tensor
    # tensor = (torch.rand(1, 3, 224, 224),)
    #
    # # 分析FLOPs
    # flops, params = profile(model.network, tensor)
    # print("FLOPs: ", flops / 1e9)
    # print("PARAMs: ", params / 1e6)

    transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
            std=[x / 255.0 for x in [63.0, 62.1, 66.7]]
        ),
        transforms.Resize(224),
    ])

    cifar10.train_transforms = transforms
    cifar10.test_transforms = transforms
    cifar10.val_transforms = transforms

    trainer.fit(
        model=cefomer,
        datamodule=cifar10,
        # ckpt_path='~/tf-logs/lightning_logs/version_15/checkpoints/epoch=13-step=4382.ckpt'
    )
