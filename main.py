import argparse
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
    parser = argparse.ArgumentParser()


    cifar10 = CIFAR10DataModule(
        data_dir='data',
        batch_size=128,
        normalize=True,
    )

    imagenet = ImagenetDataModule(
        data_dir="/root/autodl-tmp/imagenet",
        batch_size=128,
    )

    minist = MNISTDataModule(
        data_dir='data',
        normalize=True,
    )

    data_map = {
        'cifar10': cifar10,
        'imagenet': imagenet,
        'minist': minist,
    }

    data_name = 'cifar10'

    trainer = pl.Trainer(
        # precision=16,
        default_root_dir=f'/root/tf-logs/{data_name}',
        max_epochs=300,
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
        nun_classes=data_map[data_name].num_classes,
        num_layers=12,
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

    imagenet.val_transforms = None
    imagenet.train_transforms = None

    print(cefomer)

    trainer.fit(
        model=cefomer,
        datamodule=data_map[data_name],
        # ckpt_path='/root/tf-logs/lightning_logs/version_3/checkpoints/epoch=2-step=939.ckpt'
    )
