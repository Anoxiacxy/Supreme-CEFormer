import os
from multiprocessing import freeze_support

import torch

import pytorch_lightning as pl
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from models import LitCEFormer, LitViT
from pl_bolts.datamodules import ImagenetDataModule
from pl_bolts.datamodules import CIFAR10DataModule


if __name__ == '__main__':
    cifar10 = CIFAR10DataModule('data')

    trainer = pl.Trainer(
        default_root_dir='checkpoints',
        max_epochs=300,
        accelerator='cpu',
    )

    model = LitCEFormer(
        embed_dim=384,
        # img_channel=cifar10.dims[0],
        # img_height=cifar10.dims[1],
        # img_width=cifar10.dims[2],
        num_layers=12,
        # output_dim=cifar10.num_classes,
        prune_rate=0.7,
        softmax=True)

    # 创建输入网络的tensor
    tensor = (torch.rand(1, 3, 224, 224),)

    # 分析FLOPs
    flops, params = profile(model.network, tensor)
    print("FLOPs: ", flops / 1e9)
    print("PARAMs: ", params / 1e6)

    # writer = SummaryWriter(log_dir='/root/tf-logs/', comment='CEFormer')
    # with writer:
    #     writer.add_graph(model.network, Variable(torch.randn(4, 3, 224, 224)))
    # 23.63 GMac
    # model.load_from_checkpoint()
    # model = LitViT(
    #     model_kwargs={
    #         "embed_dim": 256,
    #         "hidden_dim": 512,
    #         "num_heads": 8,
    #         "num_layers": 6,
    #         "patch_size": 4,
    #         "num_channels": 3,
    #         "num_patches": 64,
    #         "num_classes": 10,
    #         "dropout": 0.2,
    #     },
    #     lr=3e-4)

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

    trainer.fit(model, datamodule=cifar10)
