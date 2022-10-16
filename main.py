import os
from multiprocessing import freeze_support

import pytorch_lightning as pl
from torchvision import transforms

from models import LitCEFormer
from pl_bolts.datamodules import ImagenetDataModule
from pl_bolts.datamodules import CIFAR10DataModule

if __name__ == '__main__':

    cifar10 = CIFAR10DataModule('data')
    trainer = pl.Trainer(
        default_root_dir='checkpoints',
        max_epochs=300,
        accelerator='gpu',
    )

    model = LitCEFormer(
        img_channel=cifar10.dims[0],
        img_height=cifar10.dims[1],
        img_width=cifar10.dims[2],
        output_dim=10, softmax=True)

    transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
            std=[x / 255.0 for x in [63.0, 62.1, 66.7]]
        ),
        # transforms.Resize(224),
    ])

    cifar10.train_transforms = transforms
    cifar10.test_transforms = transforms
    cifar10.val_transforms = transforms

    trainer.fit(model, datamodule=cifar10)
