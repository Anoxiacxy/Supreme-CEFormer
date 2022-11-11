import argparse
import os
from typing import Optional

import torch
import pytorch_lightning as pl
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from models import LitCEFormer, LitViT
from pl_bolts.datamodules import ImagenetDataModule
from pl_bolts.datamodules import CIFAR10DataModule
from pl_bolts.datamodules import MNISTDataModule
from datasets import CIFAR100DataModule
from thop import profile


from torchvision.models import VisionTransformer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", metavar="<phase>", choices=["fit", "test"])
    parser.add_argument("--model", type=str, choices=['ceformer', 'vit'], default='ceformer')
    parser.add_argument("--model_attention", type=str, choices=['e-attention', 'basic'], default="e-attention")
    parser.add_argument("--model_feedforward", type=str, choices=['enhanced', 'basic'], default="enhanced")
    parser.add_argument("--model_num_layers", type=int, default=12)
    parser.add_argument("--model_embed_dim", type=int, default=128)
    parser.add_argument("--model_hidden_dim", type=int, default=384)
    parser.add_argument("--model_prune_rate", type=float, default=0.7)
    parser.add_argument("--model_sampler_strategy", type=str, choices=[
        "adaptive", "top", "random", "distribution"], default="adaptive")
    parser.add_argument("--model_stem", type=str, default="conv", choices=[
        "thin_conv",
        "conv",
        "conv_s",
        "conv_m",
        "conv_l",
        "patchify",
        "alter1",
        "alter2",
        "alter3",
        "alter4",
    ])
    parser.add_argument("--params", action="store_true")
    parser.add_argument("--flops", action="store_true")
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--dataset", type=str, choices=[
        'cifar10', 'cifar100', 'minist', 'imagenet'], default='cifar10', help="dataset to fit")
    parser.add_argument("--ckpt_path", type=str, default=None,
                        help="path to your checkpoints")
    parser.add_argument("--model_ckpt_path", type=str, default=None,
                        help="path to your checkpoints")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--accumulate_batches", type=int, default=8)
    parser.add_argument("--precision", type=int, default=32)
    args = parser.parse_args()

    if args.dataset == "cifar10":
        datamodule = CIFAR10DataModule(
            data_dir='data',
            batch_size=args.batch_size,
            normalize=True,
        )
        transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                std=[x / 255.0 for x in [63.0, 62.1, 66.7]]
            ),
            transforms.Resize(224),
        ])
    elif args.dataset == "cifar100":
        datamodule = CIFAR100DataModule(
            data_dir='data',
            batch_size=args.batch_size,
            normalize=True,
        )
        transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                std=[x / 255.0 for x in [63.0, 62.1, 66.7]]
            ),
            transforms.Resize(224),
        ])
    elif args.dataset == "imagenet":
        datamodule = ImagenetDataModule(
            data_dir="/root/autodl-tmp/imagenet",
            batch_size=args.batch_size,
        )
        transforms = None
    elif args.dataset == "imagenet100":
        datamodule = ImagenetDataModule(
            data_dir="/root/autodl-tmp/imagenet100",
            batch_size=args.batch_size,
        )
    elif args.dataset == "minist":
        datamodule = MNISTDataModule(
            data_dir='data',
            normalize=True,
            batch_size=args.batch_size,
        )
        transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(224),
        ])
    else:
        raise f"Do not support dataset {args.dataset}"

    trainer = pl.Trainer(
        accumulate_grad_batches=args.accumulate_batches,
        precision=args.precision,
        default_root_dir=f'/root/tf-logs/{args.model}-{args.dataset}',
        max_epochs=300,
        accelerator='gpu',
        callbacks=[
            pl.callbacks.LearningRateMonitor(
                logging_interval='step'
            ),
            pl.callbacks.RichProgressBar(),
        ]
    )
    if args.model == "ceformer":
        model = LitCEFormer(
            embed_dim=args.model_embed_dim,
            hidden_dim=args.model_hidden_dim,
            nun_classes=datamodule.num_classes,
            num_layers=args.model_num_layers,
            stem=args.model_stem,
            prune_rate=args.model_prune_rate,
            lr=args.learning_rate,
            attention=args.model_attention,
            feedforward=args.model_feedforward,
            sampler_strategy=args.model_sampler_strategy,
            softmax=False)
    elif args.model == "vit":
        model = LitViT(
            embed_dim=args.model_embed_dim,
            hidden_dim=args.model_hidden_dim,
            nun_classes=datamodule.num_classes,
            num_layers=args.model_num_layers,
            lr=args.learning_rate,
        )
    else:
        raise f"Do not support model {args.model}"

    datamodule.train_transforms = transforms
    datamodule.val_transforms = transforms

    if args.model_ckpt_path is not None:
        model = model.__class__.load_from_checkpoint(args.model_ckpt_path, lr=args.learning_rate / args.accumulate_)

    if args.params or args.flops:
        flops, params = profile(model.network, inputs=(model.example_input_array,))
        print("Params(M)", params / 1e6)
        print("FLOPs(G)", flops / 1e9)

    if args.phase == "fit":
        trainer.fit(
            model=model,
            datamodule=datamodule,
            ckpt_path=args.ckpt_path
        )
    elif args.phase == "test":
        trainer.test(
            model=model,
            datamodule=datamodule,
            ckpt_path=args.ckpt_path
        )
