import torch
from torch import nn


class ThinConvModule(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ThinConvModule, self).__init__()
        self.conv_7x7 = nn.Conv2d(in_channel, out_channel, kernel_size=(7, 7), stride=7, padding=1)
        # self.conv_7x7 = nn.Sequential(
        #     nn.Conv2d(in_channel, out_channel // 4, kernel_size=(3, 3), padding=1),
        #     nn.MaxPool2d((2, 2), (2, 2)),
        #     nn.Conv2d(out_channel // 4, out_channel // 2, kernel_size=(3, 3), padding=1),
        #     nn.MaxPool2d((2, 2), (2, 2)),
        #     nn.Conv2d(out_channel // 2, out_channel, kernel_size=(3, 3), padding=1),
        # )
        self.batch_norm = nn.BatchNorm2d(out_channel)
        self.max_pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

    def forward(self, img: torch.Tensor):
        img = self.conv_7x7(img)
        img = self.batch_norm(img)
        img = self.max_pool(img)
        img = img.permute(0, 2, 3, 1)
        return img
