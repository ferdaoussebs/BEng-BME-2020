import torch
import torch.nn as nn


class ConvolutionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv1_bn = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.conv2_bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(inplace=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv1_bn(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.conv2_bn(x)
        x = self.activation(x)

        return x


class UNet(nn.Module):
    def __init__(self, n_class):
        super().__init__()

        self.downsample_block1 = ConvolutionBlock(3, 64)
        self.downsample_block2 = ConvolutionBlock(64, 128)
        self.downsample_block3 = ConvolutionBlock(128, 256)
        self.downsample_block4 = ConvolutionBlock(256, 512)

        self.upsample_block4 = ConvolutionBlock(512, 256)
        self.upsample_block3 = ConvolutionBlock(256, 128)
        self.upsample_block2 = ConvolutionBlock(128, 64)

        self.conv_last = nn.Conv2d(64, n_class, 1)

        self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.upsample = torch.nn.Upsample(scale_factor=2, mode="nearest")
        self.dropout2d = nn.Dropout2d(p=0.5, inplace=False)

    def forward(self, x):

        conv1 = self.downsample_block1(x)  # 64 512 512
        x = self.maxpool(conv1)  # 64 256 256
        x = self.dropout2d(x)

        conv2 = self.downsample_block2(x)  # 128 256 256
        x = self.maxpool(conv2)  # 128 128 128
        x = self.dropout2d(x)

        conv3 = self.downsample_block3(x)  # 256 128 128
        x = self.maxpool(conv3)  # 256 64 64
        x = self.dropout2d(x)

        x = self.downsample_block4(x)  # 512, 64, 64

        x = self.upsample(x)  # 512, 128, 128
        x = self.upsample_block4(x)  # 256, 128, 128
        x = x + conv3  # 256, 128, 128

        # 256, 256, 256
        # 128, 256, 256
        # 128, 256, 256

        x = self.upsample(x)
        x = self.upsample_block3(x)
        x = x + conv2

        # 128, 512, 512
        # 64, 512, 512
        # 64, 512, 512

        x = self.upsample(x)
        x = self.upsample_block2(x)
        x = x + conv1

        x = self.conv_last(x)

        return x
