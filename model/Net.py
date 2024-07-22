import pandas as pd
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
import time


class ResBlock(nn.Module):
    """
    残差块
    """
    def __init__(self, input_channels, output_channels, use_1x1conv=False, strides=1):
        """
        :param input_channels: 输入的通道数
        :param output_channels: 输出的通道数
        :param use_1x1conv: 是否使用1x1的卷积核[默认不使用]
        :param strides: 滑动步长
        """
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(output_channels)
        self.bn2 = nn.BatchNorm2d(output_channels)

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = F.relu(self.bn2(self.conv2(y)))
        if self.conv3:
            x = self.conv3(x)
        y = y + x
        return y


class BottleNeck(nn.Module):
    """
    Bottleneck瓶颈层
    """
    def __init__(self, input_channels, output_channels, use_1x1conv=False, strides=1):
        """
        :param input_channels: 输入的通道数
        :param output_channels: 输出的通道数
        :param use_1x1conv: 是否使用1x1的卷积核[默认不使用]
        :param strides: 滑动步长
        """
        super().__init__()
        mid_channels = input_channels
        if input_channels / 4 != 0:
            mid_channels = int(mid_channels / 4)
        self.conv1 = nn.Conv2d(input_channels, mid_channels, kernel_size=5, padding=2, stride=strides)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(mid_channels, output_channels, kernel_size=5, padding=2, stride=strides)
        if use_1x1conv:
            self.conv4 = nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=strides)
        else:
            self.conv4 = None
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.bn3 = nn.BatchNorm2d(output_channels)

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = F.relu(self.bn2(self.conv2(y)))
        y = F.relu(self.bn3(self.conv3(y)))
        if self.conv4:
            x = self.conv4(x)
        y = y + x
        return y


class MP(nn.Module):
    def __init__(self, k=2):
        super(MP, self).__init__()
        self.m = nn.MaxPool2d(kernel_size=(1,k), stride=(1,k))

    def forward(self, x):
        return self.m(x)


class Transition_Block(nn.Module):
    def __init__(self, c1, c2):
        super(Transition_Block, self).__init__()
        self.cv1 = nn.Conv2d(c1, c2, 1, 1)
        self.cv2 = nn.Conv2d(c1, c2, 3, 1,2)
        self.cv3 = nn.Conv2d(c2, c2, 3, 2)
        self.cv4 = nn.Conv2d(c2*2, c2, 5, 2, 2)
        self.mp = MP()

    def forward(self, x):
        x_1 = self.mp(x)
        x_1 = self.cv1(x_1)

        x_2 = self.cv2(x)
        x_2 = self.cv3(x_2)
        x = torch.cat([x_2, x_1], 1)
        x = self.cv4(x)
        return x


class ResNet(nn.Module):
    """
    残差网络
    """
    _loss = []
    _loss4epochs = []
    model_weights = []
    conv_layers = []

    def __init__(self):
        super().__init__()
        self._loss.clear()
        self._loss4epochs.clear()
        self.bl1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                                 nn.BatchNorm2d(64), nn.ReLU(),
                                 nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.bl2 = nn.Sequential(*self.res_layer_maker(64, 64, 2, first_block=True))
        self.TB = Transition_Block(64,64)
        self.bl3 = nn.Sequential(*self.res_layer_maker(64, 128, 2))
        self.bl4 = nn.Sequential(*self.res_layer_maker(128, 256, 2))
        self.bottleneck = BottleNeck(256, 256)
        self.bl5 = nn.Sequential(*self.res_layer_maker(256, 512, 2))
        self.pool1 = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(512, 3)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.bl1(x)
        x = self.bl2(x)
        x = self.TB(x)
        x = self.bl3(x)
        x = self.bl4(x)
        x = self.bottleneck(x)
        x = self.bl5(x)
        x = self.linear(self.flatten(self.pool1(x)))
        return x

    @staticmethod
    def res_layer_maker(input_channels, output_channels, num_residuals,
                        first_block=False):
        """
        调用残差块，对第一个残差块进行处理（1x1卷积核），并连接重复的残差块

        :param input_channels: 输入的通道数
        :param output_channels: 输出的通道数
        :param num_residuals: 残差块重复的次数
        :param first_block: 是否是第一个残差块
        :return: 处理好的残差块
        """
        blk = []
        for i in range(num_residuals):
            if i == 0 and not first_block:
                blk.append(ResBlock(input_channels, output_channels,
                                    use_1x1conv=True, strides=2))
            else:
                blk.append(ResBlock(output_channels, output_channels))
        # blk.append(SEAttention(output_channels,1))
        return blk


class SEAttention(nn.Module):
    def __init__(self, channel=512, reduction=16):
        super().__init__()
        # 对空间信息进行压缩
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 经过两次全连接层，学习不同通道的重要性
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 取出batch size和通道数
        b, c, _, _ = x.size()
        # b,c,w,h -> b,c,1,1 -> b,c 压缩与通道信息学习
        y = self.avg_pool(x).view(b, c)
        # b,c->b,c->b,c,1,1
        y = self.fc(y).view(b, c, 1, 1)
        # 激励操作
        return x * y.expand_as(x)