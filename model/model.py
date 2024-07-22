import math

import torch
import torch.nn as nn

class channel_aggregation(nn.Module):
    def __init__(self, in_channel, ratio=4):
        super(channel_aggregation, self).__init__()
        self.conv = nn.Conv1d(in_channel, in_channel//2,3,1,1)


    # 前向传播
    def forward(self, inputs):



        outputs = inputs * x

        return outputs


def autopad(k, p=None):
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class SiLU(nn.Module):
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)


class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=SiLU()):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        self.conv = nn.Conv1d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm1d(c2, eps=0.001, momentum=0.03)
        self.act = nn.LeakyReLU(0.1, inplace=True) if act is True else (
            act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))

class Multi_Concat_Block(nn.Module):
    def __init__(self, c1, c2, c3, n=4, e=1, ids=[0]):
        super(Multi_Concat_Block, self).__init__()
        c_ = int(c2 * e)

        self.ids = ids
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = nn.ModuleList(
            [Conv(c_ if i == 0 else c2, c2, 3, 1) for i in range(n)]
        )
        self.cv4 = Conv(c_ * 2 + c2 * (len(ids) - 2), c3, 1, 1)

    def forward(self, x):
        x_1 = self.cv1(x)
        x_2 = self.cv2(x)

        x_all = [x_1, x_2]
        for i in range(len(self.cv3)):
            x_2 = self.cv3[i](x_2)
            x_all.append(x_2)

        out = self.cv4(torch.cat([x_all[id] for id in self.ids], 1))
        return out

class net(nn.Module):
    def __init__(self):
        super(net, self).__init__()
        self.processing = nn.Sequential(
            nn.Conv1d(1, 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(4),
            nn.LeakyReLU(),
            nn.Conv1d(4, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(),
            Multi_Concat_Block(8,16,32)
        )

        self.encoder = nn.Sequential(
            nn.Conv1d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(),
            nn.Conv1d(16, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(),
            nn.Conv1d(8, 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(4),
            nn.LeakyReLU()
        )

        self.channel_aggregation = channel_aggregation(4)

        self.decoder = nn.Sequential(
            nn.Conv1d(4, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(),
            nn.Conv1d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(),
            nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU()
        )
        self.classification = nn.Sequential(
            nn.Conv1d(32, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(),
            nn.Conv1d(8, 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(2),
            nn.LeakyReLU(),
            nn.Conv1d(2, 1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(1),
            nn.LeakyReLU(),
            nn.Linear(512, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 3)
        )
    def forward(self, x):
        x = self.processing(x)
        x = self.encoder(x)
        x = self.channel_aggregation(x)
        x = self.decoder(x)
        x = self.classification(x)
        return x.squeeze()