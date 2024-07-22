import math
import torch.nn.functional as F
import torch
import torch.nn as nn

class SelectiveConv1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(SelectiveConv1D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.conv = nn.Conv1d(in_channels // 2, out_channels // 2, kernel_size, stride, padding)

    def forward(self, x):
        x1, x2 = torch.split(x, self.in_channels // 2, dim=1)

        x1 = self.conv(x1)

        if x1.shape[-1] != x2.shape[-1]:
            x2 = F.pad(x2, (0, x1.shape[-1] - x2.shape[-1]))

        x = torch.cat((x1, x2), dim=1)

        return x

class GCN(nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        self.processing = nn.Sequential(
            nn.Conv1d(1, 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(4),
            nn.LeakyReLU(0.2),
            nn.Conv1d(4, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool1d(128),
        )

        self.encoder = nn.Sequential(
            nn.Conv1d(8, 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(4),
            nn.LeakyReLU(0.2),
            nn.Conv1d(4, 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(4),
            nn.LeakyReLU(0.2),
            nn.Conv1d(4, 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(4),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool1d(32),
        )

        self.A = nn.Parameter(torch.ones(1, 1, 32))
        self.decoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.2),
            nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
            nn.Conv1d(32, 32, kernel_size=1, stride=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
        )

        self.classification = nn.Sequential(
            nn.Conv1d(4, 1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(1),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 16),
            nn.LeakyReLU(0.2),
            nn.Linear(16, 3)
        )
    def forward(self, x):
        x = self.processing(x)
        x = self.encoder(x)
        A = self.decoder(self.A)
        x = torch.matmul(x, A)
        x = self.classification(x)
        return x.squeeze()