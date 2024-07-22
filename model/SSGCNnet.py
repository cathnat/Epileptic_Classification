import torch.nn as nn
import torch.nn.functional as F
import torch
class SSGCNnet(torch.nn.Module):
    def __init__(self, node_number, batch_size, k_hop):
        super(SSGCNnet, self).__init__()
        self.node_number = node_number
        self.batch_size = batch_size
        self.k_hop = k_hop
        self.aggregate_weightT = torch.nn.Parameter(torch.ones(1, 1, node_number))

        self.features = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(16, 32, kernel_size=9, stride=1, padding=4),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(32, 32, kernel_size=9, stride=1, padding=4),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(16 * 32, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 3),
        )

    def forward(self, x_T):
        x_T = torch.matmul(self.aggregate_weightT, x_T)

        x_T = self.features(x_T)
        x = self.classifier(x_T.view(x_T.size(0), -1))

        return x.squeeze()