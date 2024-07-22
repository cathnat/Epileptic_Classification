import torch.nn as nn
import torch.nn.functional as F
import torch
class GNNnet(torch.nn.Module):
    def __init__(self, node_number, batch_size, k_hop):
        super(GNNnet, self).__init__()
        self.node_number = node_number
        self.batch_size = batch_size
        self.k_hop = k_hop
        self.aggregate_weightT = torch.nn.Parameter(torch.ones(1, 1, node_number))

        self.features = nn.Sequential(
            torch.nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            torch.nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            torch.nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            torch.nn.Linear(256, 256),  # 12
            nn.ReLU(inplace=True),
            torch.nn.Linear(256, 3),
        )

    def forward(self, x_T):
        tmp_x_T = x_T
        for _ in range(self.k_hop):
            tmp_x_T = torch.matmul(tmp_x_T, x_T)
        x_T = torch.matmul(self.aggregate_weightT, tmp_x_T)
        x = self.features(x_T.view(x_T.size(0), -1))

        return x.squeeze()
