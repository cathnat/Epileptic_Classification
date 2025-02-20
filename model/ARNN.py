from thop import profile
from torch.nn import functional as F
import numpy as np
import pandas as pd

import torch
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class GCN_layer(nn.Module):

    def __init__(self, signal_shape, bias=False):
        super(GCN_layer, self).__init__()

        # input_shape=(node,timestep)
        self.W = nn.Parameter(torch.ones(signal_shape[0], signal_shape[0]), requires_grad=True)
        self.theta = nn.Parameter(torch.randn(signal_shape[1]), requires_grad=True)
        self.b = nn.Parameter(torch.zeros([1, 1, 1, signal_shape[1]]), requires_grad=True)
        self.bias = bias

        # self.params = nn.ParameterDict({
        #         'W': nn.Parameter(torch.rand(signal_shape[0], signal_shape[0]), requires_grad=True),
        #         'theta': nn.Parameter(torch.rand(signal_shape[1]), requires_grad=True)
        # })

    def forward(self, Adj_matrix, input_features):

        # G = torch.from_numpy(Adj_matrix).type(torch.FloatTensor)

        hadamard = Adj_matrix
        hadamard= hadamard.to("cpu")
        input_features =input_features.to("cpu")

        aggregate = torch.einsum("ce,abed->abcd", hadamard, input_features).to(device)
        output = torch.einsum("abcd,d->abcd", aggregate, self.theta)

        if self.bias == True:
            output = output + self.b

        return output
class ARNN(nn.Module):
    def __init__(self, input_shape, A):
        super(ARNN, self).__init__()
        self.h_n = input_shape[0]
        A = A.cuda()
        self.A = nn.Parameter(A, requires_grad=True).to("cuda")


        self.conv1 = nn.Conv2d(1, 16, kernel_size=(1, 16), stride=(1, 1))
        self.norm1 = nn.BatchNorm2d(16)
        self.ELU1 = nn.ELU(inplace=True)
        self.drop1 = nn.Dropout2d(0.25)

        self.gconv2 = GCN_layer((60, 256), bias=True)
        self.norm2 = nn.BatchNorm2d(16)
        self.ELU2 = nn.ELU(inplace=True)
        self.drop2 = nn.Dropout2d(0.25)

        self.dconv3 = nn.Conv2d(16, 16, kernel_size=(1, 8), stride=(1, 1), groups=16)
        self.norm3 = nn.BatchNorm2d(16)
        self.ELU3 = nn.ELU(inplace=True)
        self.pool1 = nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4))
        self.drop3 = nn.Dropout2d(0.25)

        self.gconv4 = GCN_layer((60, 64), bias=True)
        self.norm4 = nn.BatchNorm2d(16)
        self.ELU4 = nn.ELU(inplace=True)
        self.drop4 = nn.Dropout2d(0.25)

        self.dconv5 = nn.Conv2d(16, 16, kernel_size=(1, 8), stride=(1, 1), groups=16)
        self.norm5 = nn.BatchNorm2d(16)
        self.ELU5 = nn.ELU(inplace=True)
        self.pool2 = nn.AvgPool2d(kernel_size=(1, 2), stride=(1, 2))
        self.drop5 = nn.Dropout2d(0.25)

        self.dconv6 = nn.Conv2d(16, 16, kernel_size=(self.h_n, 1), stride=(1, 1), groups=16)
        self.pconv6 = nn.Conv2d(16, 1, kernel_size=(1, 1), groups=1)
        self.norm6 = nn.BatchNorm2d(1)
        self.ELU6 = nn.ELU(inplace=True)
        self.pool3 = nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4))
        self.drop6 = nn.Dropout2d(0.25)

        self.gconv7 = GCN_layer((60, 32), bias=True)
        self.norm7 = nn.BatchNorm2d(16)
        self.ELU7 = nn.ELU(inplace=True)
        self.drop7 = nn.Dropout2d(0.25)

        self.linear1 = nn.Linear(176, 3)


    def forward(self, input):
        x = F.pad(input, pad=(7, 8, 0, 0))
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.ELU1(x)
        x = self.drop1(x)

        x = self.gconv2(self.A, x)
        x = self.norm2(x)
        x = self.ELU2(x)
        x = self.drop2(x)

        x = F.pad(x, pad=(3, 4, 0, 0))
        x = self.dconv3(x)
        x = self.norm3(x)
        x = self.ELU3(x)
        x = self.pool1(x)
        x = self.drop3(x)

        x = self.gconv4(self.A, x)
        x = self.norm4(x)
        x = self.ELU4(x)
        x = self.drop4(x)

        x = F.pad(x, pad=(3, 4, 0, 0))
        x = self.dconv5(x)
        x = self.norm5(x)
        x = self.ELU5(x)
        x = self.pool2(x)
        x = self.drop5(x)

        x = self.gconv7(self.A, x)
        x = self.norm7(x)
        x = self.ELU7(x)
        x = self.drop7(x)

        x = self.dconv6(x)
        x = self.pconv6(x)
        x = self.norm6(x)
        x = self.ELU6(x)
        x = self.pool3(x)
        x = self.drop6(x)

        x = x.view(x.size()[0],-1)
        x = self.linear1(x)
        # x = F.softmax(x, dim=1)

        return x
def normalize_adj(adj):
    d = np.diag(np.power(np.array(adj.sum(1)), -0.5).flatten(), 0)
    a_norm = adj.dot(d).transpose().dot(d)
    return a_norm
def preprocess_adj(adj):
    adj = adj + np.eye(adj.shape[0])
    adj = normalize_adj(adj)
    return adj
# if __name__ == '__main__':
#     df = pd.read_excel('./init_adj_2a.xlsx')
#     Abf = df.iloc[:, 1:].values
#     A = preprocess_adj(Abf)
#     # A = np.ones((60,60))
#     A = np.float32(A)
#     A = torch.from_numpy(A)
#     A=A.to("cpu")
#
#     net = ARNN((1, 256), A)
#     # 输入数据
#     input_size = (20, 1, 256)
#     data = torch.rand(input_size)
#
#     import time
#
#     start_time = time.time()
#
#     for _ in range(500):
#         results = net(data)
#
#     end_time = time.time()
#
#     execution_time = (end_time - start_time)/500
#     print("运行500次的时间为:", execution_time, "秒")
#
#     # 计算 FLOPs 和参数数量
#     input_data = (data,)
#     flops, params = profile(net, input_data)
#
#     print("Total parameters:", params)
#     print("Total FLOPs:", flops)