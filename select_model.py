import numpy as np
import torch
import pandas as pd
from model.DenseNet import DenseNet
from model.resnet import MyResNet
from model.InceptionV2 import InceptionV2
from model.Convnext import ConvNeXt
from model.GCN import GCN
from model.SSGCNnet import SSGCNnet
from model.GNN import GNNnet
from  model.ARNN import ARNN,preprocess_adj

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def pick_models(netname):
    if netname == 'DenseNet':
        model = DenseNet(num_classes=3).to(device)
    elif netname == 'ResNet':
        model = MyResNet().to(device)
    elif netname == 'InceptionV2':
        model = InceptionV2(3).to(device)
    elif netname == 'ConvNeXt':
        model = ConvNeXt(depths=[3, 3, 9, 3],
                     dims=[96, 192, 384, 768],
                     num_classes=3).to(device)
    elif netname == 'GNN':
        model = GNNnet(256, 64, 3).to(device)
    elif netname == 'SSGCNnet':
        model = SSGCNnet(256, 64, 1).to(device)
    elif netname == 'ARNN':
        df = pd.read_excel('./model/init_adj_2a.xlsx')
        Abf = df.iloc[:, 1:].values
        A = preprocess_adj(Abf)
        # A = np.ones((60,60))
        A = np.float32(A)
        A = torch.from_numpy(A)
        A = A.to(device)
        model = ARNN((1, 256), A).to(device)
    else:
        model = GCN().to(device)

    return model