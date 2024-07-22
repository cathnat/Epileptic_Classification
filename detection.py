import random
import time
from thop import profile
import numpy as np
import torch

from dataload import overlook_wnfg2
from select_model import pick_models

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":

    modelname = 'GNN'
    net = pick_models(modelname)

    data = torch.rand((20, 256))
    time1 = time.time()
    data = overlook_wnfg2(data)
    time2 = time.time()
    data = torch.tensor(data, dtype=torch.float32)
    start_time = time.time()
    for _ in range(1000):
        out = net(data)
    end_time = time.time()
    execution_time = (end_time - start_time) / 1000 + (time2 - time1)
    print("平均运行时间为:", execution_time, "秒")

    with torch.no_grad():
        flops, params = profile(net, inputs=(torch.tensor(data, dtype=torch.float32),))
        print("Total parameters:", params)
        print("Total FLOPs:", flops)

