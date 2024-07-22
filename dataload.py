import csv
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

def datanorm(x):
    for i in range(np.shape(x)[0]):
        x[i] = (x[i] - np.min(x[i])) / (np.max(x[i]) - np.min(x[i]))
    return x
def split_dataset(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        train_size=0.7,
                                                        random_state=42,
                                                        stratify=y)

    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test,
                                                    train_size=2/3,
                                                    random_state=42,
                                                    stratify=y_test)

    data = [X_train, X_test, X_val]
    labels = [y_train, y_test, y_val]

    return data, labels


def load():
    data = []
    lable = []

    with open('./database/all_data_epileptic_seizures.csv') as f:
        reader = csv.reader(f)
        next(reader)

        for row in reader:
            signal = row[1:-1]
            if row[-1] == '1' or row[-1] == '2':
                label = 0
            elif row[-1] == '3' or row[-1] == '4':
                label = 1
            else:
                label = 2

            signal = np.array(signal, dtype=float)

            for i in range(16):
                data.append(signal[i*256:(i+1)*256])
                lable.append(label)

    data = np.array(data)
    lable = np.array(lable)

    data = datanorm(data)
    data, labels = split_dataset(data, lable)

    train_set = TensorDataset(torch.FloatTensor(data[0]), torch.LongTensor(labels[0]))
    val_set = TensorDataset(torch.FloatTensor(data[1]), torch.LongTensor(labels[1]))
    test_set = TensorDataset(torch.FloatTensor(data[2]), torch.LongTensor(labels[2]))

    train_loader = DataLoader(train_set, batch_size=20, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=20, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=20, shuffle=True, num_workers=0)

    return train_loader, val_loader, test_loader

def overlook_wnfg2(data):
    field = 10
    samples = data.shape[0]
    length = data.shape[1]
    adjmatrix = np.zeros((samples, length, length))
    for k in range(samples):
        if k%100 == 0 : print(k)
        for i in range(1, length - field):
            for j in range(i + 1, i + 1 + field):
                if data[k][i] > data[k][j]:
                    adjmatrix[k][i][j] = (data[k][i] - data[k][j])/abs(j - i)
                    adjmatrix[k][j][i] = -1 * ((data[k][i] - data[k][j])/abs(j - i))
                elif data[k][i] < data[k][j]:
                    adjmatrix[k][i][j] = -1 * ((data[k][j] - data[k][i])/abs(j - i))
                    adjmatrix[k][j][i] = (data[k][j] - data[k][i])/abs(j - i)
                else:
                    adjmatrix[k][i][j] = 0
                    adjmatrix[k][j][i] = 0

        for i in range(length - field + 1, length):
            for j in range(length - field + 1, length):
                if data[k][i] > data[k][j]:
                    adjmatrix[k][i][j] = (data[k][i] - data[k][j])/abs(j - i)
                    adjmatrix[k][j][i] = -1 * ((data[k][i] - data[k][j])/abs(j - i))
                elif data[k][i] < data[k][j]:
                    adjmatrix[k][i][j] = -1 * ((data[k][j] - data[k][i])/abs(j - i))
                    adjmatrix[k][j][i] = (data[k][j] - data[k][i])/abs(j - i)
                else:
                    adjmatrix[k][i][j] = 0
                    adjmatrix[k][j][i] = 0
    return adjmatrix


def load_graph():
    data = []
    lable = []

    with open('./database/all_data_epileptic_seizures.csv') as f:
        reader = csv.reader(f)
        next(reader)

        for row in reader:
            signal = row[1:-1]
            if row[-1] == '1' or row[-1] == '2':
                label = 0
            elif row[-1] == '3' or row[-1] == '4':
                label = 1
            else:
                label = 2

            signal = np.array(signal, dtype=float)

            for i in range(16):
                data.append(signal[i*256:(i+1)*256])
                lable.append(label)

    data = np.array(data)
    lable = np.array(lable)

    data = datanorm(data)
    data = overlook_wnfg2(data)
    data, labels = split_dataset(data, lable)

    train_set = TensorDataset(torch.FloatTensor(data[0]), torch.LongTensor(labels[0]))
    val_set = TensorDataset(torch.FloatTensor(data[1]), torch.LongTensor(labels[1]))
    test_set = TensorDataset(torch.FloatTensor(data[2]), torch.LongTensor(labels[2]))

    train_loader = DataLoader(train_set, batch_size=20, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=20, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=20, shuffle=True, num_workers=0)

    return train_loader, val_loader, test_loader