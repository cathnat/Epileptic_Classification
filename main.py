import random
import numpy as np
import torch
from dataload import load, load_graph
from model.model import net
from sklearn.metrics import confusion_matrix,precision_score, recall_score, f1_score
from torch import nn, optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from select_model import pick_models

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def plot_heat_map(y_true, y_pred, modelname):
    cm = confusion_matrix(y_true, y_pred)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(10, 8))
    class_labels = ['N', 'I', 'S']
    sns.heatmap(cm, annot=True, fmt=".2%", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel("Predicted classes")
    plt.ylabel("Actual classes")
    plt.title("Confusion Matrix")
    plt.savefig('./result/' + modelname + '_heat_map.png', dpi=500)

def Accuracy_per_Category(predictions, labels, modelname):
    plt.figure()
    accuracy = []
    for i in range(3):
        total = sum(1 for p, l in zip(predictions, labels) if p == l == i)
        count = sum(1 for l in labels if l == i)
        acc = total / count if count > 0 else 0
        accuracy.append(acc)

    # 绘制直方图
    categories = ['normal phase', 'interictal phase', 'ictal phase']
    plt.bar(categories, accuracy, width=0.3)
    plt.ylabel('Accuracy')
    plt.title('Accuracy per Category')
    for i in range(len(categories)):
        plt.text(categories[i], accuracy[i], '{:.3f}'.format(accuracy[i]), ha='center', va='bottom')
    plt.savefig('./result/' + modelname + '_Accuracy_per_Category.png', dpi=500)

def train(net ,train_loader, val_loader, epoch, modelname):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    best_acc = 0
    loss_list = []
    acc_list = []
    for epoch in range(epoch):
        running_loss = 0.0
        train_acc = 0
        train_total = 0
        for i, (data, labels) in enumerate(tqdm(train_loader), 0):
            if modelname != 'SSGCNnet' and modelname != 'GNN':
                data = data.unsqueeze(dim=1).to(device)
            else:
                data = data.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = net(data)
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_acc += (predicted == labels).sum().item()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        acc_list.append(train_acc/train_total)
        loss_list.append(running_loss)
        print('epoch:{:d} loss = {:.5f}'.format(epoch, running_loss))
        acc = 0
        total = 0
        with torch.no_grad():
            for i, (data, labels) in enumerate(tqdm(val_loader), 0):
                if modelname != 'SSGCNnet' and modelname != 'GNN':
                    data = data.unsqueeze(dim=1).to(device)
                else:
                    data = data.to(device)
                labels = labels.to(device)
                outputs = net(data)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                acc += (predicted == labels).sum().item()
        print('epoch:{:d} Val Acc = {:.5f}'.format(epoch, acc/total))

        if(acc > best_acc):
            filepath = './model_save/' + modelname + '_best.pth'
            torch.save(obj=net.state_dict(), f=filepath)
            best_acc = acc
            print('best model save')

    epochs = range(1, len(loss_list) + 1)

    # 创建一个图形对象和两个子图
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    ax1.plot(epochs, loss_list, 'b-', label='Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss and Accuracy')
    ax1.grid(True)

    ax2.plot(epochs, acc_list, 'r-', label='Accuracy')
    ax2.set_ylabel('Accuracy')

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='lower right')
    plt.savefig('./result/' + modelname + '_Loss_accuracy.png', dpi=500)

def test(net, test_loader, modelname):
    checkpoint = torch.load('./model_save/' + modelname + '_best.pth')
    net.load_state_dict(checkpoint)
    acc = 0
    total = 0
    y_pred = []
    y_true = []
    for i, (data, labels) in enumerate(tqdm(test_loader), 0):
        if modelname != 'SSGCNnet' and modelname != 'GNN':
            data = data.unsqueeze(dim=1).to(device)
        else:
            data = data.to(device)
        labels = labels.to(device)
        outputs = net(data)
        _, predicted = torch.max(outputs.data, 1)
        y_pred.extend(predicted)
        y_true.extend(labels)
        total += labels.size(0)
        acc += (predicted == labels).sum().item()
    print('test Acc = {:.5f}'.format(acc/total))
    y_true = [tensor.to('cpu') for tensor in y_true]
    y_pred = [tensor.to('cpu') for tensor in y_pred]
    plot_heat_map(y_true, y_pred, modelname)
    Accuracy_per_Category(y_true, y_pred, modelname)
    recall = recall_score(y_true, y_pred, average='macro')
    precision = precision_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    print('test recall  = {:.5f}'.format(recall))
    print('test precision = {:.5f}'.format(precision))
    print('test f1_score = {:.5f}'.format(f1))
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":

    modelname = 'ARNN'
    net = pick_models(modelname)
    epoch = 50

    param_num = sum(p.numel() for p in net.parameters())
    print('total parameters  = {}'.format(param_num))

    if modelname == 'SSGCNnet' or modelname == 'GNN':
        train_loader, val_loader, test_loader = load_graph()
    else:
        train_loader, val_loader, test_loader = load()
    train(net, train_loader, val_loader, epoch, modelname)
    test(net, test_loader, modelname)