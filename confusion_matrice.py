import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 示例数据
confusion_matrices = [
    np.array([[80, 18, 1], [18, 64, 18], [8, 34, 58]]),
    np.array([[86, 12, 2], [15, 72, 13], [5, 31, 64]]),
    np.array([[75, 22, 3], [10, 72, 18], [6, 31, 63]]),
    np.array([[89, 7, 4], [29, 49, 22], [6, 16, 78]]),
    np.array([[70, 23, 7], [13, 74, 13], [9, 43, 48]]),
    np.array([[69, 31, 0], [57, 41, 2], [48, 52, 0]]),
    np.array([[87, 11, 2], [18, 67, 15], [4, 26, 70]]),
    np.array([[81, 16, 3], [12, 78, 10], [0, 20, 80]])

]

titles = [
    "DenseNet",
    "ResNet",
    "InceptionV2",
    "ConvNeXt",
    "GNN",
    "EEG-ARNN",
    "SSGCNet",
    "Ours"
]

# 类别标签
class_labels = ['N', 'I', 'S']

# 创建图表
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.flatten()

for i in range(len(confusion_matrices)):
    cm = confusion_matrices[i]
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_normalized, annot=True, fmt=".2%", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels, ax=axes[i])
    axes[i].set_xlabel("Predicted classes")
    axes[i].set_ylabel("Actual classes")

# 添加标题到子图下方
for i, ax in enumerate(axes):  # 最后一个是被删除的，跳过
    ax.set_title('')
    ax.text(0.5, -0.25, titles[i], transform=ax.transAxes, ha='center', va='top', fontsize=12)

# 调整子图间距
plt.subplots_adjust(left=0.05, right=0.95, top=0.90, bottom=0.10, wspace=0.3, hspace=0.4)

plt.savefig('./all_confusion_matrices.png', dpi=500)
plt.show()
