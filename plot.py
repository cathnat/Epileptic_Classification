import matplotlib.pyplot as plt
import numpy as np

# 数据
pruning_methods = ['1', '2', '3', '4', '5', '6', '7', '8', '9']
accuracy = [76.50, 76.87, 77.75, 79.50, 78.12, 77.37, 75.12, 75.25, 75.87]
colors = ['#4B0082', '#4682B4', '#2E8B57', '#B22222', '#BDB76B', '#6A5ACD', '#696969', '#8B4513', '#DAA520']

plt.rcParams.update({'font.size': 18, 'font.family': 'Times New Roman'})

# 创建图表
fig, ax = plt.subplots()

bars = ax.bar(pruning_methods, accuracy, color=colors)

# 显示数值
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval, round(yval, 2), ha='center', va='bottom')

# 设置标签和标题
ax.set_xlabel('Pruning Methods')
ax.set_ylabel('Accuracy (%)')
ax.set_ylim(73, 80)

# 调整图表的边距
plt.subplots_adjust(bottom=0.13)

# 显示图表
plt.savefig('./result/prune.png', dpi=500)