import matplotlib.pyplot as plt
import numpy as np

# 数据
pruning_methods = ['10', '20', '30', '40']
accuracy = [76.5, 78.12, 75.87, 74.25]
colors = ['#1a5276', '#8b0000', '#196f3d', '#d35400']

plt.rcParams.update({'font.size': 15, 'font.family': 'Times New Roman'})

# 创建图表
fig, ax = plt.subplots()

bars = ax.bar(pruning_methods, accuracy, color=colors, width=0.5)

# 显示数值
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval, round(yval, 2), ha='center', va='bottom')

# 设置标签和标题
ax.set_xlabel('Pruning Rate (%)')
ax.set_ylabel('Accuracy (%)')
ax.set_ylim(72, 79)

# 显示图表
plt.savefig('./result/prune_all.png', dpi=500)