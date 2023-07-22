"""
@author: hx
@func: 数据收集与处理
A4C1网络中，不同算法在三种不同数据集中的调度结果
数据集有三种，分别为周期短而小数据，周期大数据，周期均衡数据
"""

import numpy as np
import matplotlib.pyplot as plt

# 选取3个数据集，每个数据集对算法的影响
# 'greedy', 'tabu-search', 'q_learning', 'DQN', 'GQ_DQN'
# 均衡 673 714 750 809 858
# 大  766 831 845 888 917
# 小  624 595 620 648 722

# 数据
# greedy (833 not 673)
categories = ['greedy', 'tabu-search', 'q_learning', 'DQN', 'GQ_DQN']
values1 = [624, 595, 620, 648, 722]
values2 = [673, 714, 750, 809, 858]
values3 = [766, 831, 845, 888, 917]

# 设置柱子的宽度
bar_width = 0.2

# 设置柱子的位置
x = np.arange(len(categories))

# 绘制柱状图
plt.bar(x - bar_width, values1, bar_width, label='Periods=[20ms, 40ms]', edgecolor="black")
plt.bar(x, values2, bar_width, label='Periods=[20ms, 40ms, 80ms]', edgecolor="black")
plt.bar(x + bar_width, values3, bar_width, label='Periods=[40ms, 80ms]', edgecolor="black")

plt.grid(True, linestyle='--', linewidth=0.5, color='gray', zorder=0)
# 设置 x 轴标签和标题
plt.xlabel('Algorithms', fontsize=14)
plt.ylabel('Scheduled Flow Number', fontsize=14)
plt.title("A4C1Net | DataSet1/2/3 | Periods=[20ms,40ms,80ms]", fontsize=14)

# 设置 x 轴刻度标签
plt.xticks(x, categories)
plt.ylim(400, 1000)
# 添加图例
# plt.legend(loc='lower left')
plt.legend(loc='upper left')
plt.savefig('picture4.svg')
# 显示图形
plt.show()






