"""
@author: hx
@func: 数据收集与处理
A7C2网络中，不同算法在不同时隙的调度结果
"""

import numpy as np
import matplotlib.pyplot as plt

# 修改greedy 从300 -> 200
greedy = [673, 773, 768, 756, 747]
greedy = [781, 791, 775, 775]

tabu = [776, 768, 747, 740]
q_learning = [657, 745, 805, 813]
DQN = [604, 732, 798, 820]
GQ_DQN = [909, 929, 931, 924]

# 小修前
# values1 = [781, 776, 657, 604, 909]
# values2 = [791, 768, 745, 749, 929]
# values3 = [775, 747, 805, 839, 931]
# values4 = [775, 740, 813, 860, 924]


import matplotlib.pyplot as plt
import numpy as np

# 数据
categories = ['greedy', 'tabu-search', 'q_learning', 'DQN', 'GQ_DQN']
# values1 = [676, 681, 657, 604, 909]
# values2 = [668, 691, 745, 749, 929]
# values3 = [647, 675, 805, 839, 931]
# values4 = [640, 675, 813, 860, 924]
values1 = [642, 665, 657, 604, 909]
values2 = [669, 698, 745, 749, 929]
values3 = [748, 767, 805, 839, 931]
values4 = [739, 757, 813, 860, 924]

# 设置柱子的宽度
bar_width = 0.2

# 设置柱子的位置
x = np.arange(len(categories))

# 绘制柱状图
plt.bar(x - 3*bar_width/2, values1, bar_width, label='Cycle=1ms', edgecolor="black")
plt.bar(x - bar_width/2, values2, bar_width, label='Cycle=2ms', edgecolor="black")
plt.bar(x + bar_width/2, values3, bar_width, label='Cycle=4ms', edgecolor="black")
plt.bar(x + 3*bar_width/2, values4, bar_width, label='Cycle=5ms', edgecolor="black")
plt.ylim(0, 1000)
plt.grid(True, linestyle='--', linewidth=0.5, color='gray', zorder=0)
# 设置 x 轴标签和标题
plt.xlabel('Algorithms', fontsize=14)
plt.ylabel('Scheduled Flow Number', fontsize=14)
plt.title("A7C2Net | 1000 Flows | Periods=[20ms,40ms,80ms]", fontsize=14)

# 设置 x 轴刻度标签
plt.xticks(x, categories)

plt.ylim(400, 1000)
# 添加图例
# plt.legend(loc='lower left')
plt.legend(loc='upper left')
plt.savefig('picture5.svg')
# 显示图形
plt.show()






