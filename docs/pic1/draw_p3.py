"""
@author: hx
@func: 数据收集与处理
A4C1网络中，不同算法在不同时隙的调度结果
设立时隙为40/20
"""

import numpy as np
import matplotlib.pyplot as plt

# 修改greedy 从300-》200
greedy = [673, 773, 768, 756, 747]
greedy = [653, 687, 691, 677, 673]

tabu = [649, 708, 710, 686, 682]
q_learning = [595, 657, 748, 756, 732]
DQN = [592, 774,  834, 842]
GQ_DQN = [803, 845, 861, 868, 859]
# values1 = [687, 708, 657, 592, 845]
# values2 = [691, 710, 748, 679, 861]
# values3 = [677, 686, 756, 756, 868]
# values4 = [673, 682, 732, 764, 859]

# 第二批
# values1 = [687, 708, 689, 592, 845]
# values2 = [691, 710, 748, 679, 861]
# values3 = [677, 686, 792, 756, 868]
# values4 = [673, 682, 799, 764, 859]

# 小修前
# values1 = [687, 708, 689, 592, 845]
# values2 = [691, 710, 748, 692, 861]
# values3 = [677, 686, 756, 792, 868]
# values4 = [673, 682, 764, 799, 859]

import matplotlib.pyplot as plt
import numpy as np

# 数据
categories = ['greedy', 'tabu-search', 'q_learning', 'DQN', 'GQ_DQN']
values1 = [587, 608, 689, 592, 845]
values2 = [591, 610, 748, 692, 861]
values3 = [577, 586, 756, 792, 868]
values4 = [573, 582, 764, 799, 859]

# 设置柱子的宽度
bar_width = 0.2

# 设置柱子的位置
x = np.arange(len(categories))
plt.grid(True, linestyle='--', linewidth=0.5, color='gray', zorder=0)

# 绘制柱状图
plt.bar(x - 3*bar_width/2, values1, bar_width, label='Cycle=1ms', edgecolor="black")
plt.bar(x - bar_width/2, values2, bar_width, label='Cycle=2ms', edgecolor="black")
plt.bar(x + bar_width/2, values3, bar_width, label='Cycle=4ms', edgecolor="black")
plt.bar(x + 3*bar_width/2, values4, bar_width, label='Cycle=5ms', edgecolor="black")

# 设置 x 轴标签和标题
plt.xlabel('Algorithms', fontsize=14)
plt.ylabel('Scheduled Flow Number', fontsize=14)
plt.title("A4C1Net | 1000 Flows | Periods=[20ms,40ms,80ms]", fontsize=14)

# 设置 x 轴刻度标签
plt.xticks(x, categories)
plt.ylim(400, 1000)
# 添加图例
# plt.legend(loc='lower left')
plt.legend(loc='upper left')
plt.savefig('picture3.svg')
# 显示图形
plt.show()






