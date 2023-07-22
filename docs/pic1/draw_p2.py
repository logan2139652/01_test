"""
@author: hx
@func: 算法离线执行时间
三（五）种算法不同网络执行时间, 可以加上Q与DQN，但是时间很长，直接飞起
"""

import numpy as np
import matplotlib.pyplot as plt
# 时隙选择50（5ms）
flow = [400, 500, 600, 700, 800, 900, 1000]
# flow = [100, 300, 500, 700, 900]

greedy_A4C1 = [0.7055, 0.8207, 0.9595, 1.0827, 1.1700, 1.3015, 1.4255]
greedy_A7C2 = [0.8179, 1.0217, 1.2760, 1.4848, 1.8778, 2.3245, 2.9353]
# 选取100次循环
tabu_A4C1 = [14.41, 17.03, 18.95]
tabu_A7C2 = [15.84, 19.27, 22.65]
# 选取4次循环
GQ_DQN_A4C1 = [0.8932, 1.1246, 1.4365, 1.7197, 2.0356, 2.3361, 2.6664]
GQ_DQN_A7C2 = [1.1219, 1.4796, 1.9019, 2.4175, 3.0034, 3.7909, 4.4565]
# 选取100次循环
# Q_learning_A4C1 = [209.8, 309.8, 209.8, 209.8, 209.8, 209.8, 209.8]
# Q_learning_A7C2 = [252.9, 352.9, 252.9, 252.9, 252.9, 252.9, 252.9]
Q_learning_A4C1 = [20.98, 22.86, 28.12]
Q_learning_A7C2 = [21.12, 25.14, 30.45]
# DQN
DQN_A4C1 = [114514]
DQN_A7C2 = [228899]

markers = ['o', 's', '*', 'x', '^']


plt.plot(greedy_A4C1, linewidth=2, label='greedy in A4C1', color='red', marker=markers[0])
plt.plot(greedy_A7C2, linewidth=2, label='greedy in A7C2', color='green', marker=markers[0])
plt.plot(tabu_A4C1, linewidth=2, label='tabu in A4C1', color='red', marker=markers[2])
plt.plot(tabu_A7C2, linewidth=2, label='tabu in A7C2', color='green', marker=markers[2])
plt.plot(GQ_DQN_A4C1, linewidth=2, label='GQ-DQN in A4C1', color='red', marker=markers[1])
plt.plot(GQ_DQN_A7C2, linewidth=2, label='GQ-DQN in A7C2', color='green', marker=markers[1])
plt.plot(Q_learning_A4C1, linewidth=2, label='Q-learning in A4C1', color='red', marker=markers[3])
plt.plot(Q_learning_A7C2, linewidth=2, label='Q-learning in A7C2', color='green', marker=markers[3])


plt.title("1000 Flows | Cycle=4ms | Periods=[20,40,80]", fontsize=14)
plt.ylabel('Scheduled Flow Number', fontsize=14)
plt.xlabel('Episodes(Iterations)', fontsize=14)
plt.xlim(0, 6)
plt.ylim(0, 5)
# 换横坐标值
x_labels = flow
x = np.arange(len(greedy_A4C1))
plt.xticks(x, x_labels)
# 加上legend
plt.legend(prop = {'size': 9}, loc='lower right', ncol=2)
# plt.savefig("picture2.svg")
# plt.show()


# 添加小图1
plt.axes([0.2, 0.6, 0.35, 0.25])  # 设置小图的位置和大小，左下角坐标为(0.2, 0.55)，宽度为0.3，高度为0.3
plt.plot(tabu_A4C1, linewidth=2, label='tabu in A4C1', color='red', marker=markers[2])
plt.plot(tabu_A7C2, linewidth=2, label='tabu in A7C2', color='green', marker=markers[2])
plt.plot(Q_learning_A4C1, linewidth=2, label='Q-learning in A4C1', color='red', marker=markers[3])
plt.plot(Q_learning_A7C2, linewidth=2, label='Q-learning in A7C2', color='green', marker=markers[3])
x_labels = [400, 500, 600]
x = np.arange(3)
plt.xticks(x, x_labels)
plt.xlim(0, 2)
plt.ylim(10, 35)

plt.savefig("picture2.svg")
plt.show()
