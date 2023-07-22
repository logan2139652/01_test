"""
@author: hx
@func: 数据收集与处理
在线调度算法时间
Greedy
GQ-DQN
在线模型性能并不好
"""

import numpy as np
import matplotlib.pyplot as plt


# greedy_A4C1 = [0.7055, 0.8207, 0.9595, 1.0827, 1.1700, 1.3015, 1.4255]
# greedy_A7C2 = [0.8179, 1.0217, 1.2760, 1.4848, 1.8778, 2.3245, 2.9353]
#
# GQ_DQN_A4C1 = [0.8932, 1.1246, 1.4365, 1.7197, 2.0356, 2.3361, 2.6664]
# GQ_DQN_A7C2 = [1.1219, 1.4796, 1.9019, 2.4175, 3.0034, 3.7909, 4.4565]
#
# flows = [400, 500, 600, 700, 800, 900, 1000]


# 数据，时隙选择40ms，greedy20ms，资源受限
greedy_A4C1 = [0.2679964542388916, 0.6082243919372559, 0.8532187938690186,
               1.1620781421661377, 1.5374269485473633]
greedy_A7C2 = [0.29012598991394043, 0.632786750793457, 1.1033782958984375,
               1.7544348239898682, 2.911745548248291]
# greedy_A7C2 = [0.4508, 0.9298, 2.156, 4.1672, 6.7974]
greedy_A4C1_flows = [100, 292, 446, 547, 622]
greedy_A7C2_flows = [100, 299, 486, 627, 733]
GQ_DQN_A4C1 = [0.20452086448669434, 0.8199710369110107, 1.10597562789917, 1.8210341930389404, 2.407428026199341]
GQ_DQN_A7C2 = [0.19576201438903809, 0.41754722595214844, 0.9958308029174805, 1.407428026199341, 2.272573471069336]
GQ_DQN_A4C1_flows = [100, 293, 447, 571, 674]
GQ_DQN_A7C2_flows = [100, 300, 481, 630, 754]

flows = [100, 300, 500, 700, 900]

for i,j in enumerate(flows):
    # greedy_A4C1[i] = greedy_A4C1[i] / greedy_A4C1_flows[i] * 1000
    # greedy_A7C2[i] = greedy_A7C2[i] / greedy_A7C2_flows[i] * 1000
    # GQ_DQN_A4C1[i] = GQ_DQN_A4C1[i] / GQ_DQN_A4C1_flows[i] * 1000
    # GQ_DQN_A7C2[i] = GQ_DQN_A7C2[i] / GQ_DQN_A7C2_flows[i] * 1000
    greedy_A4C1[i] = greedy_A4C1[i] / j * 1000
    greedy_A7C2[i] = greedy_A7C2[i] / j * 1000
    GQ_DQN_A4C1[i] = GQ_DQN_A4C1[i] / j * 1000
    GQ_DQN_A7C2[i] = GQ_DQN_A7C2[i] / j * 1000



markers = ['o', 's', '*', 'x', '^']
plt.plot(greedy_A4C1, linewidth=2, label='greedy in A4C1', color='red', marker=markers[0])
plt.plot(greedy_A7C2, linewidth=2, label='greedy in A7C2', color='green', marker=markers[0])
plt.plot(GQ_DQN_A4C1, linewidth=2, label='GQ-DQN in A4C1', color='red', marker=markers[1])
plt.plot(GQ_DQN_A7C2, linewidth=2, label='GQ-DQN in A7C2', color='green', marker=markers[1])


plt.title(" Cycle=2ms | Periods=[20,40,80]", fontsize=14)
plt.ylabel('Average Execution time (millisecond)', fontsize=14)
plt.xlabel('Number of Time-Sensitive Flows', fontsize=14)

plt.plot([0, 4], [3.5, 3.5], linewidth=2, label='upper bound', color='black', linestyle='--')
plt.plot([0, 4], [1.3, 1.3], linewidth=2, label='lower bound', color='blue', linestyle='--')
plt.xlim(0, 4, 1)
plt.xticks(np.arange(5))
plt.ylim([0, 5])
plt.legend(loc='upper right', ncol=2)
plt.savefig("picture8.svg")
plt.show()

"""
模拟在线调度下流量场景，网络资源已被占用部分，突然出现新的流量，如何快速调度
时隙选择4ms
flow = [500, 1000, 1500, 2000]
# 资源不足
GQ = [0.41279101371765137, 1.3040494918823242, 1.43560791015625, 2.268080234527588]
GQ_flows = [494, 844, 1069, 1257]
# 资源足够
GQ_times = [0.2856173515319824, 0.5939846038818359, 0.8781542778015137, 1.1914737224578857]

for i,j in enumerate(flow):
    GQ_times[i] = GQ_times[i] / j * 1000
    GQ

markers = ['o', 's', '*', 'x', '^']
plt.plot(GQ_times, linewidth=2, label='greedy in A4C1', color='red', marker=markers[0])
plt.show()

    greedy_A4C1[i] = greedy_A4C1[i] / j * 1000
    greedy_A7C2[i] = greedy_A7C2[i] / j * 1000
    GQ_DQN_A4C1[i] = GQ_DQN_A4C1[i] / j * 1000
    GQ_DQN_A7C2[i] = GQ_DQN_A7C2[i] / j * 1000
"""