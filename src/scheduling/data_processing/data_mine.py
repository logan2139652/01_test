"""
@ version 0.1
数据处理模块

@ version 1.0
完成数据提取与预处理设计
后续需要补充备选路径与数据清洗部分
数据集存在问题，需要进行补充
"""
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import math
import numpy as np

# .xlsx文件读取
# df = pd.read_excel(r'flow_set_cqf.xlsx')[:20]
# df = pd.read_excel(r'D:\Python\PyCharmWorkplace\CQF_CQCF_DQN\Date_set\flow200_simple.xlsx')[:200]
df = pd.read_excel(r'D:\Python\PyCharmWorkplace\ArticleOneCode\src\scheduling\data_set\flow2000_mixed.xlsx')[:2000]
# print(df)

# 网络流量信息提取
# 优先级提取并排序
# df_sort = df.sort_values(by=['prior'], ascending=[False])
df_sort = df
# print(df_sort)
# 周期提取
flow_period = df_sort['period'].tolist()
# print(flow_period)
# 最大允许传输时间提取
flow_deadline = df_sort['deadline'].tolist()
# print(flow_deadline)
# 数据包长度提取
flow_length = df_sort['length'].tolist()
# print(flow_length)

# 清洗
'''
for i, period in enumerate(flow_period):
    if period == 400:
        flow_period[i] = 300
    if period == 500:
        flow_period[i] = 600
'''
'''
简易拓扑
# 网络拓扑与路径编排

------1------3-------
|     |      |      |
0     |      |      5
|     |      |      |
------2------4-------

Network_node = ['10.0.0.1', '10.0.0.2', '10.0.0.3',
                '10.0.0.4', '10.0.0.5', '10.0.0.6']
Network_node_id = [num for num in range(len(Network_node))]
Network_node_dict = dict(zip(Network_node, Network_node_id))
# print(Network_node_dict)

edge_list = [(0, 1), (1, 3), (3, 5),
             (0, 2), (1, 2), (3, 4),
             (2, 4), (4, 5)]
G = nx.Graph()
G.add_edges_from(edge_list)
# print(list(G.nodes))
# nx.draw(G, with_labels=True, font_weight='bold')
# plt.show()
# print(list(nx.all_simple_paths(G, source=1, target=4)))
# print(nx.dijkstra_path(G, source=1, target=4))
'''

G = nx.Graph()
Edge_list = [("10.0.0.1", "10.0.0.2",10.0), ("10.0.0.1", "10.0.0.3",10.0),
             ("10.0.0.2","10.0.0.4",10.0), ("10.0.0.2","10.0.0.5",20.0),
             ("10.0.0.3","10.0.0.4",30.0), ("10.0.0.3","10.0.0.5",20.0),
             ("10.0.0.4","10.0.0.6",100.0), ("10.0.0.5","10.0.0.6",10.0),
             ("10.0.0.6","10.0.1.1",1000.0),
             ("10.0.1.1","10.0.1.2",10.0), ("10.0.1.1","10.0.1.3",10.0),
             ("10.0.1.2","10.0.1.4",10.0), ("10.0.1.2","10.0.1.5",20.0),
             ("10.0.1.3","10.0.1.4",30.0), ("10.0.1.3","10.0.1.5",20.0),
             ("10.0.1.4","10.0.1.6",100.0), ("10.0.1.5","10.0.1.6",10.0)]

G.add_weighted_edges_from(Edge_list)
# print(G)
Network_node = nx.nodes(G)
Network_node_id = [num for num in range(len(Network_node))]
Network_node_dict = dict(zip(Network_node, Network_node_id))


# 流量的源与目的
flow_source = df_sort['src']
flow_target = df_sort['dst']
# 将ip地址转换为节点编号
flow_source_id = []
flow_target_id = []
for source, target in zip(list(flow_source), list(flow_target)):
    flow_source_id.append(Network_node_dict[source])
    flow_target_id.append(Network_node_dict[target])

# 使用NetworkX显示源与目的节点之间的所有路径
flow_path = []
'''
for source, target in zip(flow_source_id, flow_target_id):
    flow_path.append(list(nx.shortest_path(G, source=source, target=target)))
#    flow_path.append(list(nx.all_simple_paths(G, source=source, target=target)))
'''
for source, target in zip(list(flow_source), list(flow_target)):
    # print(list(nx.all_simple_paths(G, source=source, target=target)))
    flow_path.append(list(nx.shortest_path(G, source=source, target=target)))
#    flow_path.append(list(nx.all_simple_paths(G, source=source, target=target)))
# 将path转成id

flow_path_id = []
for path in flow_path:
    temp = []
    for node in path:
        temp.append(Network_node_dict[node])
    flow_path_id.append(temp)
# print(flow_path_id)


# 为流量选择路径
# 目前简化
'''
for all_path in flow_path:
    continue
'''

# 根据流量路径得到端到端传输时延
long_path = [(5, 6), (6, 5)]
E2e_flow = []
for path in flow_path_id:
    e2e = []
    for num in range(len(path) - 1):
        if (path[num], path[num + 1]) in long_path:
            e2e.append(2)
        else:
            e2e.append(1)
    E2e_flow.append(e2e)


# 最大公约数（Greatest Common Divisor，GCD）函数
'''
def get_gcd(s):
    gcd = 0
    for length in range(len(s)):
        if length == 0:
            gcd = s[length]
        else:
            gcd = math.gcd(gcd, s[length])
    return gcd
'''
def get_gcd(num):
    """
    求任意多个数的最大公约数
    :param num:一个int的列表
    :return: 这个int列表中的最大公约数
    """
    minimum = max(num)
    for i in num:
        minimum = math.gcd(int(i), int(minimum))
    return int(minimum)

# 最小公倍数（Least Common Multiple, LCM）函数
'''
def get_lcm(s):
    a, b = s[0], s[1]
    # // 双除取整
    a = a // math.gcd(a, b) * b // math.gcd(a, b) * math.gcd(a, b)
    if len(s) > 2:
        for i in range(2, len(s)):
            b = s[i]
            a = a // math.gcd(a, b) * b // math.gcd(a, b) * math.gcd(a, b)
    return a
'''
def get_lcm(num):
    """
    求任意多个数的最小公倍数
    :param num: 一个int的列表
    :return: 这个int列表中的最小公倍数
    """
    # CQF中，LCM需要为2的倍数，方便规划调度
    # LCQF中，LCM需要为4的倍数
    minimum = 4
    for i in num:
        minimum = int(i) * int(minimum) / math.gcd(int(i), int(minimum))
    return int(minimum)


# 超周期与时隙计算
# 超周期 所有流量的最小公倍数
HyperCycle = get_lcm(flow_period)
# 最大时隙 所有流量的最大公约数
Max_C = get_gcd(flow_period)
# 最小时隙 最长链路之间的传输时间
Min_C = 10
# 时隙选择
cycle_list = []
for C in range(Min_C, Max_C + 1):
    if not Max_C % C:
        cycle_list.append(C)
print('cycle list', cycle_list)
Cycle = cycle_list[-1]
# 跨域问题特定
# Cycle = 50
# print(Cycle)
Cycle_num = int(HyperCycle / Cycle)


# 需要传输信息
# 周期
print('flow_period', flow_period)
# 最大允许传输时间
print('flow_deadline', flow_deadline)
# 数据包长度
print('flow_length', flow_length)
# 节点ip与编号
print('Network_node_id', Network_node_id)
# 流量路径
print('flow_path_id', flow_path_id)
# 超周期
print('HyperCycle', HyperCycle)
# 时隙
print('Cycle', Cycle)
# 时隙总数
print('Cycle_num', Cycle_num)
# 最大传输时间
print('E2e_flow', E2e_flow)

# 需要传输信息补充
flow_name = df_sort['flowid'].tolist()
print('flow_name', flow_name)
flow_starttime = df_sort['starttime'].tolist()
print('flow_starttime', flow_starttime)

# DQN需要传输的信息
print('\nDQN需要传输的信息')
flow_period_normalization = (np.array(flow_period) / Cycle).astype(int).tolist()
print('flow_period_normalization', flow_period_normalization)
flow_pktnum = (HyperCycle / np.array(flow_period)).astype(int).tolist()
print('flow_pkt_number', flow_pktnum)
# Deadline的归一化处理，对于无法被Cycle整除的deadline，我们取较小的整数
# 即(int(Deadline/Cycle) - E2e_time), E2e_time = sum(E2e_flow)
# ALLOW_FWD_MAX = (int(Deadline/Cycle) - sum(E2e_flow))
# ALLOW_FWD_MIN = math.ceil(baseline/Cycle)
# ALLOW_FWD_MAX = ((np.array(flow_deadline) / Cycle) -
#                   np.array([sum(E2e_flow[i]) for i in range(len(flow_period))])).astype(int).tolist()
# print(ALLOW_FWD_MAX)
# 此处的Deadline相比原有Deadline进行了缩小
flow_deadline_normalization = (np.array(flow_deadline) / Cycle).astype(int).tolist()
print('flow_deadline_normalization', flow_deadline_normalization)
# 路径信息
print('flow_path_id', flow_path_id)
# 状态总数
N_states = len(flow_period) + 1
print('N_states', N_states)
Actions = [i for i in range(Cycle_num)]
print('Actions', Actions)
Node_port = len(Network_node_id)
print('Node_port', Node_port)
print("------------------------------data mine-------------------------")
'''
print(flow_path[1246])
print(flow_path_id[1246])
print(E2e_flow[1246])
'''
