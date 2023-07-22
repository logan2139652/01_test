"""
贪婪算法运行函数
首先建立一个网络，调用参数中CORE_NODE默认是None对应[3,4,5]，之后可以根据拓扑设立CORE
然后调用贪婪算法主函数
"""
import copy

from src.data_processing.A4C1_data_processing import A4C1Net
from src.data_processing.A7C2_data_processing import A7C2Net
from src.scheduling.tabu_search.Tabu import neighborhood_func, objective_func
from src.scheduling.tabu_search.traffic_env import Maze
from src.scheduling.tabu_search.greedy import greedy_algorithm
import time
import pandas as pd
import numpy as np
import math

if __name__ == '__main__':
    choose_slot = 10
    bandwidth_rate = 300  # unit : Bytes/0.1ms   eg:500 Bytes/0.1ms = 5M B/s = 40M bps
    queue_capacity = choose_slot * bandwidth_rate

    net = A4C1Net(
        # excel_path=r'D:\Python\PyCharmWorkplace\01ArticleOneCode\src\db\dataset\A4C1_1000_big.xlsx',
        excel_path=r'D:\Python\PyCharmWorkplace\01ArticleOneCode\src\db\dataset\A4C1_1000.xlsx',
        flow_number=1000,
        test_slot_size=choose_slot)

    # ————————————————————————求解动作空间—————————————————————— #
    max_allow_forwarding = [min((net.flow_period_normalization[i] - 1),
                                (net.flow_deadline_normalization[i] - sum(net.flow_hop_slot[i][0]) - 2))
                            for i in range(len(net.flow_period))]
    min_allow_forwarding = [math.ceil(int(net.flow_starttime[i] / net.Cycle))
                            for i in range(len(net.flow_period))]

    slot_allow_forwarding = []
    for i, j in zip(min_allow_forwarding, max_allow_forwarding):
        temp = [k for k in range(i, j + 1)]
        slot_allow_forwarding.append(temp)

    new_slot_allow_forwarding = copy.deepcopy(slot_allow_forwarding)
    for slot_list in new_slot_allow_forwarding:
        slot_list.append(-1)
    # ———————————————————————————函数开始——————————————————————— #
    # 基于贪婪
    actions = greedy_algorithm(net, QUEUE_SIZE=queue_capacity)
    print(len(actions))
    print(actions)
    print(sum(1 for num in actions if num > 0))

    # 初始化
    # current_solution = [-1 for i in range(len(net.flow_period))]
    current_solution = actions
    best_solution = current_solution
    slot_allow_forwarding = slot_allow_forwarding
    tabu_list_size = 50
    max_iterations = 100
    tabu_list = []
    # 建立环境
    env = Maze(net=net, queue_size=queue_capacity)
    best_neighbor = None
    # 最小化设置为inf，最大化设置为-inf
    best_neighbor_value = float('-inf')
    # ---------------------------------
    # print('-------------------------------------')
    # actions_value, ne = objective_func(actions, env)
    # print(actions_value)
    # print('-------------------------------------')
    # --------------------------------
    a = time.time()  # 记录时间
    sched_plt = []
    for _ in range(max_iterations):
        print(f"<<<<<<<<<<<<<第{_}次epochs<<<<<<<<<<<<<<<")
        neighborhood = neighborhood_func(current_solution, slot_allow_forwarding)
        # print(neighborhood[0])

        for neighbor in neighborhood:
            if neighbor not in tabu_list:
                neighbor_value, neighbor = objective_func(neighbor, env)
                if neighbor_value > best_neighbor_value:
                    best_neighbor = neighbor
                    best_neighbor_value = neighbor_value

        current_solution = best_neighbor
        tabu_list.append(best_neighbor)
        if len(tabu_list) > tabu_list_size:
            tabu_list = tabu_list[1:]

        # objective_func返回值是一个二元的，第一元为结果
        if objective_func(best_neighbor, env)[0] > objective_func(best_solution, env)[0]:
            best_solution = best_neighbor

        # print(best_solution)
        # print(len([i for i in current_solution if i >= 0]))
        # print(len([i for i in best_neighbor if i >= 0]))
        reward = len([i for i in best_solution if i >= 0])
        sched_plt.append(reward)
        print(reward)

    b = time.time()
    # print(len(tabu_list))
    print("算法时间:", b - a)
    print(sched_plt)
