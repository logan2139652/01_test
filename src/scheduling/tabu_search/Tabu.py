"""
@author: huangxu
@function: 禁忌搜索主函数
"""
import copy
import random
import numpy as np
import math
import src.data_processing.A4C1_data_processing as A4C1Net


def update_env(flow, slot, env):
    current_byte = []
    # 不同端口的时隙列表，即不同端口容纳该流的时隙
    node_slot = [[] for i in range(env.node_port_num)]  # 根据port的数目设置不同port的slot
    # 路径遍历
    for k in range(len(env.ALL_PATH[flow])):
        for j in range(env.PKT_NUM[flow]):
            # 数据包个数：j为0，1，2，一共3个数据包
            # 根据该流的路径，选择流需要进入的port与对应编号i
            thorough_e2e_slot = 0
            # 选择flow的第route个路径，并对其port与index遍历
            for i, port in enumerate(env.ALL_PATH[flow][k]):
                # 当前数据包在当前节点的入队时隙为初始偏置+路径传输时隙+第j个数据包的发送时隙
                slot_temp = int((slot + thorough_e2e_slot + j * env.PERIOD[flow]) % env.slot_num)
                # 这里很巧妙得变换了hop slot 与through slot
                thorough_e2e_slot += env.flow_hop_slot[flow][k][i]
                # print('slot temp', slot_temp)
                # 在port的slot_temp时隙进行入队
                current_byte.append(env.env_list[port][slot_temp])
                # node_slot记录该流的所有数据包会进入的各端口时隙，以便后续直接相减更新环境
                node_slot[port].append(slot_temp)
        # print('node_slot', node_slot)
        # 判断是否可以转发
        if min(current_byte) >= env.FLOW_LEN[flow]:
            # 可以转发，更新状态
            for i, port in enumerate(env.ALL_PATH[flow][k]):
                # 所有的都减去flow.length
                env.env_list[port][node_slot[port]] -= env.FLOW_LEN[flow]
            return True
        else:
            # 无法转发
            return False


def neighborhood_func(current_solution, slot_allow_forwarding):
    """
    根据当前解答，求解一组邻居解，我们采用贪婪的方式
    注意，neighborhood_func求解的是一组邻居解，而不是单个邻居解
    :param current_solution: 当前解，所有流量动作，失败的动作为-1，成功为时隙
    :param slot_allow_forwarding: 每流可选时隙空间
    :return:
    """

    neighborhood = []
    # 邻居解有2个
    for j in range(10):
        temp = []
        for i in range(len(current_solution)):
            neighbor = current_solution[i]
            # 在当前解的第i个变量上进行改变
            if neighbor < 0:
                 neighbor = np.random.choice(slot_allow_forwarding[i])
            # neighbor = np.random.choice(slot_allow_forwarding[i])
            temp.append(neighbor)
        neighborhood.append(temp)
    return neighborhood


def objective_func(solution, env):
    """
    计算当前解的优劣
    :param solution:
    :return:
    """
    # 计算能够装进箱子的物品总价值
    # total_schedule = sum(i >= 0 for i in solution)
    # 或者 total_schedule = len([i for i in solution if i >= 0])
    env.reset()  # 环境更新，每次判断动作时需要进行更新
    total_schedule = 0
    new_solution = copy.copy(solution)
    for flow, action in enumerate(solution):
        if action < 0:
            total_schedule = total_schedule - 1
            new_solution[flow] = action
        elif update_env(flow, action, env):
            total_schedule = total_schedule + 1
            new_solution[flow] = action
        else:
            total_schedule = total_schedule - 1
            new_solution[flow] = -1

    return total_schedule, new_solution


def tabu_search(initial_solution, neighborhood_func, objective_func, tabu_list_size, max_iterations):
    """
    tabu搜索主函数
    :param initial_solution: 初始解，所有流量的动作合集
    :param neighborhood_func: 生成初始解邻居解的函数，比如变换初始解中部分动作的解
    :param objective_func: 目标函数，判断初始解与邻居解好坏，比如成功调度流量数目
    :param tabu_list_size: 禁忌表长度，每次会记录最优解，如超出长度，则移除第一个解
    :param max_iterations: 最大迭代次数，超出后返回最终值
    :return:
    """
    current_solution = initial_solution
    best_solution = current_solution
    slot_allow_forwarding = []
    tabu_list = []

    for _ in range(max_iterations):
        neighborhood = neighborhood_func(current_solution, slot_allow_forwarding)
        best_neighbor = None
        best_neighbor_value = float('inf')

        for neighbor in neighborhood:
            if neighbor not in tabu_list:
                neighbor_value = objective_func(neighbor)
                if neighbor_value < best_neighbor_value:
                    best_neighbor = neighbor
                    best_neighbor_value = neighbor_value

        if best_neighbor is None:
            break

        current_solution = best_neighbor
        tabu_list.append(best_neighbor)
        if len(tabu_list) > tabu_list_size:
            tabu_list = tabu_list[1:]

        if objective_func(best_neighbor) < objective_func(best_solution):
            best_solution = best_neighbor

    return best_solution

