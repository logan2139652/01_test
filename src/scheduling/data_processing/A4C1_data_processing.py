"""
@ author:hx
@ data:2023/05/09
A4C1数据处理 与 src.scheduling.data_processing.mininet_data.py和算法模块同用
# route处理，去除第一跳与最后一跳，即所以与host相连接的路径必须清除

"""
import pandas as pd
import networkx as nx
import math
import numpy as np
from src.db.topology.A4C1topo import A4C1topo, topoRoute
import copy

# 最大公约数（Greatest Common Divisor，GCD）函数
def getGCD(num):
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
# 且必须时cycle的四倍
def getLCM(num, slot):
    """
    求任意多个数的最小公倍数
    :param num: 一个int的列表
    :param slot: 时隙大小
    :return: 这个int列表中的最小公倍数
    """
    # CQF中，LCM需要为2的倍数，方便规划调度
    # CQCF中，LCM需要为4的倍数
    minimum = 4 * slot
    for i in num:
        minimum = int(i) * int(minimum) / math.gcd(int(i), int(minimum))
    return int(minimum)


def flowPath2id(flow_path, edge_port_dict):
    """
    将路径转换为id
    :param flow_path:
    :param edge_port_dict:
    :return:
    """
    flow_path_id = []
    for i in range(len(flow_path)):
        result = []
        for path in flow_path[i]:
            temp = []
            for edge in path:
                temp.append(edge_port_dict[edge])
            result.append(temp)
        flow_path_id.append(result)
    return flow_path_id


def longEdgeGet(CORE_NODE, edge_list):
    """
    得到长距离链路
    :param CORE_NODE: 核心节点
    :param edge_list: 所有edge集合
    :return: 长距离连接的edge
    """
    result = []
    for i in CORE_NODE:
        for j in CORE_NODE:
            if tuple((i, j)) in edge_list:
                result.append(tuple((i, j)))
    return result


def hopSlotGet(flow_path, long_edge, cross_edge):
    """
    计算端到端时隙
    :param flow_path: edge路径
    :param long_edge: 长距离edge
    :return: edge路径对应的hop_slot
    """
    flow_hop_slot = []
    for i in range(len(flow_path)):
        result = []
        for path in flow_path[i]:
            e2e = []
            for edge in path:
                if edge in long_edge:
                    e2e.append(2)
                elif edge in cross_edge:
                    e2e.append(2)
                else:
                    e2e.append(1)
            result.append(e2e)
        flow_hop_slot.append(result)
    return flow_hop_slot


def crossEdgeGet(CORE_NODE, edge_list, NOT_CORE_NODE):
    """
    得到跨域链路
    :param CORE_NODE: 核心节点
    :param edge_list: 所有edge集合
    :param NOT_CORE_NODE: 所有节点
    :return: 长距离连接的edge
    """
    result = []
    for i in CORE_NODE:
        for j in NOT_CORE_NODE:
            if tuple((i, j)) in edge_list:
                result.append(tuple((i, j)))
            if tuple((j, i)) in edge_list:
                result.append(tuple((j, i)))
    return result


class A4C1Net:
    def __init__(self, flow_number, excel_path=None, test_slot_size=None):
        """
        初始化网络
        :param path:流量文件存放路径
        :param flow_number:想要调度的流数目
        :param test_slot_size:测试时隙，默认为None，即不设置，自动生成
        :param CORE_NODE:默认跨域节点，设置为[3，4，5], 测试时设置为[]
        """

        CORE_NODE = [0, 1, 2, 3]
        NOT_CORE_NODE = [i for i in range(0, 16) if i not in CORE_NODE]

        if excel_path is None:
            excel_path = r'D:\Python\PyCharmWorkplace\01ArticleOneCode\src\db\dataset\A4C1_1000.xlsx'

        # .xlsx文件读取
        DATA_PATH = excel_path
        df = pd.read_excel(DATA_PATH)[:flow_number]
        # print(df)

        G = A4C1topo()

        df = topoRoute(G, df)
        # 网络流量信息提取
        # df_sort = df.sort_values(by=['prior'], ascending=[False])  # 优先级提取并排序
        df_sort = df
        flow_period = df_sort['period'].tolist()  # 周期提取
        # print(flow_period)
        flow_deadline = df_sort['deadline'].tolist()  # 最大允许传输时间提取
        # print(flow_deadline)
        flow_length = df_sort['length'].tolist()  # 数据包长度提取
        # print(flow_length)
        flow_path = df_sort["path"]  # 使用NetworkX的all_simple_route得到的所有源与目的节点之间的路径

        # —————————————————————————————————————————————————————————————— #
        # 路径route-->path id
        # 将路由转换成节点id
        # —————————————————————————————————————————————————————————————— #
        G1 = G.to_directed()  # 将无向图G设置为有向图G
        edge_to_remove = [(u, v) for u, v in G1.edges() if isinstance(u, str) | isinstance(v, str)]  # 与host有关的edge，删除
        G1.remove_edges_from(edge_to_remove)
        edge_list = nx.edges(G1)
        edge_port_dict = dict(zip(edge_list, [i for i in range(len(edge_list))]))
        # print(edge_port_dict)
        flow_path_id = flowPath2id(flow_path, edge_port_dict)  # 将edge_path转换为path_id
        # print(flow_path[1])
        # print(flow_path_id[1])

        # 计算链路端到端时间，即每条路径的端到端传输时隙
        long_edge = longEdgeGet(CORE_NODE, edge_list)  # 选择长距离路径
        cross_edge = crossEdgeGet(CORE_NODE, edge_list, NOT_CORE_NODE)
        # print(long_edge)
        # print(cross_edge)

        # —————————————————————————端到端时隙变换—————————————————————————————————#
        # 短距离一个slot，长距离两个或者多个slot，仿真时使用两个slot
        # 跨域链路需要考虑链路长度，CQF链路长度约为0，因此端到端时间为cycle
        # CQCF不能忽略链路长度，因此端到端时间为2cycle + link_delay
        # 这里的2cycle其实是包括部分链路时间的，我们假设CQCF端到端传输为2Cycle
        # 因为唯一需要考虑的是，跨域传输是的链路时间
        # CQF-》CQCF，不能直接加cycle，还需要加上link_delay，我们设置链路时间为2ms，因此加上一个cycle处理
        # CQCF-》CQCF，采用2cycle的传输时间，模拟链路长度
        # CQCF-》CQF，由于CQCF默认采用2Cycle的传输时间，因此包括了Cycle时间
        flow_hop_slot = hopSlotGet(flow_path, long_edge, cross_edge)
        # print(flow_hop_slot[1])

        # 超周期与时隙计算
        Max_Cycle = getGCD(flow_period)  # 最大时隙 所有流量的最大公约数
        Min_Cycle = 10  # 最小时隙 最长链路之间的传输时间 10百微妙, 即1ms, 如果是跨域链路，可以设置为50
        cycle_list = []  # 时隙选择
        for C in range(Min_Cycle, Max_Cycle + 1):
            if not Max_Cycle % C:
                cycle_list.append(C)

        if test_slot_size:
            Cycle = test_slot_size
        else:
            Cycle = cycle_list[5]

        # Cycle = 50   # 测试专用时隙
        HyperCycle = getLCM(flow_period, Cycle)  # 超周期 所有流量的最小公倍数
        print('————————————————————————————————Data Processing——————————————————————————————————')
        print('cycle list which we can choose:', cycle_list)
        print('HyperCycle:', HyperCycle)
        print("-----------------------------------")
        Cycle_num = int(HyperCycle / Cycle)
        # print(Cycle_num)

        # 需要传输信息
        self.flow_period = flow_period  # 周期
        self.flow_deadline = flow_deadline  # 最大允许传输时间
        self.flow_length = flow_length  # 数据包长度
        self.flow_path_id = flow_path_id  # 路由集合 edge_id
        self.HyperCycle = HyperCycle  # 超周期
        self.Cycle_list = cycle_list  # 时隙选择范围
        self.Cycle = Cycle  # 时隙
        self.Cycle_num = Cycle_num  # 时隙总数
        self.flow_hop_slot = flow_hop_slot  # 每跳传输时间
        self.flow_name = df_sort['flowid'].tolist()  # 流量id
        self.flow_starttime = df_sort['starttime'].tolist()  # 流量起始时间
        self.net_edge_port_dict = edge_port_dict  # 端口总数
        self.flow_path = flow_path  # 所有流路径
        self.flow_data = df_sort  # 流集合

        # dqn与q_learning额外需要信息
        self.flow_period_normalization = (np.array(flow_period) / Cycle).astype(int).tolist()
        self.flow_pkt_num = (HyperCycle / np.array(flow_period)).astype(int).tolist()
        # Deadline的归一化处理，对于无法被Cycle整除的deadline，我们取较小的整数
        # 此处的Deadline相比原有Deadline进行了缩小
        self.flow_deadline_normalization = (np.array(flow_deadline) / Cycle).astype(int).tolist()
        # 状态总数
        self.n_states = len(flow_period) + 1
        # 路由最大个数
        self.n_route = [len(flow_path[i]) for i in range(len(flow_path))]
        # 动作集合
        self.actions = [i for i in range(Cycle_num * max(self.n_route))]


if __name__ == '__main__':
    mininet = A4C1Net(excel_path=r'D:\Python\PyCharmWorkplace\01ArticleOneCode\src\db\dataset\A4C1_1000.xlsx',
                      flow_number=1000, test_slot_size=50)
    print(mininet.flow_path_id[0])
    print(mininet.flow_path[0])
    print(mininet.n_states)
    print(mininet.n_route)
    print(mininet.actions)

    print(mininet.flow_hop_slot[0])
