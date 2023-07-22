"""
q——learning 环境函数

@2023/5/16
bug1 我们将路由选路添加进去，但是发现未收敛，起初认为是参数因素，但在检查环境后，发现环境出了问题：
我们将动作空间扩大，但是环境依旧按照action去选择，所以导致部分环境与设置为port*action而不是port*slot
同时Q-table应该还是state*action，进行了扩大维度
"""

import numpy as np
from src.scheduling.data_processing.mininet_data import MiniNet
import math
import copy
import pandas as pd


class Maze(object):
    # 初始化参数
    def __init__(self, net, queue_size=None):
        super(Maze, self).__init__()
        # 从net中获取
        self.slot_num = net.Cycle_num  # 时隙数目
        self.slot_size = net.Cycle   # 时隙大小
        self.FLOW_LEN = net.flow_length  # 数据长度
        self.flow_num = len(net.flow_length)  # 流量总数
        self.PERIOD = net.flow_period_normalization  # 每个流的周期
        self.PKT_NUM = net.flow_pkt_num  # 每个流在一个HC中的数据包数目
        self.DEADLINE = net.flow_deadline_normalization  # 每流的数据包最大允许传输时间（以时隙为最小单元）
        self.ALL_PATH = net.flow_path_id  # 每流路径，上述为发送端口，包含port0->por1->port2->port3->port4
        # --------------------------------------------------
        # 注意，net.flow_hop_slot与net.flow_path_id是三维数组
        # self.ALL_PATH[flow] 需要写为 self.ALL_PATH[flow][0]
        # --------------------------------------------------
        #  状态设计
        # self.n_states = net.n_states  # 状态总数，流量数 + 终止态
        # self.states = [0 for i in range(self.flow_num)]
        # self.states = []  # 初始状态，没有调度任何一条流
        self.states = 0  # 初始状态，没有调度任何一跳流
        # self.n_states = 2 * self.flow_num + 2  # 每条流分为调度成功与失败两种状态，然后再加上初始与全部调度完成两种状态
        self.n_states = 2 * self.flow_num  # 每条流分为调度成功与失败两种状态，初始为0，最后状态为最后两个流的状态
        self.node_port = net.net_edge_port_dict  # 网络节点dict 用于建立网络环境
        self.node_port_num = len(self.node_port)  # 计算网络节点总数
        self.action_space = net.actions  # 动作空间, 遍历所有时隙与路径
        self.n_actions = len(self.action_space)  # 动作空间长度,也是时隙总数与路径乘积

        # 如果队列长度未设置，则设置为1500
        if queue_size is None:
            self.queue_size = 1500
        else:
            self.queue_size = queue_size

        # 在每个port需要等待一个或者多个slot时间，这个等待的slot时间根据不同机制来给定，CQF为1个，DIP为1到2个，CQCF为2个（暂定）
        # net.flow_hop_slot[i][0] 是一个[flow:[path1:[],path2:[]]], [[[], []], [[]]]
        self.flow_hop_slot = net.flow_hop_slot  # 每个流跳所需的时隙
        # print(self.flow_hop_slot)
        self.flow_starttime = net.flow_starttime  # 每个流起始时间

        # 计算流最大最小可以转发时隙，可等于max或者min
        self.max_allow_forwarding = [min((self.PERIOD[i] - 1), (self.DEADLINE[i] - sum(self.flow_hop_slot[i][0]) - 2))
                                     for i in range(self.flow_num)]
        # 可选转发时隙为周期加上最大传输时延同时决定
        self.min_allow_forwarding = [math.ceil(int(self.flow_starttime[i] / self.slot_size)) for i in
                                     range(self.flow_num)]
        # 流可以转发的路径选择
        self.max_allow_path = net.n_route  # 路径数目：如果self.max_allow_path[1]=4,说明有0,1,2,3四种路径可以选择

        # 每次episode循环的初始化
        self.env_list = np.full((len(self.node_port), self.slot_num), self.queue_size, dtype=float)  # 网络环境，port*slot
        self.has_schedule = []  # 成功调度数目
        self.not_schedule = []
        self.S = 0  # 初始状态
        self.R_sum = 0  # 初始奖励总和
        self.is_terminated = False  # 循环终止符号

    # 初始化函数
    def reset(self):
        # 新建立一个完整的网络空间，其中没有流量占用，大小为port数*总的时隙数目SLOT_NUM
        self.env_list = np.full((self.node_port_num, self.slot_num), self.queue_size, dtype=float)
        # 已经调度的流量总数，一个episode之初为0
        self.has_schedule = []
        self.not_schedule = []
        # 初始状态，为0，一个流都未调度，但是准备调度第一个流
        # self.S = []
        self.S = 0
        # 一次episode的总奖励值
        self.R_sum = 0
        # 是否结束，这个其实可以转换为t=0, until t=M
        self.is_terminated = False
        # 返回初始状态S
        return self.S

    # 更新网络环境
    def update_env(self, flow, slot, route):
        # 收集网络状态与更新
        # 当前网络需要容纳该flow的各节点的byte数目
        current_byte = []
        # 不同端口的时隙列表，即不同端口容纳该流的时隙
        node_slot = [[] for i in range(self.node_port_num)]  # 根据port的数目设置不同port的slot
        # j为该flow的数据包遍历
        for j in range(self.PKT_NUM[flow]):  # 数据包个数：j为0，1，2，一共3个数据包
            # 根据该流的路径，选择流需要进入的port与对应编号i
            thorough_e2e_slot = 0
            # 选择flow的第route个路径，并对其port与index遍历
            for i, port in enumerate(self.ALL_PATH[flow][route]):  # 根据action的route选路
                #
                # CQF为slot+0, slot+1, slot+2, slot+3, slot+4
                # CQCF为slot+0, slot+2, slot+4
                # DIP/CSQF为...
                # current_byte加入环境中需要占据的端口与对应时隙时隙，时隙在溢出时选择取所有时隙的模
                # slot_temp = int((slot + i * 1 + j * PERIOD[flow]) % self.slot_num) 只有CQF的方案
                # 加入CQCF的方案
                slot_temp = int((slot + thorough_e2e_slot + j * self.PERIOD[flow]) % self.slot_num)
                # 这里很巧妙得变换了hop slot 与through slot
                thorough_e2e_slot += self.flow_hop_slot[flow][route][i]
                # print('slot temp', slot_temp)
                # 在port的slot_temp时隙进行入队
                current_byte.append(self.env_list[port][slot_temp])
                # node_slot记录该流的所有数据包会进入的各端口时隙，以便后续直接相减更新环境
                node_slot[port].append(slot_temp)

        # print('node_slot', node_slot)
        # 判断是否可以转发
        if min(current_byte) >= self.FLOW_LEN[flow]:
            # 可以转发，更新状态
            for i, port in enumerate(self.ALL_PATH[flow][route]):
                # 所有的都减去flow.length
                self.env_list[port][node_slot[port]] -= self.FLOW_LEN[flow]
            return True
        else:
            # 无法转发
            return False

    def step(self, action):
        """
        # 通过动作得出奖励
        :param action: 动作，需要进行分解
        :return: S_NEW, R, done，新动作，奖励，与是否完成
        """
        # 状态变换，当前状态为old状态
        S_old = self.S
        # S_NEW = copy.copy(S_old)
        # flow = len(S_old)  # 第几条流，通过状态空间设定这是第几条流
        flow = int((S_old + 1) / 2)
        # 动作变换， 0a,0b,1a,1b,2a,2b,3a,3b  0 1 2 3 4 5 6 7    7/4= 1  7%4=3  其中1对应路径b，3对应时隙3
        slot = action % self.slot_num  # 动作除以时隙，取余得到时隙
        route = int(action / self.slot_num)  # 动作除以时隙，取商得到路径
        # 判断
        flow_result = -1  # 当前流调度情况，默认失败
        if (slot > self.max_allow_forwarding[flow]) | (slot < self.min_allow_forwarding[flow]):
            # 大于可选时隙，调度失败
            # R = -2 - 0.001 * len(self.not_schedule)
            R = -1
            self.not_schedule.append(flow)
        elif route >= self.max_allow_path[flow]:
            R = -2 - 0.001 * len(self.not_schedule)
            R = -1
            self.not_schedule.append(flow)
        else:
            # 在可选时隙内，通过环境约束，判断是否可以转发，可以转发R为1，否则为-1
            if self.update_env(flow, slot, route):
                # 调度成功，奖励+1
                # R = 2 + 0.001 * len(self.has_schedule) - 0.001 * len(self.not_schedule)
                R = 1
                self.has_schedule.append(flow)
                # flow_result = 1
                flow_result = 0
            else:
                # 调度失败，奖励-1
                # R = -1 - 0.001 * len(self.not_schedule) + 0.001 * len(self.has_schedule)
                R = -1
                self.not_schedule.append(flow)
        # 环境变换，由于是时间步数的，也就是流量状态变换的，因此统一设立状态变换即可，不会应为选择不同而到达不同状态
        # 更新新的环境
        # S_NEW.append(flow_result)
        S_NEW = flow * 2 + 2 + flow_result
        # self.S = copy.copy(S_NEW)
        self.S = S_NEW
        if S_NEW >= self.n_states - 1:  # 此处为n_state还是n_state-1需要进行商议
        # if len(S_NEW) == self.flow_num:  # 所以的流都调度完成
            done = True
        else:
            done = False
        return S_NEW, R, done

    def render(self):
        pass


def update():
    for t in range(1):
        s = env.reset()
        while True:
            env.render()
            if s == 0:
                a = 0
            else:
                a = 1
            s_, r, done = env.step(a)
            print(s_, r, done, s)
            s = s_
            if done:
                break
        print(len(env.has_schedule))


if __name__ == '__main__':

    net = MiniNet(
        excel_path=r'D:\Python\PyCharmWorkplace\ArticleOneCode\src\db\dataset\flow_mininet_1000.xlsx',
        flow_number=10,
        test_slot_size=20,
        CORE_NODE=None)
    env = Maze(net=net, queue_size=100)
    update()
    # print(env.env_list)
    df1 = pd.DataFrame(env.env_list)
    idx = list(net.net_edge_port_dict)
    df1.set_index(pd.Index(idx), inplace=True)
    with pd.ExcelWriter("test.xlsx") as writer:
        df1.to_excel(writer, sheet_name='Schedule')
    print(net.flow_path[0][0])
    print(net.flow_path[1][1])
    print(net.flow_path[1][0])