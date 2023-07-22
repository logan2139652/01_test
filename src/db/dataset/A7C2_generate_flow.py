"""
@ author:hx
@ date:2023/05/09
自动生产流量 专为A4C1网络准备
域内 60% 600
跨域 40% 400
总共1000条
"""
import copy

import numpy as np
import pandas as pd
from collections import OrderedDict


def generate_mininet_flow(flow_num):
    """
    流量生成函数
    :param flow_num:生成数目
    :return:
    """
    # src and dest
    flow_source = ["h1", "h2", "h3", "h4", "h5", "h6", "h7", "h8", "h9",
                   "h10", "h11", "h12", "h13", "h14"]
    flow_destination = ["h1", "h2", "h3", "h4", "h5", "h6", "h7", "h8", "h9",
                        "h10", "h11", "h12", "h13", "h14"]
    domain1 = flow_source
    domain2 = flow_destination
    # (period, deadline)集合
    # 均衡
    period_deadline = np.array([(200, 800), (400, 1000), (800, 2000),
                                (400, 1000), (800, 2000)])
    # 小
    period_deadline = np.array([(200, 800), (400, 1000), (200, 800)])
    # 大
    period_deadline = np.array([(800, 2000), (400, 1000), (400, 1000)])

    length = [1000, 100, 500, 1500]


    # 源和目的随机生成
    flow_src_0_N = np.random.choice(domain1, flow_num)
    flow_dst_0_N = np.random.choice(domain2, flow_num)

    for i in range(flow_num):
        if flow_dst_0_N[i] == flow_src_0_N[i]:
            newChoice = copy.copy(flow_destination)
            newChoice.remove(flow_dst_0_N[i])
            flow_dst_0_N[i] = np.random.choice(newChoice)
    # 数据包长度域时隙周期
    length_0_N = np.random.choice(length, flow_num)
    period_dead_0_N = period_deadline[np.random.choice(period_deadline.shape[0], flow_num), :]

    data = {
        'flowid': [f'flow{i}' for i in range(1, flow_num+1)],
        'src': flow_src_0_N,
        'dst': flow_dst_0_N,
        'period': np.array(period_dead_0_N)[:, 0],
        'deadline': np.array(period_dead_0_N)[:, 1],
        'length': length_0_N,
        'starttime': [0 for i in range(flow_num)],
        'prior': [3 for i in range(flow_num)],
        'slot': [0 for i in range(flow_num)],
        'path': [[] for i in range(flow_num)]
    }
    data = OrderedDict(data)
    df = pd.DataFrame(data)
    print(df)
    name = f'A7C2_{flow_num}_big' + '.xlsx'
    print(name)
    df.to_excel(name, index=False)


if __name__ == '__main__':
    flow_num = 1000
    generate_mininet_flow(flow_num)
