"""
@ author:hx
@ date:2023/05/09
自动生产流量
域内 60% 1200
跨域 40% 800
总共2000条
"""
import numpy as np
import pandas as pd
from collections import OrderedDict


def generate_mininet_flow():
    # src and dest
    flow_source = ['10.0.1.1', '10.0.2.2', '10.0.3.3']
    flow_destination = ['10.0.4.4']
    domain1 = flow_source
    domain2 = flow_destination
    # (period, deadline)集合
    period_deadline = np.array([(200, 1000), (400, 1500), (800, 2000),
                       (200, 1000), (400, 1500)])
    length = [100, 100, 500, 1500]

    # 源和目的随机生成
    flow_src_0_N = np.random.choice(domain1, 1000)
    flow_dst_0_N = np.random.choice(domain2, 1000)

    # # 如果域1和域2有节点重合，进行处理
    # if set(domain1).isdisjoint(domain2):
    #     for i in range(len(flow_src_0_N)):
    #         if flow_dst_0_N[i] == flow_src_0_N[i]:
    #             flow_dst_0_N[i] = np.random.choice(np.setdiff1d(np.array(domain1), flow_src_0_N[i]))
    # else:
    #     print("无重合")

    # 数据包长度域时隙周期
    length_0_N = np.random.choice(length, 1000)
    period_dead_0_N = period_deadline[np.random.choice(period_deadline.shape[0], 1000), :]

    data = {
        'flowid': [f'flow{i}' for i in range(1, 1001)],
        'src': flow_src_0_N,
        'dst': flow_dst_0_N,
        'period': np.array(period_dead_0_N)[:, 0],
        'deadline': np.array(period_dead_0_N)[:, 1],
        'length': length_0_N,
        'starttime': [0 for i in range(1000)],
        'prior': [3 for i in range(1000)],
        'slot': [0 for i in range(1000)],
        'path': [[] for i in range(1000)]
    }
    data = OrderedDict(data)
    df = pd.DataFrame(data)
    print(df)
    df.to_excel('flow_mininet_1000.xlsx', index=False)


def generate_time_triggered_flow():
    # flowid src dst period length starttime deadline prior
    # src and dst
    domain1 = ["10.0.0.1", "10.0.0.2", "10.0.0.3", "10.0.0.4", "10.0.0.5", "10.0.0.6"]
    domain2 = ["10.0.1.1", "10.0.1.2", "10.0.1.3", "10.0.1.4", "10.0.1.5", "10.0.1.6"]

    # length
    length = [100, 100, 500, 1500]  # 100的概率更大
    # start time
    start_time = [50, 100, 120, 75]
    # period 与 deadline
    # 域内流量集合
    period_deadline = np.array([(500, 1000), (500, 1000), (500, 1000),
                       (800, 2000), (1000, 2000)])
    # 跨域流量集合
    period_deadline_cross_domain = np.array([(500, 1000),(800, 2000), (1000, 2000),
                                             (1600, 4000), (2000, 4000)])
    # 优先级
    prior = 0  # TT流同等级捏

    # domain1 30% 600
    flow_src_0_600 = np.random.choice(domain1, 600)
    flow_dst_0_600 = np.random.choice(domain1, 600)
    for i in range(len(flow_src_0_600)):
        if flow_dst_0_600[i] == flow_src_0_600[i]:
            flow_dst_0_600[i] = np.random.choice(np.setdiff1d(np.array(domain1), flow_src_0_600[i]))
    length_0_600 = np.random.choice(length, 600)
    # period_dead_0_300 = np.random.choice(period_deadline, 600)
    period_dead_0_600 = period_deadline[np.random.choice(period_deadline.shape[0], 600), :]

    data0 = {
        'flowid': [f'flow{i}' for i in range(1, 601)],
        'src': flow_src_0_600,
        'dst': flow_dst_0_600,
        'period': np.array(period_dead_0_600)[:, 0],
        'deadline': np.array(period_dead_0_600)[:, 1],
        'length': length_0_600,
        'starttime': [0 for i in range(600)],
        'prior': [3 for i in range(600)],
        'path': [[] for i in range(600)]
    }
    df0 = pd.DataFrame(data0)
    print(df0)

    # domain2 30% 600
    flow_src_600_1200 = np.random.choice(domain2, 600)
    flow_dst_600_1200 = np.random.choice(domain2, 600)
    for i in range(len(flow_src_600_1200)):
        if flow_dst_600_1200[i] == flow_src_600_1200[i]:
            flow_dst_600_1200[i] = np.random.choice(np.setdiff1d(np.array(domain2), flow_src_600_1200[i]))
    length_600_1200 = np.random.choice(length, 600)
    # period_dead_0_300 = np.random.choice(period_deadline, 600)
    period_dead_600_1200 = period_deadline[np.random.choice(period_deadline.shape[0], 600), :]

    data1 = {
        'flowid': [f'flow{i}' for i in range(601, 1201)],
        'src': flow_src_600_1200,
        'dst': flow_dst_600_1200,
        'period': np.array(period_dead_600_1200)[:, 0],
        'deadline': np.array(period_dead_600_1200)[:, 1],
        'length': length_600_1200,
        'starttime': [0 for i in range(600)],
        'prior': [3 for i in range(600)],
        'path': [[] for i in range(600)]
    }
    df1 = pd.DataFrame(data1)
    # print(df1)

    # 跨域 40% 800, 源与目的不会碰撞
    flow_src_1200_1600 = np.random.choice(domain1, 400)
    flow_dst_1200_1600 = np.random.choice(domain2, 400)
    flow_src_1600_2000 = np.random.choice(domain2, 400)
    flow_dst_1600_2000 = np.random.choice(domain1, 400)

    flow_src_1200_2000 = np.append(flow_src_1200_1600, flow_src_1600_2000)
    flow_dst_1200_2000 = np.append(flow_dst_1200_1600, flow_dst_1600_2000)

    # 长度选取
    length_1200_2000 = np.random.choice(length, 800)
    # period_dead_0_300 = np.random.choice(period_deadline_cross_domain, 600)
    period_dead_1200_2000 = period_deadline_cross_domain[np.random.choice(period_deadline_cross_domain.shape[0], 800), :]

    data2 = {
        'flowid': [f'flow{i}' for i in range(1201, 2001)],
        'src': flow_src_1200_2000,
        'dst': flow_dst_1200_2000,
        'period': np.array(period_dead_1200_2000)[:, 0],
        'deadline': np.array(period_dead_1200_2000)[:, 1],
        'length': length_1200_2000,
        'starttime': [0 for i in range(800)],
        'prior': [3 for i in range(800)],
        'path': [[] for i in range(800)]
    }
    df2 = pd.DataFrame(data2)
    # print(df2)

    # 整体拼接
    # df2 = pd.merge(df0, df1, how='outer',
    # on=['flowid','src','dst','period','deadline','length','starttime','prior','path'])
    df = pd.concat([df0, df1, df2], ignore_index=True)  # ignore index 为True会重新排列索引
    print(df)

    # 存储
    # df.to_excel('flow2000_mixed.xlsx', index=False)


if __name__ == '__main__':
    # generate_time_triggered_flow()
    generate_mininet_flow()
