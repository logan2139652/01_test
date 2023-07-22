"""
@ 基于贪婪的算法设计
@ author : hx
@ create : 2023/5/6
@ info: 基于贪婪的算法tool函数, 与src.scheduling.Data_set.mininet_data.py配合和本文件夹下greedy.py配合
"""
import math
import numpy as np
import pandas as pd


# ------------------------------------------------ #
# 偏置约束， deadline、slot_size、hop_slot计算根据端到端 #
# 时延的最大允许时隙偏置其一，周期计算最大可允许偏置其二，取小者#
# ------------------------------------------------ #
def offset_constraint(deadline, slot_size, hop_slot, period):
    """
    offset约束：包括端到端时延约束与周期约束
    :param deadline: 最大允许时间
    :param slot_size: 时隙大小
    :param hop_slot: 路径每跳时隙
    :param period: 周期
    :return: 满足端到端时延约束的最大offset
    """
    offset1 = deadline_constraint(deadline, slot_size, hop_slot)  # 调用函数计算offset1
    offset2 = int(period / slot_size) - 1  # 时隙范围为[0,1,2,N-1],period/slot_size = N, 不可取N
    return min(offset1, offset2)


def deadline_constraint(deadline, slot_size, hop_slot):
    """
    端到端时延约束
    :param deadline: 最大允许时间
    :param slot_size: 时隙大小
    :param hop_slot: 路径时隙
    :return: 满足端到端时延约束的最大offset
    """
    e2e = e2e_delay(hop_slot, slot_size, 0)  # 调用e2e_delay函数计算端到端时延
    # 如果端到端时延大于允许时延，则调度失败，否则计算可以调度的最大时隙偏置
    if e2e > deadline:
        # print(e2e, deadline, "deadline false")
        return -1
    else:
        offset = (deadline - e2e) / slot_size
        return int(offset)


def e2e_delay(hop_slot, slot_size, current_slot, record=None):
    """
    重写端到端时延函数
    :param record: 关键字参数，默认为None，传输为list的参数
    :param hop_slot: 路径时隙列表 [0, 1, 2, 1, 2]
    :param slot_size: 时隙
    :param current_slot: 当前时隙
    :return: 端到端时延
    """
    sum_slot = 2  # 从源host到交换机与交换机到目的host的两跳
    # sum_slot = 1  # 只包括交换机到目的host的那一跳
    # 计算每跳的时隙之和
    for i in hop_slot:
        sum_slot = sum_slot + i
    # 计算端到端时延
    e2e = (sum_slot + current_slot) * slot_size
    if record:
        record.append(e2e)
    return e2e


# ------------------------------------------------ #
# 队列约束，判断数据包是否能入队                         #
# ------------------------------------------------ #
def queue_constraint(slist, pathid, slot, length, through_slot, pktnum, pkthop, slotnum):
    """
    判断当前流的所有数据包是否满足途径交换机的队列约束
    :param slist: 队列资源表
    :param pathid:  横轴表，交换机id列表
    :param slot:    纵坐标，时隙id
    :param length:  队列资源长度
    :param through_slot: 路径时隙
    :param pktnum: 所有数据包数量
    :param pkthop: 任意两个数据包之间的时隙间隔
    :param slotnum: 时隙数目
    :return: slist or 0
    """
    temp = [[0 for j in range(len(i))] for i in slist]  # 建立一个空的资源空间
    # print(temp)
    basicslot = slot
    for j in range(pktnum):
        slot = int(basicslot + j * pkthop) % slotnum
        k = 0
        for i in through_slot:
            switchid = pathid[k]
            slot = int(slot + i) % slotnum
            temp[switchid][slot] = length
            k += 1
    if (np.array(slist) - np.array(temp) >= 0).all():
        slist = (np.array(slist) - np.array(temp))
        return slist.tolist()
    else:
        return False


def enter_schedule_list(through_slot, offset, slotnum, flownum, pktnum, pkthop, flow_name, schedule_list, switchid):
    """
    输入flow所有数据包至调度表
    :param through_slot:路径时隙
    :param current_offset:当前offset
    :param slotnum: 时隙数
    :param flownum: 当前流量num
    :param pktnum:流量数据包数量
    :param pkthop:流量数据包跳数
    :param flow_name:存有流量名称的列表
    :param schedule_list:调度表
    :param switchid:交换机id列表
    :return:
    """
    basic_offset = offset
    for j in range(pktnum):
        pkt = flow_name[flownum] + '.' + str(j)
        offset = int((basic_offset + j * pkthop) % slotnum)
        temp = offset
        for k in range(len(through_slot)):
            temp = int((temp + through_slot[k]) % slotnum)
            schedule_list[switchid[k]][temp].append(pkt)
    return schedule_list


# ------------------------------------------------ #
# 将流量调度结果，队列剩余资源，流量路径与时隙信息存储为excel #
# ------------------------------------------------ #
def list_to_excel(data, data2, data3, columns, idx):
    """
    将list文件保存为excel
    :param data: slot与pkt
    :param data2: queue
    :param data3: flow
    :param columns: 纵索引
    :param idx:横索引
    :return: 保存excel文件到本地
    """
    # df1 = pd.DataFrame(data, columns=columns)
    # df2 = pd.DataFrame(data2, columns=columns)
    # df3 = data3.sort_index(ascending=True)
    # df1.set_index(pd.Index(idx), inplace=True)
    # df2.set_index(pd.Index(idx), inplace=True)
    # with pd.ExcelWriter("Schedule.xlsx") as writer:
    #     df1.to_excel(writer, sheet_name='Schedule')
    #     df2.to_excel(writer, sheet_name='Queue')
    #     data3.to_excel(writer, sheet_name="flow")


if __name__ == '__main__':
    a = [100, 300, 50, 200]
    b = [1000, 10000, 2000, 3000]
    S = np.array([np.arange(3) for i in range(2)], dtype=str)
    # 函数测试
    t = e2e_delay()

