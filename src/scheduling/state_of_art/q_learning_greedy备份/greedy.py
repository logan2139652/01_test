"""
@ 基于贪婪的算法设计
@ author : hx
@ create : 2023/5/6
@ info: 基于贪婪的算法主函数, 与src.scheduling.data_processing.mininet_data.py配合和本文件夹下greedy_tool.py配合
"""
import copy
from src.data_processing.A4C1_data_processing import A4C1topo
from src.scheduling.state_of_art.q_learning_greedy import greedy_tool
import math
import time


def hop2through(hop_slot):
    """
    将hop_slot转换为through_slot
    edge的[1,2,2,1,1] ---> 涉及到六个节点，取前五个，分别对应前五个节点的port，
    最后一个port的发送队列无用，即变为[0, 1, 2, 2, 1]
    :param hop_slot:
    :return:
    """
    through_slot = copy.deepcopy(hop_slot)
    through_slot.insert(0, 0)
    through_slot.pop()
    return through_slot


def greedy_algorithm(network, env=None, QUEUE_SIZE=10000):
    """
    贪婪算法主函数
    :param QUEUE_SIZE: 队列容量，默认为10000
    :param env : q——learning后的环境   best[0] = env.env_list / best[1] = env.has_schedule / best[2] = env.not_schedule
    :param network:网络，networkx的Graph属性，是一个类，包括很多属性
    :return:
    """
    # 显示所有列    pd.set_option('display.max_columns', None)
    # 显示所有行    pd.set_option('display.max_rows', None)
    print('----------------------------greedy---------------------')
    # 数据提取
    flow_name = network.flow_name  # 又名flow id
    flow_period = network.flow_period
    flow_length = network.flow_length
    flow_starttime = network.flow_starttime
    flow_deadline = network.flow_deadline
    # flow_through_time = data_mine.E2e_flow
    flow_path_id = network.flow_path_id  # 注意是edge_id

    all_ports = network.net_edge_port_dict

    # 计算GCL与时隙
    GCL = network.HyperCycle
    slot_list = network.Cycle_list
    print("GCL周期为", GCL)
    print("可选时隙列表为", slot_list)

    slot = network.Cycle
    print('所选slot', slot)
    slotnum = network.Cycle_num

    # 调度矩阵与队列矩阵
    schedule_list = [[[] for x in range(slotnum)] for y in range(len(all_ports))]  # 流调度矩阵
    queue_size = QUEUE_SIZE  # 队列容量
    if env is None:
        queue_list = [[queue_size for x in range(slotnum)] for y in range(len(all_ports))]  # 队列资源矩阵
        failed_flow = [i for i in range(len(flow_name))]
        success_flow = []
    else:
        queue_list = env[0]
        success_flow = env[1]
        failed_flow = env[2]

    schedule_fail = []  # 调度失败流量集合
    schedule_success = []  # 调度成功流量集合
    flow_list = network.flow_data  # 流量信息

    # record
    max_slot = []  # 记录成功调度流可选区最大时隙
    all_e2e = []  # 记录成功调度流的端到端时间
    actions_slot_route = [-1 for i in range(len(flow_name))]  # 记录每个流执行的slot与route，对应到q——learning中的action

    print("--------------------------------------算法开始-------------------------------")
    for i in failed_flow:
        fail = 1  # 判断该流是否成功调度，默认为失败，成功调度置为0
        # 选择路径
        for path_index in range(len(flow_path_id[i])):

            pathid = flow_path_id[i][path_index]
            hop_slot = network.flow_hop_slot[i][path_index]

            # 将hop_slot头尾去掉，并在hop_slot头部加上一个0
            through_slot = hop2through(hop_slot)

            # 如果deadline都无法满足，直接跳出
            offset = greedy_tool.offset_constraint(flow_deadline[i], slot, hop_slot, flow_period[i])
            if offset < 0:
                break

            pktslot = int(math.ceil(flow_starttime[i] / slot))
            maxoffset = offset + pktslot

            max_slot.append(maxoffset)
            pktnum = int(GCL / flow_period[i])  # 一个Hypercycle中的数据包个数
            pkthop = int(flow_period[i] / slot)  # 单个流的相邻两数据包之间的时隙差值

            # maxoffset = 0  # 测试专用
            # 时隙选择，在最大时隙与发送时隙范围内
            for current_slot in range(maxoffset, pktslot-1, -1):
                flag = greedy_tool.queue_constraint(queue_list, pathid, current_slot, flow_length[i],
                                                through_slot, pktnum, pkthop, slotnum)

                if flag:
                    queue_list = flag
                    schedule_list = greedy_tool.enter_schedule_list(through_slot, current_slot, slotnum, i, pktnum, pkthop,
                                                       flow_name, schedule_list, pathid)
                    fail = 0
                    # 报错，但不知为何
                    # flow_list.set_value(i, "path", network.flow_path[i][path_index])
                    # flow_list.set_value(i, "slot", current_slot)
                    actions_slot_route[i] = path_index * slotnum + current_slot
                    break
                else:
                    continue
            # 当前路径所有时隙选择循环结束或者跳出后，判断是否调度成功，如果没有则选择下一条路径，否则代表成功，不再调度，跳出
            if fail:
                continue
            else:
                break
        # 所有路径遍历完或者成功调度
        if fail:
            schedule_fail.append(flow_name[i])
            flow_list.set_value(i, "path", 0)
            flow_list.set_value(i, "slot", 0)
        else:
            schedule_success.append(flow_name[i])

    print("----------------------------------------调度结束-----------------------------------")
    print("q-learning成功调度流量数目{1}, greedy成功调度流量数目{0}".format(len(schedule_success), len(success_flow)))
    print("成功调度流量数目{0}".format(len(schedule_success) + len(success_flow)))
    print("失败调度流量数目{0},流量包括:{1}".format(len(schedule_fail), schedule_fail))
    print("调度成功率为：", 1 - len(schedule_fail) / len(flow_name))

    # print(np.array(queue_list))
    per_q = []
    for i in actions_slot_route:
        if i >= 0:
            per_q.append(i)
            per_q.append(-1)
        else:
            per_q.append(-1)
            per_q.append(i)
    return per_q, actions_slot_route


if __name__ == '__main__':
    net = MiniNet(
        excel_path=r'D:\Python\PyCharmWorkplace\ArticleOneCode\src\db\dataset\flow_mininet_1000.xlsx',
        flow_number=2,
        test_slot_size=20,
        CORE_NODE=None)
    a = time.time()
    greedy_algorithm(net, QUEUE_SIZE=100)
    b = time.time()
    print("算法时间:", b - a)
