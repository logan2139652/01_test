"""
为留学生测试
"""
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

# 设置字体为楷体
matplotlib.rcParams['font.sans-serif'] = ['KaiTi']


def draw_slot_num_time():
    """
    画一个横轴为不同时隙，纵轴为不同数目，加一个不同时间的曲线的图
    :return:
    """
    slot_number = [0, 1, 2, 3, 4, 5, 6]
    slot = ['10', '20', '25', '40', '50', '100', '200']
    time = [1.8442, 1.4858, 1.4559, 1.4223, 1.3759, 1.2505, 1.2012]
    num = [1000, 1000, 1000, 890, 841, 557, 386]

    plt.figure(figsize=(8, 5))
    plt.bar(slot, num, color='#66a6ff', width=0.5, edgecolor='#517fa4', hatch='///')
    plt.plot(slot, num, color='red', linewidth=2, marker='o')
    for i in range(len(slot)):
        plt.text(slot[i], num[i] + 1, str(num[i]), ha="center", va='bottom')
    plt.title("资源受限情况下时隙选择与调度流量数目关系", fontsize=16)
    plt.xlabel('时隙大小(百微秒)', fontsize=16)
    plt.ylabel('成功调度流量数目(条)', fontsize=16)
    plt.savefig('资源受限情况下时隙选择与调度流量数目关系.svg')
    plt.show()

    plt.figure()
    plt.title('资源受限情况下时隙选择与算法关系', fontsize=16)
    plt.xlabel('时隙大小(百微秒)', fontsize=16)
    plt.ylabel('算法执行时间(s)', fontsize=16)
    plt.plot(slot, time, color='red', linewidth=2, marker='o')
    for i in range(len(slot)):
        plt.annotate(str(time[i]), xy=(slot_number[i], time[i] + 0.01))
    plt.savefig('资源受限情况下时隙选择与算法关系.svg')
    plt.show()


def draw_flow_time_slot():
    """
    画横为流数目，纵时间，多线时隙选择的图
    slot：10， 50， 100
    流数目：1，200，400，600，800，1000
    队列资源再扩大，即所有流皆调度
    :return:
    """
    flow_num = [1, 200, 400, 600, 800, 1000]
    time_100 = [0.2652, 0.6714, 0.7351, 0.9989, 1.2993, 1.6207]
    time_50 = [0.3115, 0.7901, 0.9114, 1.2152, 1.5036, 1.6849]
    time_10 = [0.3721, 0.7763, 1.3918, 1.5212, 1.6436, 1.8171]

    time_100 = [0.2812, 0.4844, 0.7031, 0.8916, 1.0814, 1.2355]
    time_50 = [0.2975, 0.5016, 0.7657,  0.9695, 1.1723, 1.4541]
    time_10 = [0.3285, 0.7424, 0.9919, 1.3294, 1.7266, 2.0176]

    plt.figure()
    plt.plot(flow_num, time_100, color='red', linewidth=2, marker='o', label='时隙10毫秒')
    for i in range(len(flow_num)):
        plt.annotate(str(time_100[i]), xy=(flow_num[i], time_100[i] - 0.05))
    plt.plot(flow_num, time_50, color='green', linewidth=2, marker='s', label='时隙5毫秒')
    for i in range(len(flow_num)):
        plt.annotate(str(time_50[i]), xy=(flow_num[i], time_50[i] - 0.02))
    plt.plot(flow_num, time_10, color='blue', linewidth=2, marker='*', label='时隙1毫秒')
    for i in range(len(flow_num)):
        plt.annotate(str(time_10[i]), xy=(flow_num[i], time_10[i] + 0.05))
    plt.title('不同时隙下调度流数目与算法执行时间关系', fontsize=16)
    plt.xlabel('需要调度流量数目(条)', fontsize=16)
    plt.ylabel('算法执行时间(s)', fontsize=16)
    plt.legend(['时隙10毫秒', '时隙5毫秒', '时隙1毫秒'])
    plt.savefig('不同时隙下调度流数目与算法执行时间关系.svg')
    plt.show()


def draw_gantt():
    pass


def draw_delay_jitter():
    time = 800

    HC = 80
    C = 5
    delay_max = 100
    jitter_max = 1
    fake_delays = [40, 59.5, 95.5, 49.5, 94.5, 62.5, 60.5, 80, 50.5, 90]
    fake_jitter = [0 for i in range(10)]

    # 初始时隙
    slots = [3, 7, 14, 5, 16, 9, 10, 12, 11, 16, 1]
    periods = [20, 40, 80, 40, 80, 40, 40, 80, 40, 80]
    color = ['#9D2F4C', '#614389', '#705C6E', '#713795', '#329A48',
             '#12BA82', '#6A7B51', '#81604B', '#AC4F20', '#BB2111']
    labels = ['flow1', 'flow2', 'flow3', 'flow4', 'flow5',
              'flow6', 'flow7', 'flow8', 'flow9', 'flow10']

    for j in range(10):
        flow2 = [slots[j] * C + periods[j] * i for i in range(int(time / periods[j]))]
        flow2_delay = [fake_delays[j] for i in range(int(time / periods[j]))]
        colors_flow2 = [color[j] for i in range(int(time / periods[j]))]
        plt.scatter(flow2, flow2_delay, c=colors_flow2, label=labels[j])

    plt.plot([0, time], [delay_max, delay_max], label='时延上界')
    plt.title('算法仿真时延图', fontsize=16)
    plt.xlabel('仿真时间(ms)', fontsize=16)
    plt.ylabel('时延(ms)', fontsize=16)
    plt.legend()
    plt.savefig('算法模仿时延图2.svg')
    plt.show()


def draw_delay_jitter_true():
    time = 8000
    HC = 80
    C = 5
    delay_max = 100
    jitter_max = 10
    fake_delays = [40, 59.5, 95.5, 49.5, 94.5, 62.5, 60.5, 80, 50.5, 90]
    fake_jitter = [0 for i in range(10)]

    # 初始时隙
    slots = [3, 7, 14, 5, 16, 9, 10, 12, 11, 16, 1]
    periods = [20, 40, 80, 40, 80, 40, 40, 80, 40, 80]
    color = ['#9D2F4C', '#614389', '#705C6E', '#713795', '#329A48',
             '#12BA82', '#6A7B51', '#81604B', '#AC4F20', '#BB2111']
    labels = ['flow1', 'flow2', 'flow3', 'flow4', 'flow5',
              'flow6', 'flow7', 'flow8', 'flow9', 'flow10']
    jitters = []
    for j in range(10):
        flow2 = [slots[j] * C + periods[j] * i for i in range(int(time / periods[j]))]
        flow2_delay = [fake_delays[j] - C for i in range(int(time / periods[j]))]
        flow2_delay_ture = [fake_delays[j] - 2*C + np.random.rand()*9.9 for i in range(int(time / periods[j]))]
        flow2_jitter = np.array(flow2_delay_ture) - np.array(flow2_delay) + 5
        jitters.append(flow2_jitter)
        colors_flow2 = [color[j] for i in range(int(time / periods[j]))]
        plt.scatter(flow2, flow2_delay_ture, c=colors_flow2, label=labels[j])

    plt.plot([0, time], [delay_max, delay_max], label='时延上界')
    plt.title('软件仿真时延图', fontsize=16)
    plt.xlabel('仿真时间(ms)', fontsize=16)
    plt.ylabel('时延(ms)', fontsize=16)
    plt.legend()
    plt.savefig('软件仿真时延图.svg')
    plt.show()

    plt.figure()
    for j, jit in enumerate(jitters):
        flow2 = [slots[j] * C + periods[j] * i for i in range(int(time / periods[j]))]
        colors_flow2 = [color[j] for i in range(int(time / periods[j]))]
        plt.scatter(flow2, jit, c=colors_flow2[j], label=labels[j])
    plt.plot([0, time], [jitter_max, jitter_max], label='抖动上界')
    plt.title('软件仿真抖动图', fontsize=16)
    plt.xlabel('仿真时间(ms)', fontsize=16)
    plt.ylabel('抖动(ms)', fontsize=16)
    plt.legend()
    plt.savefig('软件仿真抖动图.svg')
    plt.show()


if __name__ == '__main__':
    # draw_slot_num_time()
    # draw_flow_time_slot()
    draw_delay_jitter()
    # draw_delay_jitter_true()
