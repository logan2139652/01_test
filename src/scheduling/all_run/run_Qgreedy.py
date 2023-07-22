"""
q-learning运行一个情况，再通过greedy填补，增加流量数目
"""
import time

import matplotlib.pyplot as plt

from src.data_processing.A4C1_data_processing import A4C1Net
from src.scheduling.state_of_art.greedy_q_learning.traffic_env import Maze
from src.scheduling.state_of_art.greedy_q_learning.RL_brain import QLearningTable
from src.scheduling.state_of_art.greedy_q_learning.Run_this import update
from src.scheduling.state_of_art.q_learning_greedy.greedy import greedy_algorithm
import pandas as pd
import numpy as np

plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] # 用来正常显示中文标签SimHei
plt.rcParams['axes.unicode_minus'] = False # 用来正常显示负号

if __name__ == '__main__':
    print('运行Q_greedy算法')
    choose_slot = 50  # 0.1ms
    bandwidth = 200  # Bytes/0.1ms
    queue_capacity = choose_slot * bandwidth

    net = A4C1Net(
        excel_path=r'D:\Python\PyCharmWorkplace\01ArticleOneCode\src\db\dataset\A4C1_1000.xlsx',
        flow_number=1000,
        test_slot_size=choose_slot)
    env = Maze(net=net, queue_size=queue_capacity)

    # RL = QLearningTable(n_states=env.n_states, actions=list(range(env.n_actions)))
    RL = QLearningTable(n_states=env.n_states, actions=env.action_space,
                        learning_rate=0.1, reward_decay=0.9, e_greedy=0.9)
    best_result, schedule_plt = update(env, RL, times=1000)
    # print(RL.q_table)
    # best[0] = env.env_list
    # best[1] = env.has_schedule
    # best[2] = env.not_schedule
    plt.figure()
    plt.plot(schedule_plt)
    plt.title("Q-learning离线与Greedy在线", fontsize=12)
    plt.xlabel('迭代次数')
    plt.ylabel('调度流量数目', fontsize=12)
    plt.show()
    print(schedule_plt)
    greedy_algorithm(net, best_result, QUEUE_SIZE=queue_capacity)
