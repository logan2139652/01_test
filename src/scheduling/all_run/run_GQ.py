"""
greedy生成q-table，指导q-learning，但是这个指导可正可负，可为0

然后在加上q-learning进行填补

# env中有一处错误，slot_num写成了slot_size

"""
import time
from src.data_processing.A4C1_data_processing import A4C1Net
from src.data_processing.A7C2_data_processing import A7C2Net
from src.scheduling.state_of_art.greedy_q_learning.traffic_env import Maze
from src.scheduling.state_of_art.greedy_q_learning.RL_brain import QLearningTable
from src.scheduling.state_of_art.greedy_q_learning.Run_this import update
from src.scheduling.state_of_art.q_learning_greedy.greedy import greedy_algorithm
import pandas as pd
import numpy as np


def action2qtable(env, actions):
    q_table = np.zeros((env.n_states, len(env.action_space)))
    for index, value in enumerate(actions):
        if value >= 0:
            # 如果偶数次的调度成功，那么奇数次失败的是否要做出和偶数次的相同的动作，因为其都指向下一个状态
            q_table[index - 1][value] = 0.0
            q_table[index][value] = 0.0
    q_table = pd.DataFrame(q_table, index=[i for i in range(env.n_states)], columns=env.action_space)
    return q_table


if __name__ == '__main__':
    print('运行state of art算法')
    choose_slot = 50
    bandwidth_rate = 300  # unit : Bytes/0.1ms   eg:500 Bytes/0.1ms = 5M B/s = 40M bps
    queue_capacity = choose_slot * bandwidth_rate

    net = A7C2Net(
        # excel_path=r'D:\Python\PyCharmWorkplace\01ArticleOneCode\src\db\dataset\A4C1_1000_big.xlsx',
        excel_path=r'D:\Python\PyCharmWorkplace\01ArticleOneCode\src\db\dataset\A7C2_1000.xlsx',
        flow_number=1000,
        test_slot_size=choose_slot)

    actions, actions_slot_route, schedule_num = greedy_algorithm(net, QUEUE_SIZE=queue_capacity)
    env = Maze(net=net, queue_size=queue_capacity)
    schedule_all = []
    a = time.time()
    for i in range(1):
        print(f'第{i}次循环')
        # 建立q-table
        q_table = action2qtable(env, actions)
        # 通过greedy指导q-learning
        # RL = QLearningTable(n_states=env.n_states, actions=env.action_space,
        #                     learning_rate=0.1, reward_decay=0.9, e_greedy=0.9,
        #                     q_table=q_table)
        RL = QLearningTable(n_states=env.n_states, actions=env.action_space,
                            learning_rate=0.1, reward_decay=0.9, e_greedy=0.9)
        best_result, schedule_plt = update(env, RL, times=50)
        # print('第{}次结果为{}'.format(i+1, len(best_result[1])))
        # 在通过q-learning指导greedy
        actions, actions_slot_route, schedule_num = greedy_algorithm(net, best_result, QUEUE_SIZE=queue_capacity)
        print(schedule_num)
        schedule_all.append(schedule_num)
    print(schedule_all)
    # print(max(schedule_all))
    b = time.time()
    print("算法时间:", b - a)