"""
q-learning 运行函数
首先建立一个网络，
然后根据网络去建立环境，
将环境的参数信息传入q—learning核心函数中
最后调度q——learning更新主函数update
"""
import time
from src.data_processing.A4C1_data_processing import A4C1Net
from src.data_processing.A7C2_data_processing import A7C2Net
from src.scheduling.rl_Q_learning.traffic_env import Maze
from src.scheduling.rl_Q_learning.RL_brain import QLearningTable
from src.scheduling.rl_Q_learning.Run_this import update
import matplotlib.pyplot as plt

if __name__ == '__main__':
    print('运行q-learning算法')
    choose_slot = 40
    bandwidth_rate = 300  # unit : Bytes/0.1ms   eg:500 Bytes/0.1ms = 5M B/s = 40M bps
    queue_capacity = choose_slot * bandwidth_rate

    net = A4C1Net(
        # excel_path=r'D:\Python\PyCharmWorkplace\01ArticleOneCode\src\db\dataset\A4C1_1000_big.xlsx',
        excel_path=r'D:\Python\PyCharmWorkplace\01ArticleOneCode\src\db\dataset\A4C1_1000.xlsx',
        flow_number=1000,
        test_slot_size=choose_slot)

    a = time.time()
    env = Maze(net=net, queue_size=queue_capacity)
    # RL = QLearningTable(n_states=env.n_states, actions=list(range(env.n_actions)))
    RL = QLearningTable(n_states=env.n_states, actions=env.action_space,
                        learning_rate=0.1, reward_decay=0.9, e_greedy=0.8)
    update(env, RL, times=100)
    # print(RL.q_table)
    b = time.time()
    print(b - a)
