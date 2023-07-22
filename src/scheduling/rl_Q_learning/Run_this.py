"""
Q_learning 主函数
"""
from src.scheduling.data_processing.mininet_data import MiniNet
from src.scheduling.rl_Q_learning.traffic_env import Maze
from src.scheduling.rl_Q_learning.RL_brain import QLearningTable
import matplotlib.pyplot as plt


def update(env, RL, times=1000):
    loss_plt = []
    reward_plt = []
    all_actions = []
    # 调度流量数目
    forward_flows = []
    for episode in range(times):
        # initial observation
        print('第{0}次循环'.format(episode))
        # print(RL.q_table)
        observation = env.reset()
        reward_sum = 0
        q_temp = RL.q_table.copy(deep=True)
        epsilon_factor = episode * 0.001 + RL.epsilon
        # 测试
        actions = []
        while True:
            # RL choose action based on observation
            # 加入episode去动态变换贪婪参数
            action = RL.choose_action(observation, epsilon_factor)
            actions.append(action)
            # RL take action and get next observation and reward
            observation_, reward, done = env.step(action)

            # RL learn from this transition
            RL.learn(observation, action, reward, observation_)

            # swap observation
            observation = observation_

            reward_sum += reward
            # break while loop when end of this episode
            if done:
                loss = RL.q_table - q_temp
                loss_norm = (loss - loss.min()) / (loss.max() - loss.min())
                loss_value = loss_norm.std().std()
                print('总奖励为{0}，成功调度流量为{1}'.format(reward_sum, len(env.has_schedule), loss_value))
                forward_flows.append(len(env.has_schedule))
                loss_plt.append(loss_value)
                reward_plt.append(reward_sum)
                all_actions.append(actions)
                # print(env.env_list)
                break

    # end of game
    print('game over')
    plt.plot(loss_plt)
    plt.show()
    plt.plot(reward_plt)
    plt.show()
    plt.title('schedule flows')
    plt.plot(forward_flows)
    plt.show()
    print(forward_flows)
    print(max(forward_flows))
    # print(all_actions)


if __name__ == "__main__":
    net = MiniNet(
        excel_path=r'D:\Python\PyCharmWorkplace\ArticleOneCode\src\db\dataset\flow_mininet_1000.xlsx',
        flow_number=200,
        test_slot_size=20,
        CORE_NODE=None)
    env = Maze(net=net, queue_size=5000)
    # RL = QLearningTable(n_states=env.n_states, actions=list(range(env.n_actions)))
    RL = QLearningTable(n_states=env.n_states, actions=env.action_space,
                        learning_rate=0.1, reward_decay=0.9, e_greedy=0.95)
    update(env, RL)
