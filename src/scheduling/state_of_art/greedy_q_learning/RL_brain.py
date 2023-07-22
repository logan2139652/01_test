"""
@author:hx
@data:2023/05/09
q-learning核心代码
"""

import numpy as np
import pandas as pd


class QLearningTable:
    def __init__(self, n_states, actions, learning_rate=0.1,
                 reward_decay=0.9, e_greedy=0.9, q_table=None):
        self.actions = actions  # 动作空间
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        # self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)
        if q_table is None:
            self.q_table = pd.DataFrame(np.zeros((n_states, len(self.actions))),
                                         columns=self.actions, dtype=np.float64)
        else:
            self.q_table = q_table
        self.terminal_state = n_states - 2

    # 根据当前状态
    def choose_action(self, observation, epsilon_factor):
        self.check_state_exist(observation)

        if epsilon_factor <= 0.9:
            self.epsilon = epsilon_factor
            # print(self.epsilon)
        else:
            self.epsilon = 0.9

        # action selection
        if np.random.uniform() <= self.epsilon:
            # choose best action
            state_action = self.q_table.iloc[observation, :]
            # some actions may have the same value, randomly choose on in these actions
            action = np.random.choice(state_action[state_action == np.max(state_action)].index)
        else:
            # choose random action
            action = np.random.choice(self.actions)
        return action

    # 此处存在问题，建议后续改为loc，即通过横坐标的值来进行索引，而不是利用index值，否则会造成溢出
    def learn(self, s, a, r, s_):
        self.check_state_exist(s_)

        # 调度成功，也可以用r去判断
        q_predict_1 = self.q_table.iloc[s_ - 1, a]
        q_predict_2 = self.q_table.iloc[s_ - 2, a]
        # 调度失败
        q_predict = self.q_table.iloc[s, a]

        if s_ != self.terminal_state:  # s_ != 'terminal'
            q_target = r + self.gamma * self.q_table.iloc[s_, :].max()  # next state is not terminal
        else:
            q_target = r  # next state is terminal

        if s_ % 2 == 0:
            self.q_table.iloc[s_ - 1, a] += self.lr * (q_target - q_predict_1)
            self.q_table.iloc[s_ - 2, a] += self.lr * (q_target - q_predict_2)
        else:
            self.q_table.iloc[s, a] += self.lr * (q_target - q_predict)  # update

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table.append(
                pd.Series(
                    [0]*len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            )


if __name__ == '__main__':
    RL = QLearningTable(n_states=10, actions=list(range(10)))
    a = RL.choose_action(0, 0.95)
    print(a)
    # print(RL.q_table)
    # RL.learn(0, 0, 1, 1)
    # print(RL.q_table)
    # RL.learn(1, 0, 1, 2)
    print(RL.q_table)
    RL.check_state_exist(1)
    print(RL.q_table)