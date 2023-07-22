from src.data_processing.A4C1_data_processing import A4C1Net
from src.data_processing.A7C2_data_processing import A7C2Net
from src.scheduling.drl_dqn.traffic_env import Maze  # 导入环境
import torch  # 导入torch
import torch.nn as nn  # 导入torch.nn
import torch.nn.functional as F  # 导入torch.nn.functional
import numpy as np  # 导入numpy
import gym  # 导入gym
# 超参数
BATCH_SIZE = 300 #128  # 样本数量
LR = 0.01  # 学习率
EPSILON = 0.9  # greedy policy
GAMMA = 0.9  # reward discount
TARGET_REPLACE_ITER = 2000000  # 目标网络更新频率
MEMORY_CAPACITY = 20000  # 记忆库容量
"""
BATCH_SIZE = 200 #128  # 样本数量
LR = 0.01  # 学习率
EPSILON = 0.9  # greedy policy
GAMMA = 0.9  # reward discount
#TARGET_REPLACE_ITER = 100  # 目标网络更新频率
TARGET_REPLACE_ITER = 1000  # 目标网络更新频率
MEMORY_CAPACITY = 20000  # 记忆库容量
"""
max_iteration = 100
# env = gym.make('CartPole-v0').unwrapped         # 使用gym库中的环境：CartPole，且打开封装(若想了解该环境，请自行百度)
choose_slot = 50
bandwidth_rate = 370  # 375  # unit : Bytes/0.1ms   eg:500 Bytes/0.1ms = 5M B/s = 40M bps
queue_capacity = choose_slot * bandwidth_rate


net = A7C2Net(
    # excel_path=r'D:\Python\PyCharmWorkplace\01ArticleOneCode\src\db\dataset\A4C1_1000_big.xlsx',
    excel_path=r'D:\Python\PyCharmWorkplace\01ArticleOneCode\src\db\dataset\A7C2_1000.xlsx',
    flow_number=1000,
    test_slot_size=choose_slot)

env = Maze(net=net, queue_size=queue_capacity)

N_ACTIONS = env.n_actions  # 状态动作个数 (2个)
N_STATES = env.n_feature  # 杆子状态个数 (4个)

# 定义DQN网络
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=(3,))
        self.fc1 = nn.Linear(16 * (N_STATES - 2), N_ACTIONS)

    def forward(self, x):
        x = x.reshape(1, 1, N_STATES)
        x = F.relu(self.conv1(x))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = x.view(x.size(0), N_ACTIONS, 1)
        x = torch.transpose(x, 1, 2)
        x = torch.squeeze(x, dim=1)
        return x
# 加载DQN网络

dqn = Net().cuda()

state_dict = torch.load('dqn_params.pth')
dqn.load_state_dict(state_dict)

# 输入数据
input_data = torch.tensor([[235.,  16.,  40.,   0.,   0.,   0.,   0.,   0.,   0.,   1.,   0.,   0.,
           0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   1.,   0.,
           0.,   0.,   0.,   1.,   0.,   1.,   0.,   0.,   0.,   0.,   0.,   0.,
           1.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   1.,   0.,   0.,
           0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.]],
       device='cuda:0')

# 使用模型进行预测
import time
a = time.time()
output = dqn.forward(input_data.cuda())
print(max(output))
b = time.time()
print(b - a)
