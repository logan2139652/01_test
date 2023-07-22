"""
贪婪算法运行函数
首先建立一个网络，调用参数中CORE_NODE默认是None对应[3,4,5]，之后可以根据拓扑设立CORE
然后调用贪婪算法主函数
"""

from src.data_processing.A4C1_data_processing import A4C1Net
from src.data_processing.A7C2_data_processing import A7C2Net
from src.scheduling.heuristic_greedy.greedy import greedy_algorithm
import time
import pandas as pd
import numpy as np

if __name__ == '__main__':
    choose_slot = 50
    bandwidth_rate = 200  # unit : Bytes/0.1ms   eg:500 Bytes/0.1ms = 5M B/s = 40M bps
    queue_capacity = choose_slot * bandwidth_rate

    net = A7C2Net(
        # excel_path=r'D:\Python\PyCharmWorkplace\01ArticleOneCode\src\db\dataset\A4C1_1000_big.xlsx',
        excel_path=r'D:\Python\PyCharmWorkplace\01ArticleOneCode\src\db\dataset\A7C2_1000.xlsx',
        flow_number=1000,
        test_slot_size=choose_slot)

    a = time.time()
    actions = greedy_algorithm(net, QUEUE_SIZE=queue_capacity)
    b = time.time()
    print("算法时间:", b - a)
    # print(actions)