"""
@ author:hx
@ data:2023/05/09
拓扑与路径生成
与src.scheduling.data_processing.mininet_data.py文件配合使用
"""
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd


def A7C2topo():
    """
    建立A7C2拓扑
    :return: G networkx
    """
    # edge连接
    dict1 = {0: [1, 2, 6, 9], 1: [2, 3, 12, 4], 2: [14], 3: [4, 5, 23], 4: [5, 17],
            5: [20], 6: [7, 8], 7: [], 8: [], 9: [10, 11], 10: [], 11: [], 12: [13],
             13: [], 14: [15, 16], 15: [], 16: [], 17: [18, 19], 18: [], 19: [],
             20: [21, 22], 21: [], 22: [], 23: [24, 25], 24: [], 25: []}

    list1 = [[0 for j in range(len(dict1))] for i in range(len(dict1))]

    for key, value in dict1.items():
        for edges in value:
            list1[key][edges] = 1

    # 建立拓扑G
    G = nx.Graph()

    # 加入edge
    Matrix = np.array(list1)
    for i in range(len(Matrix)):
        for j in range(len(Matrix)):
            if Matrix[i, j] == 1:
                G.add_edge(i, j)

    G.add_nodes_from(["h1", "h2", "h3", "h4", "h5", "h6", "h7", "h8", "h9", "h10",
                      "h11", "h12", "h13", "h14"], category="host")
    G.add_edges_from([("h1", 7), ("h2", 8), ("h3", 10), ("h4", 11),
                      ("h5", 12), ("h6", 13), ("h7", 15), ("h8", 16),
                      ("h11", 21), ("h12", 22), ("h13", 24), ("h14", 25),
                      ("h9", 18), ("h10", 19)])
    return G


def draw(G):
    """
    对图G绘制
    :param G:
    :return:
    """
    if G is None:
        G = A4C1topo()
    # 节点位置信息
    # pos = nx.kamada_kawai_layout(G)

    pos = nx.kamada_kawai_layout(G)

    # 节点标记
    lable = {0: "s0", 1: "s1", 2: "s2", 3: "s3", 4: "s4",
             5: "s5", 6: "s6", 7: "s7",8: "s8", 9: "s9", 10: "s10",
             11: "s11", 12: "s12", 13: "s13", 14: "s14", 15: "s15",
             16: "s16", 17: "s17", 18: "s18", 19: "s19", 20: "s20",
             21: "s21", 22: "s22", 23: "s23", 24: "s24", 25: "s25",
             "h1": "h1", "h2": "h2", "h3": "h3", "h4": "h4",
             "h5": "h5", "h6": "h6", "h7": "h7", "h8": "h8",
             "h9": "h9", "h10": "h10", "h11": 'h11', "h12": "h12",
             "h13": "h13", "h14": "h0"}

    plt.title('A7C2 Net with 26 switches and 14 hosts')

    nx.draw_networkx_nodes(G, pos=pos,
                           nodelist=[i for i in range(6, 26)],
                           node_size=500,
                           node_color="#1f78b4",
                           )
    nx.draw_networkx_nodes(G, pos=pos,
                           nodelist=[0, 1, 2, 3, 4, 5],
                           node_size=500,
                           node_color="#FF0000",
                           # node_color="#1f78b4",
                           )
    nx.draw_networkx_nodes(G, pos=pos,
                           nodelist=["h1", "h2", "h3", "h4",
                                     "h5", "h6", "h7", "h8", "h9", "h10",
                                     "h11", "h12", "h13", "h14"],
                           node_size=400,
                           node_color="#fa709a",
                           node_shape='s')
    nx.draw_networkx_edges(G, pos=pos)
    nx.draw_networkx_labels(G, labels=lable, pos=pos, font_color='#FFFFFF')
    plt.savefig('A7C2topo.svg')
    plt.show()

# —————————————————————————————————————路由—————————————————————————————————————————————#
# routing
# —————————————————————————————————————路由—————————————————————————————————————————————#
def transformRoute2List(allpath):
    """
    将[(0,1),(1,2),(2,3)]路由转换为[0,1,2,3]
    仅仅针对edge_path使用, 但是edge_path的函数比较少，因此可以基本不用该函数
    :param allpath: 一个源到目的的所有路径，[[(),(),()], [(),(),()], [(),(),()]]
    :return: [[],[],[]]
    """
    result = []
    for path in allpath:
        temp = []  #单个path的
        for index, item in enumerate(path):
            if index == len(path) - 1:
                temp.extend(item)  # extend将tuple叉开为一个个元素加入
            else:
                temp.append(item[0])  # extend无法添加item[0]
        result.append(temp)
    return result


def transformRoute2tuple(allpath):
    """
    将路由[0,1,2,3]转换为[(0,1),(1,2),(2,3)]
    仅仅针对path使用
    :param allpath: 一个源到目的的所有路径，[[],[],[]]
    :return: [[(),(),()], [(),(),()], [(),(),()]]
    """
    result = []
    for path in allpath:
        temp = []
        for index, item in enumerate(path):
            try:
                if index == len(path) - 2:
                    temp.append(tuple((path[index], path[index+1])))
                    break
                elif index == len(path) - 1:
                    temp.append(tuple((path[index], path[index])))
                else:
                    temp.append(tuple((path[index], path[index+1])))
            except IndexError:
                print("-------------------------------------")
                print(path)
                print("-------------------------------------")
        result.append(temp)
    return result


def topoRoute(G, df):
    """
    拓扑路径生成
    :param G: 图 networkx
    :param df: 数据 dataframe
    :return: 生成路径后的df
    """
    # host2ip_dict = {"10.0.1.1": "h1", "10.0.2.2": "h2", "10.0.3.3": "h3", "10.0.4.4": "h4"}
    # path_key = [f"{i}to{j}" for j in host2ip_dict.keys() for i in host2ip_dict.keys() if i != j]
    # path_source = [i for j in host2ip_dict.values() for i in host2ip_dict.values() if i != j]
    # path_dest = [j for j in host2ip_dict.values() for i in host2ip_dict.values() if i != j]
    # 源与目的
    path_source = ["h1", "h2", "h3", "h4", "h5", "h6", "h7", "h8", "h9", "h10",
                   "h11", "h12", "h13", "h14"]
    path_dest = ["h1", "h2", "h3", "h4", "h5", "h6", "h7", "h8", "h9", "h10",
                 "h11", "h12", "h13", "h14"]
    path_key = [f"{i}to{j}" for j in path_dest for i in path_source if i != j]
    src = [i for j in path_dest for i in path_source if i != j]
    dst = [j for j in path_dest for i in path_source if i != j]
    path_value = []

    for i, j in zip(src, dst):
        all_path = []
        for path in nx.all_simple_paths(G, source=i, target=j):
            path.pop(0)
            path.pop()
            all_path.append(path)
        path_value.append(transformRoute2tuple(all_path))

    route_dict = dict(zip(path_key, path_value))

    # 路径存储
    for i in range(len(df)):
        # print("----------------------")
        index = df.loc[i]["src"] + "to" + df.loc[i]["dst"]
        df.set_value(i, 'path', route_dict[index])
        # print(df.iloc[i])
    return df


if __name__ == '__main__':
    G = A7C2topo()
    draw(G)
    topoRoute(G, [])
