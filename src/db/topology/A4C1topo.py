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


def A4C1topo():
    """
    建立A4C1拓扑
    :return: G networkx
    """
    # edge连接
    dict1 = {0: [1, 2, 4], 1: [3, 13], 2: [3, 7], 3: [10], 4: [5, 6],
            5: [], 6: [], 7: [8, 9], 8: [], 9: [], 10: [11], 11: [12], 12: [],
             13: [14, 15], 14: [], 15: []}

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

    G.add_nodes_from(["h1", "h2", "h3", "h4", "h5", "h6", "h7", "h8", "h9", "h10"], category="host")
    G.add_edges_from([("h1", 5), ("h2", 6), ("h3", 7), ("h4", 9),
                      ("h5", 11), ("h6", 12), ("h7", 14), ("h8", 15),
                      ("h9", 8), ("h10", 13)])
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
    pos = {0: (-2, 2), 1: (2, 2), 2: (-2, -2), 3: (2, -2),
           4: (-4, 4), 5: (-6, 4), 6: (-3.5, 5.5),
           7: (-4, -4), 8: (-5, -4), 9: (-3, -5),
           10: (4, -4), 11: (5, -5), 12: (4, -5),
           13: (4, 4), 14: (5, 4), 15: (3, 5),
           "h1": (-6.5, 2.5), "h2": (-2, 5), "h3": (-5.5, -4.5), "h4": (-4.5, -5.5),
           "h5": (5.5, -6), "h6": (4.5, -6), "h7": (5.5, 4.5), "h8": (4.5, 5.5)}
    pos = nx.kamada_kawai_layout(G)

    # 节点标记
    lable = {0: "s0", 1: "s1", 2: "s2", 3: "s3",
             4: "s4", 5: "s5", 6: "s6", 7: "s7",
             8: "s8", 9: "s9", 10: "s10", 11: "s11",
             12: "s12", 13: "s13", 14: "s14", 15: "s15",
             "h1": "h1", "h2": "h2", "h3": "h3", "h4": "h4",
             "h5": "h5", "h6": "h6", "h7": "h7", "h8": "h8",
             "h9": "h9", "h10": "h0"}

    plt.title('A4C1 Net with 16 switches and 10 hosts')

    nx.draw_networkx_nodes(G, pos=pos,
                           nodelist=[4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
                           node_size=500,
                           node_color="#1f78b4",
                           )
    nx.draw_networkx_nodes(G, pos=pos,
                           nodelist=[0, 1, 2, 3],
                           node_size=500,
                           node_color="#FF0000",
                           # node_color="#1f78b4",
                           )
    nx.draw_networkx_nodes(G, pos=pos,
                           nodelist=["h1", "h2", "h3", "h4",
                                     "h5", "h6", "h7", "h8", "h9", "h10"],
                           node_size=300,
                           node_color="#fa709a",
                           node_shape='s')
    nx.draw_networkx_edges(G, pos=pos)
    nx.draw_networkx_labels(G, labels=lable, pos=pos, font_color='#FFFFFF')
    plt.savefig('A4C1topo.svg')
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
    path_source = ["h1", "h2", "h3", "h4", "h5", "h6", "h7", "h8", "h9", "h10"]
    path_dest = ["h1", "h2", "h3", "h4", "h5", "h6", "h7", "h8", "h9", "h10"]
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
    # print(route_dict)

    # 路径存储
    for i in range(len(df)):
        # print("----------------------")
        index = df.loc[i]["src"] + "to" + df.loc[i]["dst"]
        df.set_value(i, 'path', route_dict[index])
        # print(df.iloc[i])
    return df


if __name__ == '__main__':
    G = A4C1topo()
    draw(G)
    topoRoute(G, [])
