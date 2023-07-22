"""
@ author:hx
@ data:2023/05/09
Mininet 拓扑与路径生成
与src.scheduling.data_processing.mininet_data.py文件配合使用
"""
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd


def cevNetTopo():
    """
    建立mininet拓扑
    :return: G networkx
    """
    # edge连接
    dict1 = {0: [4], 1: [3], 2: [3], 3: [4, 5], 4: [5],
            5: [6, 7], 6: [8], 7: [8], 8: []}
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

    G.add_nodes_from(["h1", "h2", "h3", "h4"], category="host")
    G.add_edges_from([("h1", 0), ("h2", 1), ("h3", 2), ("h4", 8)])


    # 节点位置信息
    # pos = nx.kamada_kawai_layout(G)
    pos = {0: (-1.5, 1.5), 1: (-2.5, 0.5), 2: (-2.5, -0.5),
           3: (-1.5, 0), 4: (0, 1), 5: (1.5, 0),
           6: (2.5, 0.5), 7: (2.5, -0.5), 8: (3.5, 0),
           "h1": (-2.5, 1.5), "h2": (-3.5, 0.5), "h3": (-3.5, -0.5), "h4": (4.5, 0)}
    # 节点颜色
    node_colors = {0: "#1f78b4", 1: "#1f78b4", 2: "#1f78b4", 3: "#FF0000",
                   4: "#FF0000", 5: "#FF0000", 6: "#1f78b4", 7: "#1f78b4",
                   8: "#1f78b4","h1": "#1f78b4", "h2": "#1f78b4", "h3": "#1f78b4", "h4": "#1f78b4"}
    # 节点标记
    lable = {0: "s1", 1: "s2", 2: "s3", 3: "s4",
                   4: "s5", 5: "s6", 6: "s7", 7: "s8",
                   8: "s9", "h1": "h1", "h2": "h2", "h3": "h3", "h4": "h4"}

    lable = {0: "s0", 1: "s1", 2: "s2", 3: "s3",
                   4: "s4", 5: "s5", 6: "s6", 7: "s7",
                   8: "s8", "h1": "h1", "h2": "h2", "h3": "h3", "h4": "h4"}

    plt.title('MiniNET with 9 nodes')

    nx.draw_networkx_nodes(G, pos=pos,
                           nodelist=[0,1,2,6,7,8],
                           node_size=500,
                           node_color="#1f78b4",
                           )
    nx.draw_networkx_nodes(G, pos=pos,
                           nodelist=[3, 4, 5],
                           node_size=500,
                           node_color="#FF0000",
                           # node_color="#1f78b4",
                           )
    nx.draw_networkx_nodes(G, pos=pos,
                           nodelist=["h1", "h2", "h3", "h4"],
                           node_size=300,
                           node_color="#1f78b4",
                           node_shape='s')
    nx.draw_networkx_edges(G, pos=pos)
    nx.draw_networkx_labels(G, labels=lable, pos=pos, font_color='#FFFFFF')
    # plt.savefig('MiniNET.svg')
    plt.show()
    return G


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
            if index == len(path) - 2:
                temp.append(tuple((path[index], path[index+1])))
                break
            else:
                temp.append(tuple((path[index], path[index+1])))
        result.append(temp)
    return result


def topoRoute(G, df):
    """
    拓扑路径生成
    :param G: 图 networkx
    :param df: 数据 dataframe
    :return: 生成路径后的df
    """
    host2ip_dict = {"10.0.1.1": "h1", "10.0.2.2": "h2", "10.0.3.3": "h3", "10.0.4.4": "h4"}
    path_key = [f"{i}to{j}" for j in host2ip_dict.keys() for i in host2ip_dict.keys() if i != j]
    path_source = [i for j in host2ip_dict.values() for i in host2ip_dict.values() if i != j]
    path_dest = [j for j in host2ip_dict.values() for i in host2ip_dict.values() if i != j]

    path_value = []
    for src, dest in zip(path_source, path_dest):
        all_path = []
        for path in nx.all_simple_paths(G, source=src, target=dest):
            path.pop(0)
            path.pop()
            all_path.append(path)
        path_value.append(transformRoute2tuple(all_path))

    route_dict = dict(zip(path_key, path_value))

    # 测试专用，固定路径
    '''
    route_dict["10.0.1.1to10.0.4.4"] = [[(0, 4), (4, 5), (5, 7), (7, 8)]]
    route_dict["10.0.2.2to10.0.4.4"] = [[(1, 3), (3, 5), (5, 6), (6, 8)]]
    route_dict["10.0.3.3to10.0.4.4"] = [[(2, 3), (3, 5), (5, 7), (7, 8)]]
    route_dict["10.0.2.2to10.0.3.3"] = [[(1, 3), (3, 2)]]
    '''

    # 路径存储
    for i in range(len(df)):
        # print("----------------------")
        index = df.loc[i]["src"] + "to" + df.loc[i]["dst"]
        # print(index)
        # print(route_dict[index])
        df.set_value(i, 'path', route_dict[index])
        # print(df.iloc[i])
    return df


if __name__ == '__main__':
    G = mininetTopo()
