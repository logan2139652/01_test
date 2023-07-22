
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


dict1 = {0: [4, 5], 1: [4, 5], 2: [4, 5], 3: [4, 5], 4: [6, 7],
        5: [6, 8], 6: [7, 8], 7: [9, 10, 12], 8: [9, 12, 14], 9: [11, 13],
        10: [11], 11: [], 12: [], 13: [14], 14: []}

list1 = [[0 for j in range(15)] for i in range(15)]

for key, value in dict1.items():
    for edges in value:
        list1[key][edges] = 1

G = nx.Graph()
# USANET(with 24 nodes and 43 links)
Matrix = np.array(list1)

for i in range(len(Matrix)):
    for j in range(len(Matrix)):
        if Matrix[i, j] == 1:
            G.add_edge(i, j)
plt.title('CEV NET with 15 nodes')
# pos = nx.kamada_kawai_layout(G)

pos = {0: (-2, 2.2), 1: (-2.5, 1.2), 2: (-2.5, -1.2), 3: (-2, -2.2), 4: (-1, 1),
       5: (-1, -1), 6: (0, 0), 7: (1, 1), 8: (1, -1), 9: (2, 0),
       10: (2, 2.5), 11: (3, 2), 12: (3, 0), 13: (3, -2), 14: (2, -2.5)}

node_colors = {0: "#1f78b4", 1: "#1f78b4", 2: "#1f78b4", 3: "#1f78b4",
              4: "#FF0000", 5: "#FF0000", 6: "#FF0000",
              7: "#FF0000", 8: "#FF0000", 9: "#FF0000",
              10: "#1f78b4", 11: "#1f78b4", 12: "#1f78b4", 13: "#1f78b4", 14: "#1f78b4"}

nx.draw(G, with_labels=False, pos=pos, node_size=500,
        node_color=[node_colors[node] for node in G.nodes()])
nx.draw_networkx_labels(G, pos=pos, font_color='#FFFFFF')
plt.savefig('CEV NET.svg')
plt.show()

