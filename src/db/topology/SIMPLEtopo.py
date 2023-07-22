import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

G = nx.Graph()

G.add_edges_from([(0, 2), (1, 2), (2, 3), (3, 4), (3, 5)])

# label = {0: "access1", 1: "access2", 2: "core1", 3: "core2", 4: "access3", 5: "access4"}
colors = {0: "#1f78b4", 1: "#1f78b4", 2: "#FF0000", 3: "#FF0000", 4: "#1f78b4", 5: "#1f78b4"}
edge_width = {(0, 2): 2, (2, 1): 2, (2, 3): 5, (3, 4): 2, (3, 5): 2}
# print(G.edges())

node_pos = {0: (-3, 2), 1: (-3, -2), 2: (-2, 0), 3: (2, 0), 4: (3, 2), 5: (3, -2)}
label_pos = {0: (-2.7, 1.7), 1: (-2.7, -1.7), 2: (-2, 0.3), 3: (2, 0.3), 4: (2.7, 1.7), 5: (2.7, -1.7)}

# plt.figure(figsize=(15, 9)) (12,8)
plt.figure(figsize=(14, 9))
plt.title('simple network with 6 nodes', fontsize=30)
ax = plt.gca()
ax.margins(0.1)

nx.draw(G, pos=node_pos, node_color=[colors[node] for node in G.nodes()],
        node_size=2500, width=[edge_width[edge] for edge in G.edges()])
nx.draw_networkx_labels(G, pos=node_pos, font_size=30, font_color='#FFFFFF')
# plt.savefig("SIMPLE.svg")
plt.show()

