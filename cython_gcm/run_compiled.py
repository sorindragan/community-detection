from community import modularity
import gcm

import networkx as nx
from networkx.generators.community import LFR_benchmark_graph
import matplotlib.cm as cm
import matplotlib.pyplot as plt


G = LFR_benchmark_graph(
        250, 2, 1.1, 0.1,
        min_degree=20,
        max_degree=50,
        max_iters=5000, 
        seed=10,
        )
# G = nx.karate_club_graph()
pos = nx.spring_layout(G)

partition = gcm.process(G, modularity)

cmap = cm.get_cmap('viridis', max(partition.values()) + 1)
nx.draw_networkx_nodes(G, pos, partition.keys(), node_size=40,
                       cmap=cmap, node_color=list(partition.values()))
nx.draw_networkx_edges(G, pos, alpha=0.5)
plt.show()

