import networkx as nx
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import itertools
from community import modularity
from networkx.generators.random_graphs import erdos_renyi_graph

def visualize_communities(partition, G, pos):
    cmap = cm.get_cmap('viridis', max(partition.values()) + 1)
    nx.draw_networkx_nodes(G, pos, partition.keys(), node_size=40,
                           cmap=cmap, node_color=list(partition.values()))
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    plt.show()


def visualize_graph(G):
    options = {
        'node_color': 'black',
        'node_size': 50,
        'width': 2,
    }
    plt.subplot()
    nx.draw(G, **options)
    plt.show()

def modularity_random(partition, G):
    connected = lambda x, y: int(x in G.neighbors(y))
    delta = lambda x, y: int(partition[x] == partition[y])
    m = G.size()
    n = G.number_of_nodes()
    # value of P in a random graph
    p = (2*m) / (n * (n-1))
    f = lambda i, j: (connected(i, j) - p) * delta(i, j)
    Q = (1 / (2*m)) * sum([f(i, j) for (i, j) in itertools.product(range(n), range(n))])
    return Q

def simple_graph():
    G = nx.Graph()
    G.add_nodes_from([i for i in range(9)])
    G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (1, 4), (0, 3)])
    G.add_edges_from([(5, 6), (6, 7), (7, 8), (8, 5), (5, 7)])
    G.add_edge(3, 5)
    partition = {v: 1 if v <= 4 else 2 for v in G}
    return G, partition


def main():
    G, partition = simple_graph()
    # visualize_graph(G)
    print(f"Modularity value using NetworkX: {modularity(partition, G)}")
    print(f"Modularity value using a random graph as the null model : {modularity_random(partition, G)}")
    pos = nx.spring_layout(G)
    visualize_communities(partition, G, pos)
    

if __name__ == "__main__":
    main()
