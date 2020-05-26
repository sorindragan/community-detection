import itertools
import networkx as nx
import matplotlib.cm as cm
import matplotlib.pyplot as plt

import community as community_louvain

from collections import defaultdict
from networkx.algorithms.community import greedy_modularity_communities
from networkx.algorithms.community import girvan_newman
from networkx.algorithms.community import modularity
from networkx.generators.community import LFR_benchmark_graph
from networkx.algorithms.community.quality import coverage, performance

from karateclub import LabelPropagation

# basic connected caveman community graph
def generate_caveman_graph():
    """
    -> returns a connected caveman graph of l cliques of size k
    """
    G = nx.connected_caveman_graph(5, 4)
    # G = nx.relaxed_caveman_graph(5, 4, 0.2)
    return G


# LFR benchmark graph generator
def genrate_lfr_graph():
    """
    -> n (int) – number of nodes
    -> tau1 (float > 1) – power law exponent for the degree distribution
    -> tau2 (float > 1) – power law exponent for the community size distribution
    -> mu (float in (0, 1)) – fraction of intra-community edges incident to each node
    -> average_degree (float in (0, n)) – average degree of nodes
    -> min_degree (int in (0, n)) – minimum degree
    -> max_degree (int) – maximum degree
    -> min_community (int, default min_degree) – minimum size of communities
    -> max_community (int, default n) – M-maximum size of communities
    -> tol (float) – tolerance when comparing floats, specifically when comparing average degree values   
    -> max_iters (int) – maximum number of iterations
    -> seed (integer, random_state, or None (default)) – indicator of random number generation state
    """
    params = {"n":100, "tau1":3, "tau2":1.1, "mu":0.1, "avg_degree":5, "max_degree":25, "min_community":5, "max_community":10}
    # n : 100, tau1 : 3, tau2 : 1.5, mu : 0.01, avg_degree:5, min_community:10, seed:10
    G = LFR_benchmark_graph(params["n"], params["tau1"], params["tau2"], params["mu"], 
                            average_degree=params["avg_degree"],
                            min_community=params["min_community"])
    print("Generated")

    # get the communities from the node attributes of the graph
    communities = {frozenset(G.nodes[v]['community']) for v in G}
    print(f"LFR graph communities: {communities}")
    
    return G

def visualize_graph(G):
    options = {
        'node_color': 'black',
        'node_size': 50,
        'width': 2,
        }
    plt.subplot()
    nx.draw(G, **options)
    plt.show()


def convert_to_sequence(partition):
    communities_dict = defaultdict(list)
    for node, c in partition.items():
        communities_dict[c].append(node)
    
    communities = [frozenset(c) for c in communities_dict.values()]
    return communities


def visualize_communities(partition, G, pos, show=True):
    cmap = cm.get_cmap('viridis', max(partition.values()) + 1)
    nx.draw_networkx_nodes(G, pos, partition.keys(), node_size=40,
                        cmap=cmap, node_color=list(partition.values()))
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    if show:
        plt.show()


def parallel_display(partitions, G, pos):
    length = len(partitions)
    for i in range(length):
        plt.subplot(1, length, i+1)
        visualize_communities(partitions[i], G, pos, show=False)
    plt.show()


def clauset_newman_moore(G):
    partition = {}
    communities = greedy_modularity_communities(G)
    for idx, c in enumerate(communities):
        for node in c:
            partition[node] = idx
    return partition, communities


def girvan_newman_(G, limit=False):
    partition = {}
    epsilon = 0.001
    com_gen = girvan_newman(G)
    
    if limit:
        limited = itertools.takewhile(lambda c: len(c) < limit, com_gen)
        com_gen = limited
    
    modularity_ = -1
    last_communities = []
    
    for communities in com_gen:
        curr_modularity = modularity(G, communities)
        if (curr_modularity - modularity_) < epsilon:
            communities = last_communities
            break
        modularity_ = curr_modularity
        last_communities = communities


    for idx, c in enumerate(communities):
        for node in c:
            partition[node] = idx
    return partition, communities


def louvain(G):
    partition = community_louvain.best_partition(G)
    communities = convert_to_sequence(partition)
    return partition, communities


def karateclub_algorithms(G, pos):
    model = LabelPropagation()
    model.fit(G)
    partition = model.get_memberships()
    print(partition)
    visualize_communities(partition, G, pos)


def individual_runs(G, pos):
    partition, communities = clauset_newman_moore(G)
    visualize_communities(partition, G, pos)
    
    partition, communities = girvan_newman_(G)
    visualize_communities(partition, G, pos)
    
    partition, communities = louvain(G)
    visualize_communities(partition, G, pos)

def main():
    print("Start process")
    
    algorithms = [clauset_newman_moore, girvan_newman_, louvain]
    
    G = generate_caveman_graph()
    # G = genrate_lfr_graph()
    
    # visualize_graph(G)
    pos = nx.spring_layout(G)

    # results = [alg(G) for alg in algorithms]

    # partitions = [r[0] for r in results]
    # communities = [(coverage(G, r[1]), performance(G, r[1]))  for r in results]

    # print(communities)
    # parallel_display(partitions, G, pos)

    # individual_runs(G, pos)

    karateclub_algorithms(G, pos)


if __name__ == "__main__":
    main()


