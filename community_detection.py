import time
import itertools
import networkx as nx
import matplotlib.cm as cm
import matplotlib.pyplot as plt

import community as community_louvain

from collections import defaultdict
from sklearn.metrics.cluster import normalized_mutual_info_score
from networkx.algorithms.community import greedy_modularity_communities
from networkx.algorithms.community import girvan_newman
from networkx.algorithms.community import modularity
from networkx.generators.community import LFR_benchmark_graph
from networkx.algorithms.community.quality import coverage, performance

from karateclub import LabelPropagation, GEMSEC


def convert_to_sequence(partition):
    communities_dict = defaultdict(list)
    for node, c in partition.items():
        communities_dict[c].append(node)
    
    communities = [frozenset(c) for c in communities_dict.values()]
    return communities


def convert_to_array(partition):
    return [partition[k] for k in partition]

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

    params = {"n":250, "tau1":2, "tau2":1.1, "mu":0.1, "min_degree":20, "max_degree":50}
    
    G = LFR_benchmark_graph(params["n"], params["tau1"], params["tau2"], params["mu"], 
                            min_degree=params["min_degree"],
                            max_degree=params["max_degree"],
                            max_iters=5000, seed = 10,
                            )
    print("Generation Completed")

    # get the communities from the node attributes of the graph
    comm_list = set([frozenset(G.nodes[v]['community']) for v in G])
    comm_dict = {comm: idx for idx, comm in enumerate(comm_list)}

    partition = {v:comm_dict[frozenset(G.nodes[v]['community'])] for v in G}
    communities = convert_to_sequence(partition)
    
    return G, partition, communities

def visualize_graph(G):
    options = {
        'node_color': 'black',
        'node_size': 50,
        'width': 2,
        }
    plt.subplot()
    nx.draw(G, **options)
    plt.show()

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
    start_time = time.time()
    communities = greedy_modularity_communities(G)
    print(f"Clauset Newman Moore ran in {time.time() - start_time} seconds")
    for idx, c in enumerate(communities):
        for node in c:
            partition[node] = idx
    return partition, communities


def girvan_newman_(G, limit=False):
    partition = {}
    epsilon = 0.001
    modularity_ = -1
    last_communities = []

    start_time = time.time()
    com_gen = girvan_newman(G)
    
    if limit:
        limited = itertools.takewhile(lambda c: len(c) < limit, com_gen)
        com_gen = limited

    for round_, communities in enumerate(com_gen):
        curr_modularity = modularity(G, communities)
        print(round_, curr_modularity, modularity_, curr_modularity - modularity_)
        if (curr_modularity - modularity_) < epsilon:
            communities = last_communities
            break
        modularity_ = curr_modularity
        last_communities = communities
    print(f"Girvan Newman ran in {time.time() - start_time} seconds and needed {round_} iterations")


    for idx, c in enumerate(communities):
        for node in c:
            partition[node] = idx
    
    return partition, communities


def louvain(G):
    start_time = time.time()
    partition = community_louvain.best_partition(G)
    print(f"Louvain ran in {time.time() - start_time} seconds")

    communities = convert_to_sequence(partition)
    return partition, communities


def label_propagation(G):
    model = LabelPropagation()
    start_time = time.time()
    model.fit(G)
    partition = model.get_memberships()
    print(f"Label Propagation ran in {time.time() - start_time} seconds")
    communities = convert_to_sequence(partition)
    return partition, communities

def gemsec_random_walks(G):
    model = GEMSEC()
    start_time = time.time()
    model.fit(G)
    partition = model.get_memberships()
    print(f"GEMSEC ran in {time.time() - start_time} seconds")
    communities = convert_to_sequence(partition)
    return partition, communities


def individual_runs(G, pos, target_partition=[]):
    partition, communities = clauset_newman_moore(G)
    visualize_communities(partition, G, pos)
   
    nmi = normalized_mutual_info_score(convert_to_array(target_partition),
                                       convert_to_array(partition))
    print('NMI: {:.4f}'.format(nmi))

    
    partition, communities = girvan_newman_(G)
    visualize_communities(partition, G, pos)
    
    partition, communities = louvain(G)
    visualize_communities(partition, G, pos)

    partition, communities = label_propagation(G, pos)
    visualize_communities(partition, G, pos)

    partition, communities = gemsec_random_walks(G)
    visualize_communities(partition, G, pos)


def main():
    print("Start process")
    
    nx_algorithms = [clauset_newman_moore, girvan_newman_, louvain]
    karate_algorithms = [label_propagation, gemsec_random_walks]
    
    # G = generate_caveman_graph()
    G, target_partition, target_communities = genrate_lfr_graph()
    
    # visualize_graph(G)
    pos = nx.spring_layout(G)

    # networkx algorithms
    results = [alg(G) for alg in nx_algorithms]

    partitions = [r[0] for r in results]
    performances = [(coverage(G, r[1]), performance(G, r[1]), 
                    normalized_mutual_info_score(convert_to_array(target_partition), convert_to_array(r[1])))  
                    for r in results]

    print(performances)
    parallel_display(partitions, G, pos)

    # karate club algorithms
    results = [alg(G) for alg in karate_algorithms]

    partitions = [r[0] for r in results]
    performances = [(coverage(G, r[1]), performance(G, r[1]), 
                    normalized_mutual_info_score(convert_to_array(target_partition), convert_to_array(r[1])))  
                    for r in results]

    print(performances)
    parallel_display(partitions, G, pos)

    individual_runs(G, pos, target_partition=target_partition)

if __name__ == "__main__":
    main()


