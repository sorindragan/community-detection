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
    partition = dict(sorted(partition.items()))
    return [partition[k] for k in partition]


def generate_karate_club_graph():
    return nx.karate_club_graph()

# basic connected caveman community graph
def generate_caveman_graph(cliques=5, size=4):
    """
    -> returns a connected caveman graph of l cliques of size k
    """
    return nx.connected_caveman_graph(cliques, size)


# LFR benchmark graph generator
def genrate_lfr_graph(size=250):
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

    params = {"n":size, "tau1":2, "tau2":1.1, "mu":0.1, "min_degree":20, "max_degree":50}
    
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
    # print(partition)
    
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

def louvain(G):
    start_time = time.time()
    partition = community_louvain.best_partition(G)
    print(f"Louvain ran in {time.time() - start_time} seconds")
    communities = convert_to_sequence(partition)
    return partition, communities

def individual_runs(G, pos, target_partition=[]):
    partition, communities = clauset_newman_moore(G)
    visualize_communities(partition, G, pos)
    
    nmi = normalized_mutual_info_score(convert_to_array(target_partition),
                                       convert_to_array(partition))
    print('NMI: {:.4f}'.format(nmi))

    partition, communities = louvain(G)
    visualize_communities(partition, G, pos)
    
    nmi = normalized_mutual_info_score(convert_to_array(target_partition),
                                       convert_to_array(partition))
    print('NMI: {:.4f}'.format(nmi))


def non_lfr_runs(algorithms):
    # Karate Club graph
    karate_g = generate_karate_club_graph()
    pos = nx.spring_layout(karate_g)
    results = [alg(karate_g) for alg in algorithms]
    partitions = [r[0] for r in results]

    metrics = [(coverage(karate_g, r[1]), 
                performance(karate_g, r[1]),
                modularity(karate_g, r[1]),)
                for r in results]

    for idx in range(len(metrics)):
        print(
            f"The coverage obtained by {algorithms[idx].__name__} on the Karate Club graph was " +
            "%.4f" % metrics[idx][0])
        print(
            f"The performance obtained by {algorithms[idx].__name__} on the Karate Club graph was " +
            "%.4f" % metrics[idx][1])
        print(
            f"The final modularity obtained by {algorithms[idx].__name__} on the Karate Club graph was " +
            "%.4f" % metrics[idx][2])

    parallel_display(partitions, karate_g, pos)

    # simple Caveman graph
    caveman_g = generate_caveman_graph(cliques=4, size=6)
    pos = nx.spring_layout(caveman_g)
    results = [alg(caveman_g) for alg in algorithms]
    partitions = [r[0] for r in results]

    metrics = [(coverage(caveman_g, r[1]), 
                performance(caveman_g, r[1]),
                modularity(caveman_g, r[1]),)
                for r in results]

    for idx in range(len(metrics)):
        print(
            f"The coverage obtained by {algorithms[idx].__name__} on the Caveman graph was " +
            "%.4f" % metrics[idx][0])
        print(
            f"The performance obtained by {algorithms[idx].__name__} on the Caveman graph was " +
            "%.4f" % metrics[idx][1])
        print(
            f"The final modularity obtained by {algorithms[idx].__name__} on the Caveman graph was " +
            "%.4f" % metrics[idx][2])

    parallel_display(partitions, caveman_g, pos)



def main():
    print("Start process")
    
    algorithms = [clauset_newman_moore, louvain]    

    # small graphs
    non_lfr_runs(algorithms)


    # lfr benchmark graphs
    sizes = [250, 500, 600, 700, 800, 900, 1000, 1200, 2000, 2500, 2800, 3000]
    for n in sizes:
        G, target_partition, target_communities = genrate_lfr_graph(size=n)
        # visualize_graph(G)
        pos = nx.spring_layout(G)

        results = [alg(G) for alg in algorithms]
        partitions = [r[0] for r in results]

        metrics = [(coverage(G, r[1]), performance(G, r[1]), 
                    normalized_mutual_info_score(convert_to_array(target_partition), 
                                                 convert_to_array(r[0])
                                                ))
                    for r in results]
        
        for idx in range(len(metrics)):
            print(
                f"The coverage obtained by {algorithms[idx].__name__} was " + "%.4f" % metrics[idx][0])
            print(
                f"The performance obtained by {algorithms[idx].__name__} was " + "%.4f" % metrics[idx][1])
            print(
                f"The NMI score obtained by {algorithms[idx].__name__} was " + "%.4f" % metrics[idx][2])

        parallel_display(partitions, G, pos)

if __name__ == "__main__":
    main()


