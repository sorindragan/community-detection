import os
import sys
import time
import json
import itertools
import networkx as nx
import matplotlib.cm as cm
import matplotlib.pyplot as plt

import community as community_louvain
from community import modularity as community_modularity

from collections import defaultdict
from sklearn.metrics.cluster import normalized_mutual_info_score
from networkx.algorithms.community import greedy_modularity_communities
from networkx.algorithms.community import modularity
from networkx.generators.community import LFR_benchmark_graph
from networkx.algorithms.community.quality import coverage, performance

from args_parser import parse

NAMES = ["CNM", "Louvain", "RenEEL", "GenCom"]
RESULTS_S = {"CNM": {},"Louvain": {},"RenEEL": {}, "GenCom": {}}
RESULTS_LFR = {"CNM": {}, "Louvain": {}, "RenEEL": {}, "GenCom": {}}
VERBOSE = False

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
def genrate_lfr_graph(size=250, mu=0.1):
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

    # params = {"n":size, "tau1":2, "tau2":1.1, "mu":0.1, "min_degree":20, "max_degree":50}
    params = {"n":size, "tau1":2, "tau2":1.1, "mu":0.4, "min_degree":20, "max_degree":50}
    

    G = None
    while G is None:
        try:
            G = LFR_benchmark_graph(params["n"], params["tau1"], params["tau2"], mu,
                                    min_degree=params["min_degree"],
                                    max_degree=params["max_degree"],
                                    max_iters=5000, seed=10,
                                    )
        except:
            pass
    
    if VERBOSE:
        print("Generation Completed")

    comm_list = set([frozenset(G.nodes[v]['community']) for v in G])
    comm_dict = {comm: idx for idx, comm in enumerate(comm_list)}

    partition = {v:comm_dict[frozenset(G.nodes[v]['community'])] for v in G}
    communities = convert_to_sequence(partition)
    
    return G, partition, communities

def visualize_communities(partition, G, pos, show=True):
    cmap = cm.get_cmap('viridis', max(partition.values()) + 1)
    nx.draw_networkx_nodes(G, pos, partition.keys(), node_size=40,
                        cmap=cmap, node_color=list(partition.values()))
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    if show:
        plt.show()


def parallel_display(algs, partitions, G, pos):
    fig = plt.figure(figsize=(19.20, 10.80))
    length = len(partitions)
    for i in range(length):
        ax = fig.add_subplot(2, 2, i+1)
        ax.title.set_text(algs[i].__name__)
        visualize_communities(partitions[i], G, pos, show=False)
    
    # plt.show()
    plt.savefig(f"results/{G.name}--N={len(G.nodes())}.png")

def clauset_newman_moore(G):
    partition = {}
    start_time = time.time()
    communities = greedy_modularity_communities(G)
    run_time = time.time() - start_time
    if VERBOSE:
        print(f"Clauset Newman Moore ran in {run_time} seconds")
    for idx, c in enumerate(communities):
        for node in c:
            partition[node] = idx
    return partition, communities, run_time

def louvain(G):
    start_time = time.time()
    partition = community_louvain.best_partition(G)
    run_time = time.time() - start_time
    if VERBOSE:
        print(f"Louvain ran in {run_time} seconds")
    communities = convert_to_sequence(partition)
    return partition, communities, run_time


def reneel(G):
    with open("RenEEL-Modularity-master/karate.txt", "w") as f:
        for e1, e2 in G.edges():
            f.write(str(e1+1) + " " + str(e2+1) + "\n")

    start_time = time.time()
    os.system('cd RenEEL-Modularity-master/;make')
    run_time = time.time() - start_time
    if VERBOSE:
        print(f"RenEEL-Modularity-based ran in {run_time} seconds")

    with open("RenEEL-Modularity-master/partition.txt", "r") as f:
        reneel_partition = dict(
            enumerate([int(e.strip())-1 for e in list(f.readlines())]))
    
    reneel_communities = convert_to_sequence(reneel_partition)

    os.system('cd RenEEL-Modularity-master/;make clean')

    return reneel_partition, reneel_communities, run_time

def gcm(G):
    os.system("cd cython_gcm/;python setup.py build_ext --inplace")
    from cython_gcm import gcm
    start_time = time.time()
    partition = gcm.process(G, community_modularity)
    run_time = time.time() - start_time
    if VERBOSE:
        print(f"GCM ran in {run_time} seconds")
    communities = convert_to_sequence(partition)
    return partition, communities, run_time


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

    runtimes = [r[2] for r in results]

    for idx in range(len(metrics)):
        RESULTS_S[NAMES[idx]]["Karate"] = {
                "coverage": metrics[idx][0],
                "performance": metrics[idx][1],
                "modularity": metrics[idx][2],
                "runtime": runtimes[idx],
            }

        if VERBOSE:
            print(
                f"The coverage obtained by {algorithms[idx].__name__} on the Karate Club graph was " +
                "%.4f" % metrics[idx][0])
            print(
                f"The performance obtained by {algorithms[idx].__name__} on the Karate Club graph was " +
                "%.4f" % metrics[idx][1])
            print(
                f"The final modularity obtained by {algorithms[idx].__name__} on the Karate Club graph was " +
                "%.4f" % metrics[idx][2])
            print("========================================================")


    parallel_display(algorithms, partitions, karate_g, pos)

    # simple Caveman graph 4 6
    caveman_g = generate_caveman_graph(cliques=4, size=6)
    pos = nx.spring_layout(caveman_g)
    results = [alg(caveman_g) for alg in algorithms]
    partitions = [r[0] for r in results]

    metrics = [(coverage(caveman_g, r[1]), 
                performance(caveman_g, r[1]),
                modularity(caveman_g, r[1]),)
                for r in results]

    runtimes = [r[2] for r in results]

    for idx in range(len(metrics)):
        RESULTS_S[NAMES[idx]]["Caveman46"] = {
                "coverage": metrics[idx][0],
                "performance": metrics[idx][1],
                "modularity": metrics[idx][2],
                "runtime": runtimes[idx],
            }
        if VERBOSE:
            print(
                f"The coverage obtained by {algorithms[idx].__name__} on the Caveman46 graph was " +
                "%.4f" % metrics[idx][0])
            print(
                f"The performance obtained by {algorithms[idx].__name__} on the Caveman46 graph was " +
                "%.4f" % metrics[idx][1])
            print(
                f"The final modularity obtained by {algorithms[idx].__name__} on the Caveman46 graph was " +
                "%.4f" % metrics[idx][2])
            print("========================================================")

    parallel_display(algorithms, partitions, caveman_g, pos)

    # simple Caveman graph 7 3
    caveman_g = generate_caveman_graph(cliques=7, size=3)
    pos = nx.spring_layout(caveman_g)
    results = [alg(caveman_g) for alg in algorithms]
    partitions = [r[0] for r in results]

    metrics = [(coverage(caveman_g, r[1]),
                performance(caveman_g, r[1]),
                modularity(caveman_g, r[1]),)
               for r in results]

    runtimes = [r[2] for r in results]

    for idx in range(len(metrics)):
        RESULTS_S[NAMES[idx]]["Caveman73"] = {
                "coverage": metrics[idx][0],
                "performance": metrics[idx][1],
                "modularity": metrics[idx][2],
                "runtime": runtimes[idx],
            }
        if VERBOSE:
            print(
                f"The coverage obtained by {algorithms[idx].__name__} on the Caveman73 graph was " +
                "%.4f" % metrics[idx][0])
            print(
                f"The performance obtained by {algorithms[idx].__name__} on the Caveman73 graph was " +
                "%.4f" % metrics[idx][1])
            print(
                f"The final modularity obtained by {algorithms[idx].__name__} on the Caveman73 graph was " +
                "%.4f" % metrics[idx][2])
            print("========================================================")

    parallel_display(algorithms, partitions, caveman_g, pos)


def main():
    global VERBOSE
    h, v, w, mu, mu_val = parse(' '.join(sys.argv[1:]))
    
    if h:
        print("- Use -v to activate Verbose")
        print("- Use -w to exclude the genetic algorithm from the run")
        print("- Use -mu value to set the value of mu in the graph generators; value should be in the range (0, 1)")
        return

    if v:
        VERBOSE = True
    else:
        VERBOSE = False

    if w:
        algorithms = [clauset_newman_moore, louvain, reneel]
    else:
        algorithms = [clauset_newman_moore, louvain, reneel, gcm]
    

    if VERBOSE:
        print("Start process")

    # small graphs
    non_lfr_runs(algorithms)

    with open('results/small_c1_test.json', 'w') as fs:
        json.dump(RESULTS_S, fs)
    
    

    # lfr benchmark graphs
    sizes = [250, 500, 1000, 1500, 2000, 2500, 3000]
    for n in sizes:
        G, target_partition, target_communities = genrate_lfr_graph(size=n, mu=mu_val)
        nodes_no = n
        edges_no = G.number_of_edges()
        avg_degree = sum([G.degree[i] for i in range(n)]) / nodes_no
        print("========================================================")
        print(nodes_no, edges_no, avg_degree)
        print("========================================================")

        pos = nx.spring_layout(G)

        results = [alg(G) for alg in algorithms]
        partitions = [r[0] for r in results]

        metrics = [(coverage(G, r[1]), performance(G, r[1]), 
                    normalized_mutual_info_score(convert_to_array(target_partition), 
                                                 convert_to_array(r[0])
                                                ))
                    for r in results]
        
        runtimes = [r[2] for r in results]
          
        for idx in range(len(metrics)):
            RESULTS_LFR[NAMES[idx]][n] = {
                "coverage": metrics[idx][0],
                "performance": metrics[idx][1],
                "nmi": metrics[idx][2],
                "runtime": runtimes[idx],
            }
            if VERBOSE:
                print(
                    f"The coverage obtained by {algorithms[idx].__name__} was " + "%.4f" % metrics[idx][0])
                print(
                    f"The performance obtained by {algorithms[idx].__name__} was " + "%.4f" % metrics[idx][1])
                print(
                    f"The NMI score obtained by {algorithms[idx].__name__} was " + "%.4f" % metrics[idx][2])
                print("========================================================")

        parallel_display(algorithms, partitions, G, pos)
    
    with open('results/lfr_c1_test.json', 'w') as fb:
        json.dump(RESULTS_LFR, fb)

if __name__ == "__main__":
    main()
