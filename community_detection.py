import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities
from networkx.algorithms.community import girvan_newman

import matplotlib.pyplot as plt

from networkx.generators.community import LFR_benchmark_graph


# LFR benchmark graph generator example
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
    n = 250
    tau1 = 3
    tau2 = 1.5
    mu = 0.1
    G = LFR_benchmark_graph(n, tau1, tau2, mu, average_degree=5,
                            min_community=20, seed=10)
    print("generated")
    # get the communities from the node attributes of the graph
    communities = {frozenset(G.nodes[v]['community']) for v in G}
    print(f"LFR graph communities: {communities}")
    return G


def visualize_graph(G):
    options = {
        'node_color': 'black',
        'node_size': 100,
        'width': 3,
        }
    plt.subplot(121)
    nx.draw(G, **options)
    # plt.subplot(122)
    # nx.draw_shell(G, nlist=[range(5, 10), range(5)], with_labels=True, font_weight='bold')
    plt.show()

# apply Clauset-Newman-Moore greedy modularity maximization
def clauset_newman_moore(G):
    c = list(greedy_modularity_communities(G))
    print(sorted(c[0]))


def girvan_newman_2002(G):
    communities_generator = girvan_newman(G)
    top_level_communities = next(communities_generator)
    next_level_communities = next(communities_generator)
    print(sorted(map(sorted, next_level_communities)))


def main():
    print("start")
    G = genrate_lfr_graph()
    visualize_graph(G)
    clauset_newman_moore(G)
    girvan_newman_2002(G)

if __name__ == "__main__":
    main()


