import networkx as nx
import numpy as np
import itertools
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from networkx.generators.community import LFR_benchmark_graph
from community import modularity



def generate_graph():
    # G = nx.relaxed_caveman_graph(5, 4, 0.6)
    # G = nx.connected_caveman_graph(5, 4)
    # G = nx.karate_club_graph()
    params = {"n": 250, "tau1": 2, "tau2": 1.1,
         "mu": 0.1, "min_degree": 20, "max_degree": 50}

    G = LFR_benchmark_graph(params["n"], params["tau1"], params["tau2"], params["mu"], 
                            min_degree=params["min_degree"],
                            max_degree=params["max_degree"],
                            max_iters=5000, seed = 10,
                            )
    return G


def initialization(G, nodes, pop_size=42, alpha=0.4):
    population = {idx:list(nodes) for idx in range(pop_size)}
    for name, individual in population.items():
        for _ in range(int(pop_size*alpha)):
            random_node = np.random.randint(len(individual))
            while not G[random_node]:
                random_node = np.random.randint(len(individual))

            population[name][random_node] = np.random.choice(
                list(G[random_node]))
    
    return population
            

def cross_over(chrom1, chrom2, theta=0.2):
    child1 = np.array(chrom1) 
    child2 = np.array(chrom2)
    for _ in range(int(len(chrom1) * 0.2)):
        comm = np.random.choice(chrom1)
        vertices = np.where(chrom1==comm)[0]
        chrom1 = np.array(chrom1)
        child2[vertices] = chrom1[vertices]
    
    # for _ in range(int(len(chrom2) * 0.2)):
    #     comm = np.random.choice(chrom2)
    #     vertices = np.where(chrom2 == comm)[0]
    #     chrom2 = np.array(chrom2)
    #     child1[vertices] = chrom2[vertices]


    return list(child2)#, list(child1)


def mutation(chrom, xi=0.5):
    for _ in range(int(len(chrom)*xi)):
        v1 = np.random.randint(len(chrom))
        v2 = np.random.randint(len(chrom))
        chrom[v1] = chrom[v2]
    return chrom


def rank(G, population, fitness_func=modularity):
    return dict(sorted(population.items(),
                          key=lambda x: fitness_func(dict(enumerate(x[1])), G),
                          reverse=True)
                          )


def get_next_generation(G, population, beta=0.1):
    length = len(population)
    cut_point = int(beta*length)
   
    ranked_population = list(rank(G, population).values())
    # print(modularity(dict(enumerate(ranked_population[0])), G))
    # print(modularity(dict(enumerate(ranked_population[1])), G))


    beta_population = ranked_population[:cut_point].copy()
    rest_population = ranked_population.copy()


    new_population = []
    for i in range(0, len(rest_population)-1):
        c1 = cross_over(rest_population[i], rest_population[i+1])
        new_population.append(c1)
        # new_population.append(c2)
    new_population.append(cross_over(rest_population[0], rest_population[-1]))
  
    for i in range(len(new_population)):
        new_population[i] = mutation(new_population[i])
    
    return dict(enumerate((beta_population + new_population)[:len(population)]))


def main():
    G = generate_graph()
    nodes = G.nodes()
    population = initialization(G, nodes, pop_size=200)
    old_population = []
    for i in range(100):
        if i % 10 == 0:
            print("PULA")
            print(population[0])
            print([idx if old_population[idx] == population[0][idx] else None for idx in range(len(population))])            old_population = population[0]
            print(f"SHITTY STEPS: {i}")
            partition = dict(enumerate(population[0]))
            print(f"THIS IS A PRETTY THING: {modularity(partition, G)}")

        population = get_next_generation(G, population)
    
    
    partition = dict(enumerate(population[0]))
    print(f"THIS IS A PRETTY THING: {modularity(partition, G)}")
    pos = nx.spring_layout(G)
    cmap = cm.get_cmap('viridis', max(partition.values()) + 1)
    nx.draw_networkx_nodes(G, pos, partition.keys(), node_size=40,
                           cmap=cmap, node_color=list(partition.values()))
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    plt.show()

if __name__ == "__main__":
    main()

