import networkx as nx
import numpy as np

from community import modularity



def generate_graph():
    G = nx.relaxed_caveman_graph(5, 4, 0.6)
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
            

def cross_over(chrom_src, chrom_dest, theta=0.2):
    for _ in range(int(len(chrom_src) * 0.2)):
        comm = np.random.choice(chrom_src)
        vertices = np.where(chrom_src==comm)[0]
        chrom_src = np.array(chrom_src)
        chrom_dest = np.array(chrom_dest)   
        chrom_dest[vertices] = chrom_src[vertices]
    return list(chrom_dest)


def mutation(chrom, xi=0.5):
    for _ in range(int(len(chrom)*xi)):
        v1 = np.random.randint(len(chrom))
        v2 = np.random.randint(len(chrom))
        chrom[v1] = chrom[v2]
    return chrom


def rank(G, population, fitness_func=modularity):
    return dict(sorted(population.items(),
                          key=lambda x: fitness_func(dict(enumerate(x[1])), G)),
                          reverse=True
                          )


def generation(G, population, p=0.5, beta=0.1):
    length = len(population)
    ranked_population = rank(G, population)
    top_population = ranked_population[:p*length]
    
    


def main():
    G = generate_graph()
    nodes = G.nodes()
    population = initialization(G, nodes)
    # print(population)
    # cross_over(population[0], population[1])
    # mutation(population[0])
    rank(G, population)

if __name__ == "__main__":
    main()

