import time
import itertools
import numpy as np
import networkx as nx

from collections import defaultdict
from community import modularity

class GCM():

    def __init__(self, iterations=-1, pop_size=-1, alpha=0.4, theta=0.2, xi=0.5, beta=0.1,
                 fitness_func=modularity, verbose=False):
        self.iterations     = iterations
        self.pop_size       = pop_size
        self.alpha          = alpha
        self.theta          = theta
        self.xi             = xi
        self.beta           = beta
        self.fitness_func   = fitness_func
        self.verbose        = verbose
    
    def convert_to_sequence(self, partition):
        communities_dict = defaultdict(list)
        for node, c in partition.items():
            communities_dict[c].append(node)

        communities = [frozenset(c) for c in communities_dict.values()]
        return communities


    def initialization(self, G, nodes):
        population = {idx:list(nodes) for idx in range(self.pop_size)}
        for name, individual in population.items():
            for _ in range(int(self.pop_size*self.alpha)):
                random_node = np.random.randint(len(individual))
                while not G[random_node]:
                    random_node = np.random.randint(len(individual))

                population[name][random_node] = np.random.choice(
                    list(G[random_node]))
        
        return population
                

    def cross_over(self, chrom1, chrom2):
        child = np.array(chrom2)
        for _ in range(int(len(chrom1) * self.theta)):
            comm = np.random.choice(chrom1)
            vertices = np.where(chrom1==comm)[0]
            chrom1 = np.array(chrom1)
            child[vertices] = chrom1[vertices]

        return list(child)


    def mutation(self, chrom):
        for _ in range(int(len(chrom) * self.xi)):
            v1 = np.random.randint(len(chrom))
            v2 = np.random.randint(len(chrom))
            chrom[v1] = chrom[v2]
        return chrom


    def rank(self, G, population):
        return dict(sorted(population.items(),
                            key=lambda x: self.fitness_func(dict(enumerate(x[1])), G),
                            reverse=True)
                            )


    def get_next_generation(self, G, population, r=0.2):
        length = len(population)
        cut_point = int(self.beta * length)
    
        ranked_population = list(self.rank(G, population).values())

        beta_population = ranked_population[:cut_point].copy()
        rest_population = ranked_population.copy()
        np.random.shuffle(rest_population)

        new_population = []
        for i in range(0, len(rest_population)-1):
            c = self.cross_over(rest_population[i], rest_population[i+1])
            new_population.append(c)

        new_population.append(self.cross_over(rest_population[0], rest_population[-1]))
    
        for i in range(len(new_population)):
            new_population[i] = self.mutation(new_population[i])
        
        new_generation = (beta_population +
                          new_population)[:int(len(population)*(1-r))] + list(self.initialization(G, G.nodes()).values())[:int(len(population)*r)]
        return dict(enumerate(new_generation))


    def gcm(self, G):
        nodes = G.nodes()

        tmp_size = 0
        tmp_iter = 0
        if (len(nodes) / 10) < 100:
            tmp_size = 100
            tmp_iter = 250
        else:
            tmp_size = int(len(nodes) / 10)
            tmp_iter = int(len(nodes) / 3)


        self.iterations = tmp_iter if self.iterations == -1 else self.iterations
        self.pop_size = tmp_size if self.pop_size == -1 else self.pop_size


        start_time = time.time()

        population = self.initialization(G, nodes)
        for i in range(self.iterations):
            if self.verbose and (i % 20 == 0):
                print(f"Best Candidate: {population[0]}")
                print(f"Step: {i} out of {self.iterations}")
                print(
                    f"Current modularity: {modularity(dict(enumerate(population[0])), G)}")

            population = self.get_next_generation(G, population)
        
        run_time = time.time() - start_time
        if self.verbose:
            print(f"GCM ran in {run_time} seconds")
        
        partition = dict(enumerate(population[0]))
        return partition, self.convert_to_sequence(partition), run_time

