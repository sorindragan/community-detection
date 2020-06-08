import numpy as np

# paper parameters
# cdef float alpha = 0.4
# cdef float theta = 0.2
# cdef float xi = 0.5
# cdef float beta = 0.1
# cdef float rr = 0

# candidate1
# cdef float alpha = 0.3
# cdef float theta = 0.5
# cdef float xi = 0.05
# cdef float beta = 0.02
# cdef float rr = 0.5

# candidate2
cdef float alpha = 0.1
cdef float theta = 0.4
cdef float xi = 0.05
cdef float beta = 0.4
cdef float rr = 0.5


cdef list initialization(G, int pop_size):
    global alpha
    cdef list population = [np.array(G.nodes()) for _ in range(pop_size)]
    cdef int length = len(population[0])
    cdef int random_node
    for individual in population:
        for _ in range(int(pop_size * alpha)):
            random_node = np.random.randint(length)
            while not G[random_node]:
                random_node = np.random.randint(length)
            for neighbour in G[random_node]:
                individual[neighbour] = random_node
    return population
            

def cross_over(chrom1, chrom2):
    global theta
    cdef int chrom_length = len(chrom1)
    child = np.array(chrom2)
    for _ in range(int(chrom_length * theta)):
        comm = np.random.choice(chrom1)
        vertices = np.where(chrom1==comm)[0]
        child[vertices] = chrom1[vertices]

    return child


def mutation(chrom):
    global xi
    cdef int chrom_length = len(chrom)
    cdef int v1
    cdef int v2
    for _ in range(int(chrom_length * xi)):
        v1 = np.random.randint(chrom_length)
        v2 = np.random.randint(chrom_length)
        chrom[v1] = chrom[v2]
    return chrom


cdef list rank(G, population, fitness_func):
    return sorted(population, 
                  key=lambda x: fitness_func(dict(enumerate(x)), G),
                  reverse=True)
                       

cdef list get_next_generation(G, population, fitness_func):
    global rr
    global beta
    cdef int length = len(population)
    cdef int cut_point = int(beta * length)

    cdef list ranked_population = rank(G, population, fitness_func)

    cdef list beta_population = ranked_population[:cut_point]
    cdef list rest_population = ranked_population
    np.random.shuffle(rest_population)

    cdef list new_population = length * [0]
    cdef int i
    for i in range(length-1):
        new_population[i] = cross_over(rest_population[i], rest_population[i+1])
        
    new_population[length-1] = cross_over(rest_population[0], rest_population[length-1])
    cdef int j
    for j in range(length):
        new_population[j] = mutation(new_population[j])
    
    cdef list new_generation = (beta_population + new_population)[:int(length*(1-rr))+1] + initialization(G, int(length*rr))
    return new_generation


def process(G, fitness_func):
    cdef int length = len(G.nodes())
    cdef int pop_size
    cdef int iterations
    cdef list population
    
    if (length / 10) <= 100:
        pop_size = 200
        iterations = 350
    else:
        pop_size = 300
        iterations = 700

    population = initialization(G, pop_size)
    cdef int i
    cdef float max_fitness
    cdef float curr_fitness = 0
    cdef int premature_stop = 0
    for i in range(iterations):
        population = get_next_generation(G, population, fitness_func)
        if i % 10 == 0:
            max_fitness = fitness_func(dict(enumerate(population[0])), G)
            if max_fitness == curr_fitness:
                premature_stop += 1
            else:
                premature_stop = 0
            curr_fitness = max_fitness
            # print(i)
            # print(max_fitness)
        if premature_stop == 10:
            break
    
    population = rank(G, population, fitness_func)
    return dict(enumerate(population[0]))
