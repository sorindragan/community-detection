import json
import numpy as np

import matplotlib.pyplot as plt

COLORS = ["b", "r", "g", "k"]


def read_json(file_name="results/lfr.json"):
    results = None
    with open(file_name, "r") as f:
        results = json.load(f)
    return results


def get_nmi_values_for(sizes, results_dict, alg):
   return [(k2, results_dict[alg][k2]["nmi"])
           for k2 in results_dict[alg]
           ]


def plot_gcm_runtimes(sizes):
    result_dict4 = read_json(file_name="results/lfr_c2_u04.json")
    result_dict1 = read_json(file_name="results/lfr_c1_u01.json")

    fig, ax = plt.subplots(figsize=(19.20, 10.80))
    ax.title.set_text("Genetic algorithm run time")
    ax.set_ylim([0, 10000])
    loc = 2
    ax.set_xlim([sizes[0]-1, sizes[-1]+1])

    ax.plot(sizes, [result_dict1["GenCom"][k]["runtime"]
                    for k in result_dict1["GenCom"]], COLORS[0], label="GCM mu = 0.1", linewidth=3)
    ax.plot(sizes, [result_dict4["GenCom"][k]["runtime"]
                    for k in result_dict4["GenCom"]], COLORS[1], label="GCM mu = 0.4", linewidth=3)

    plt.xlabel('Seconds')
    plt.ylabel("Graph Size")
    leg = ax.legend(loc=loc, prop={'size': 20})
    plt.show()


def plot_nodes_edges(sizes, edges1, edges2):
    fig, ax = plt.subplots(figsize=(19.20, 10.80))
    ax.title.set_text("Nodes / Edges number")
    ax.set_ylim([0, 60000])
    loc = 2
    ax.set_xlim([sizes[0]-1, sizes[-1]+1])

    ax.plot(sizes, edges1, COLORS[1], label="mu = 0.1", linewidth=2.5)
    ax.plot(sizes, edges2, COLORS[2], label="mu = 0.4", linewidth=2.5)

    plt.xlabel('Number of nodes')
    plt.ylabel("Number of edges")
    leg = ax.legend(loc=loc, prop={'size': 20})
    plt.show()

def plot_result(sizes, results_dict, result_type="Coverage"):
    fig, ax = plt.subplots(figsize=(19.20, 10.80))
    ax.title.set_text(result_type)
    if result_type == "Runtime":
        ax.set_ylim([0, 15])
        loc = 2
    else:    
        ax.set_ylim([0, 1.05])
        loc = 3
    ax.set_xlim([sizes[0]-1, sizes[-1]+1])

    for idx, key in enumerate(results_dict):
        coverages = [results_dict[key][k2][result_type.lower()]
                     for k2 in results_dict[key]
                    ]
        ax.plot(sizes, coverages, COLORS[idx], label=key, linewidth=3)
    
    plt.xlabel('Size of graph')
    plt.ylabel(result_type)
    leg = ax.legend(loc=loc, prop={'size': 20})
    plt.show()


def main():
    result_dict = read_json(file_name="results/lfr_c1_u01.json")

    sizes = [250, 500, 1000, 1500, 2000, 2500, 3000]

    edges_u01 = [4480, 8886, 17473, 26150, 35044, 43711, 52810]
    edges_u04 = [4980, 9842, 19260, 28727, 38558, 48607, 58565]
    plot_nodes_edges(sizes, edges_u01, edges_u04)
    
    for alg in ["CNM", "Louvain", "RenEEL", "GenCom"]:
        print(get_nmi_values_for(sizes, result_dict, alg))

    for rtype in ["Coverage", "Performance", "Runtime", "Nmi"]:
        plot_result(sizes, result_dict, result_type=rtype)
    
    plot_gcm_runtimes(sizes)


if __name__ == "__main__":
    main()
