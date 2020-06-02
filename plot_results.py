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
   return [(k2, results_dict[alg][k2]["NMI"])
           for k2 in results_dict[alg]
           ]


def plot_result(sizes, results_dict, result_type="Coverage"):
    fig, ax = plt.subplots(figsize=(19.20, 10.80))
    ax.title.set_text(result_type)
    if result_type == "Runtime":
        ax.set_ylim([0, 11])
    else:    
        ax.set_ylim([0, 1.05])
    ax.set_xlim([sizes[0]-1, sizes[-1]+1])

    for idx, key in enumerate(results_dict):
        coverages = [results_dict[key][k2][result_type.lower()]
                     for k2 in results_dict[key]
                    ]
        ax.plot(sizes, coverages, COLORS[idx], label=key)
    
    plt.xlabel('Size / Number of nodes)')
    plt.ylabel(result_type)
    leg = ax.legend()
    plt.show()


def main():
    result_dict = read_json()
    sizes = [250, 400, 600, 800, 1000, 1500, 2000, 2500, 3000]
    # print(get_nmi_values_for(sizes, result_dict, "GenCom"))
    for rtype in ["Coverage", "Performance", "Runtime"]:
        plot_result(sizes, result_dict, result_type=rtype)


if __name__ == "__main__":
    main()
