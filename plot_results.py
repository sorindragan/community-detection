import json
import numpy as np

import matplotlib.pyplot as plt

COLORS = ["-b", "-r", "-g", "-k"]

def read_json():
    results = None
    with open("results/small.json", "r") as f:
        results = json.load(f)
    return results

def plot_result(sizes, results_dict, result_type="Coverage"):
    fig, ax = plt.subplots(figsize=(19.20, 10.80))
    ax.title.set_text(result_type)
    ax.set_ylim([0, 1])
    ax.set_xlim([sizes[-1]-1, sizes[0]+1])

    for idx, key in enumerate(results_dict):
        coverages = [results_dict[key][k2]["NMI"]
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
    # "Coverage", "Performance", "NMI", "Runtime":
    rtype = "NMI"
    plot_result(sizes, result_dict, result_type=rtype)


if __name__ == "__main__":
    main()
