import random
import math
from tqdm import tqdm
import matplotlib.pyplot as plt
import itertools
import pdb
import numpy as np
import networkx as nx
import seaborn as sns
import pandas as pd

import sys
sys.path.append('../pyzx')
import pyzx as zx

from ga_optimizers import default_score, GeneticOptimizer, rand_pivot, rand_lc, do_nothing, Mutant
from utilities import to_graph_like, c_score, g_score



def legend_without_duplicate_labels(ax, title=""):
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    ax.legend(*zip(*unique), title=title)


if __name__ == "__main__":
    cs = list()
    for _ in range(3):
        for n_qubits in range(4, 12, 2):
            for gates_per_qubit in range(10, 25, 5):
    # for _ in range(10):
        # for n_qubits in range(8, 10, 2):
            # for gates_per_qubit in range(10, 15, 5):
                cs.append(zx.generate.CNOT_HAD_PHASE_circuit(qubits=n_qubits,
                                                               depth=n_qubits * gates_per_qubit,
                                                               clifford=False))
        # cs.append(zx.generate.CNOT_HAD_PHASE_circuit(qubits=7,
                                                     # depth=100,
                                                     # clifford=False))


    N_GENS = 100
    N_MUTANTS = [10, 20, 40]
    colors = ['red', 'green', 'blue']
    colors = {n_mut: c for (n_mut, c) in zip(N_MUTANTS, colors)}
    INTERVAL = 4

    plt.rcParams.update({'font.size': 14})
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)

    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)

    REDUCE_METHOD = "TR"
    ACTIONS = [rand_pivot, rand_lc, do_nothing]
    GA_OPT = GeneticOptimizer(ACTIONS, quiet=True)

    for n_mut in N_MUTANTS:
        improvement_after = { x: list() for x in range(0, N_GENS, INTERVAL) }

        for c in tqdm(cs, desc="Circuits..."):
            g = c.to_graph()

            g_tmp = g.copy()

            if REDUCE_METHOD == "TR":
                zx.teleport_reduce(g_tmp)
                c_tr = zx.Circuit.from_graph(g_tmp.copy()).to_basic_gates()
                c_tr = zx.basic_optimization(c_tr)
                g_simp = c_tr.to_graph()
            elif REDUCE_METHOD == "FR":
                zx.full_reduce(g_tmp)
                g_simp = g_tmp.copy()
            else:
                raise RuntimeError(f"Invalid REDUCE_METHOD: {REDUCE_METHOD}")

            to_graph_like(g_simp)
            orig_score = c_score(c_tr)

            best_scores, gen_scores, c_opt = GA_OPT.evolve(g_simp, n_mutants=n_mut, n_generations=N_GENS)
            final_score = c_score(c_opt) # best_scores[-1]
            reduction = (orig_score - final_score) / orig_score * 100
            reductions = [(orig_score - best_score) / orig_score * 100 for best_score in best_scores]
            ax2.plot(list(range(len(reductions))), reductions, c=colors[n_mut], label=str(n_mut))
            for x in range(0, N_GENS, INTERVAL):
                if final_score < best_scores[x]:
                    improvement_after[x].append(1)
                else:
                    improvement_after[x].append(0)

        improvement_after_probs = {k: np.mean(v) for k, v in improvement_after.items()}
        xs = list(improvement_after_probs.keys())
        ys = list(improvement_after_probs.values())
        ax1.scatter(xs, ys, label=str(n_mut), c=colors[n_mut])

    pdb.set_trace()
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Probability of Future Improvement")
    ax1.set_title("Likelihood of Improvement Throughout GA")
    ax1.legend(title="Population Size")

    ax2.set_xlabel("Generation")
    ax2.set_ylabel("Reduction (%)")
    ax2.set_title(f"Complexity Reduction over {N_GENS} Generations")
    legend_without_duplicate_labels(ax2, title="Population Size")
    # ax2.legend(title="Population Size")

    plt.show()


    pdb.set_trace()
    print("HI")
