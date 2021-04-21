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
from utilities import to_graph_like

if __name__ == "__main__":
    cs = list()
    for _ in range(3):
        for n_qubits in range(4, 12, 2):
            for gates_per_qubit in range(10, 25, 5):
                cs.append(zx.generate.CNOT_HAD_PHASE_circuit(qubits=n_qubits,
                                                             depth=n_qubits * gates_per_qubit,
                                                             clifford=False))

    N_GENS = 50
    N_MUTANTS = [10, 25, 50]
    INTERVAL = 2

    plt.rcParams.update({'font.size': 14})
    fig = plt.figure()
    ax = fig.add_subplot(111)

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

            best_scores, gen_scores, c_opt = GA_OPT.evolve(c_tr, n_mutants=n_mut, n_generations=N_GENS)

            final_score = best_scores[-1]
            for x in range(0, N_GENS, INTERVAL):
                if final_score < best_scores[x]:
                    improvement_after[x].append(1)
                else:
                    improvement_after[x].append(0)

        improvement_after_probs = {k: np.mean(v) for k, v in improvement_after.items()}
        xs = list(improvement_after_probs.keys())
        ys = list(improvement_after_probs.values())
        ax.scatter(xs, ys, label=str(n_mut))

    pdb.set_trace()
    ax.set_xlabel("Generation")
    ax.set_ylabel("Probability of Future Improvement")
    ax.set_title("Likelihood of Improvement Throughout GA")
    ax.legend(title="Population Size")
    plt.show()


    pdb.set_trace()
    print("HI")
