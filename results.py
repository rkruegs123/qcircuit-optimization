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
from anneal import pivot_anneal


def sa_with_restarts(g, n_restarts=3, n_iters=2500, quiet=True):
    best_g = g.copy()
    best_score = g_score(best_g)
    for i in range(n_restarts):
        g_sa, _ = pivot_anneal(g, iters=2500, quiet=quiet)
        trial_score = g_score(g_sa)
        if trial_score < best_score or i == 0:
            best_score = trial_score
            best_g = g_sa.copy()
    return best_g


if __name__ == "__main__":
    n_qubits = list(range(4, 16, 2))
    # n_qubits = list(range(4, 8, 2))
    # n_gates_per_qubit = [10, 25]
    n_gates_per_qubit = [15]

    METHOD = "SA"
    SIMPLIFY = "TR"

    N_SA_RESTARTS = 1
    SA_ITERS = 2500

    ACTIONS = [rand_pivot, rand_lc, do_nothing]
    GA_OPT = GeneticOptimizer(ACTIONS, quiet=True)
    N_MUT = 20
    N_GENS = 40

    plt.rcParams.update({'font.size': 14})
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)

    for gates_per_qubit in tqdm(n_gates_per_qubit, desc="Gates/qubit..."):
        avg_reductions = list()

        for nq in tqdm(n_qubits, desc="Qubits..."):
            nq_reductions = list()

            for _ in tqdm(range(10), desc="Trials"):

                c = zx.generate.CNOT_HAD_PHASE_circuit(qubits=nq,
                                                       depth=nq * gates_per_qubit,
                                                       clifford=False)

                g = c.to_graph()
                g_tmp = g.copy()

                if SIMPLIFY == "TR":
                    zx.teleport_reduce(g_tmp)
                    c_tr = zx.Circuit.from_graph(g_tmp.copy()).to_basic_gates()
                    c_tr = zx.basic_optimization(c_tr)

                    orig_score = c_score(c_tr)
                    g_simp = c_tr.to_graph()
                elif SIMPLIFY == "FR":
                    zx.full_reduce(g_tmp)
                    c_fr = zx.extract_circuit(g_tmp.copy()).to_basic_gates()
                    c_fr = zx.basic_optimization(c_fr)

                    orig_score = c_score(c_fr)
                    g_simp = g_tmp.copy()
                else:
                    raise RuntimeError(f"Invalid SIMPLIFY: {SIMPLIFY}")
                to_graph_like(g_simp)

                if METHOD == "SA":
                    # g_opt = sa_with_restarts(g_simp, n_restarts=N_SA_RESTARTS, n_iters=SA_ITERS)
                    g_opt, _ = pivot_anneal(g_simp, iters=2500, quiet=False)
                elif METHOD == "GA":
                    _, _, c_opt = GA_OPT.evolve(g_simp, n_mutants=N_MUT, n_generations=N_GENS)
                    g_opt = c_opt.to_graph()
                else:
                    raise RuntimeError(f"Invalid method: {method}")

                trial_score = g_score(g_opt)
                reduction = (orig_score - trial_score) / orig_score * 100
                nq_reductions.append(reduction)
            avg_reductions.append(np.mean(nq_reductions))
        ax1.plot(n_qubits, avg_reductions, label=str(gates_per_qubit))

    ax1.set_xlabel("Qubits")
    ax1.set_ylabel("Avg. Reduction")
    ax1.legend(title="Gates per Qubit")
    plt.show()

    pdb.set_trace()
    print("HI")
