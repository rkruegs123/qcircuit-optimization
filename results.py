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
from os.path import isfile, join
import os
import time

import sys
sys.path.append('../pyzx')
import pyzx as zx

from ga_optimizers import default_score, GeneticOptimizer, rand_pivot, rand_lc, do_nothing, Mutant
from utilities import to_graph_like, c_score, g_score
from anneal import pivot_anneal
from baseline_optimizers import TketFullPeephole, QiskitTranspiler




n_qubits = list(range(4, 16, 2))
gates_per_qubit = 40


def sa_with_restarts(g, n_restarts=3, n_iters=2500, quiet=True):
    best_g = g.copy()
    best_score = g_score(best_g)
    for i in range(n_restarts):
        g_sa, _ = pivot_anneal(g, iters=n_iters, quiet=quiet)
        trial_score = g_score(g_sa)
        if trial_score < best_score or i == 0:
            best_score = trial_score
            best_g = g_sa.copy()
    return best_g


def gen_circuits():
    basedir = f"circuits/core_results/{gates_per_qubit}-gates-per-qb"

    for qb in n_qubits:
        odir = join(basedir, f"{qb}-qb")
        if os.path.exists(odir):
            sys.exit("ERROR: output directory already exists")
        os.makedirs(odir)
        for i in range(10):
            c = zx.generate.CNOT_HAD_PHASE_circuit(qubits=qb,
                                                   depth=qb * gates_per_qubit,
                                                   clifford=False)
            fname = f"{i}.qasm"
            with open(join(odir, fname), 'w') as f:
                f.write(c.to_qasm())


if __name__ == "__main__":

    # n_qubits = list(range(4, 8, 2))
    # n_gates_per_qubit = [10, 25]
    # n_gates_per_qubit = [15]

    METHOD = "GA" # "SA"
    SIMPLIFY = "TR"

    N_SA_RESTARTS = 3
    SA_ITERS = 2500

    ACTIONS = [rand_pivot, rand_lc, do_nothing]
    GA_OPT = GeneticOptimizer(ACTIONS, quiet=False)
    N_MUT = 20
    N_GENS = 75

    QISKIT_OPT = QiskitTranspiler(opt_level=2)
    TKET_OPT = TketFullPeephole()

    # gen_circuits()

    times = list()
    avg_reductions = dict()
    for qb in n_qubits:
        cdir = f"circuits/core_results/{gates_per_qubit}-gates-per-qb/{qb}-qb"
        nq_reductions = list()
        for i in range(10):
            c = zx.Circuit.load(f"{cdir}/{i}.qasm")
            init_score = c_score(c)
            g = c.to_graph()
            g_tmp = g.copy()

            if METHOD == "FR":
                g_opt = g_tmp.copy()

                start = time.time()
                zx.full_reduce(g_opt)
                c_opt = zx.extract_circuit(g_opt.copy()).to_basic_gates()
                c_opt = zx.basic_optimization(c_opt)
                end = time.time()

                opt_score = c_score(c_opt)
            elif METHOD == "TR":
                g_opt = g_tmp.copy()

                start = time.time()
                zx.teleport_reduce(g_opt)
                c_opt = zx.Circuit.from_graph(g_opt.copy()).to_basic_gates()
                c_opt = zx.basic_optimization(c_opt)
                end = time.time()

                opt_score = c_score(c_opt)
                # g_opt = c_opt.to_graph()
            elif METHOD == "qiskit":
                start = time.time()
                c_opt = QISKIT_OPT._optimize(c.copy())
                end = time.time()

                opt_score = c_score(c_opt)
            elif METHOD == "tket":
                start = time.time()
                c_opt = TKET_OPT._optimize(c.copy())
                end = time.time()

                opt_score = c_score(c_opt)
            elif METHOD == "SA" or METHOD == "GA":
                # obtain the initial simplified ZX-diagram
                start = time.time()
                if SIMPLIFY == "TR":
                    zx.teleport_reduce(g_tmp)
                    c_tr = zx.Circuit.from_graph(g_tmp.copy()).to_basic_gates()
                    c_tr = zx.basic_optimization(c_tr)
                    g_simp = c_tr.to_graph()
                    to_graph_like(g_simp)
                elif SIMPLIFY == "FR":
                    g_simp = g_tmp.copy()
                    zx.full_reduce(g_simp)
                else:
                    raise RuntimeError(f"Invalid SIMPLIFY: {SIMPLIFY}")

                # feed the simplified ZX-diagram to search
                if METHOD == "SA":
                    # g_opt, _ = pivot_anneal(g_simp, iters=SA_ITERS, quiet=False)
                    g_opt = sa_with_restarts(g_simp, n_restarts=N_SA_RESTARTS, n_iters=SA_ITERS, quiet=False)
                else: # METHOD == "GA":
                    _, _, c_opt = GA_OPT.evolve(g_simp, n_mutants=N_MUT, n_generations=N_GENS)
                    g_opt = c_opt.to_graph()
                end = time.time()
                opt_score = g_score(g_opt)
            else:
                raise RuntimeError(f"Invalid method: {METHOD}")
            reduction = (init_score - opt_score) / init_score * 100
            nq_reductions.append(reduction)
            times.append(end - start)
        avg_reductions[qb] = np.mean(nq_reductions)
    print(f"Method: {METHOD}")
    print(f"Simplify: {SIMPLIFY}")
    print(f"Average reductions: {avg_reductions}")
    print(f"Average time elapsed: {np.mean(times)}")

    """
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
    """
