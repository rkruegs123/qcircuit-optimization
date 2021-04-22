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
from os import listdir
import time

import sys
sys.path.append('../pyzx')
import pyzx as zx

from ga_optimizers import default_score, GeneticOptimizer, rand_pivot, rand_lc, do_nothing, Mutant
from utilities import to_graph_like, c_score, g_score
from anneal import pivot_anneal
from baseline_optimizers import TketFullPeephole, QiskitTranspiler



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





if __name__ == "__main__":



    METHOD = "TR" # "SA"
    SIMPLIFY = "TR"

    N_SA_RESTARTS = 3
    SA_ITERS = 2500

    ACTIONS = [rand_pivot, rand_lc, do_nothing]
    GA_OPT = GeneticOptimizer(ACTIONS, quiet=False)
    N_MUT = 20
    N_GENS = 75

    QISKIT_OPT = QiskitTranspiler(opt_level=2)
    TKET_OPT = TketFullPeephole()


    QUBIT_THRESHOLD = 10

    bench_dir = "circuits/bench"
    bench_fs = [f for f in listdir(bench_dir) if isfile(join(bench_dir, f))]
    bench_cs = [(f, zx.Circuit.load(join(bench_dir, f)).to_basic_gates()) for f in bench_fs] # (filename, circuit) pairs
    qbs = [c.qubits for (f, c) in bench_cs]

    # plt.hist(qbs)
    # plt.show()
    cs_thresholded = [(f, c) for (f, c) in bench_cs if c.qubits <= QUBIT_THRESHOLD]
    print(f"{len(bench_cs)} benchmark circuits, {len(cs_thresholded)} of which have at most {QUBIT_THRESHOLD} qubits")

    reductions = dict()

    for (f, c) in cs_thresholded[:3]:

        init_score = c_score(c)
        g = c.to_graph()
        g_tmp = g.copy()

        if METHOD == "FR":
            g_opt = g_tmp.copy()

            zx.full_reduce(g_opt)
            c_opt = zx.extract_circuit(g_opt.copy()).to_basic_gates()
            c_opt = zx.basic_optimization(c_opt)

            opt_score = c_score(c_opt)
        elif METHOD == "TR":
            g_opt = g_tmp.copy()

            zx.teleport_reduce(g_opt)
            c_opt = zx.Circuit.from_graph(g_opt.copy()).to_basic_gates()
            c_opt = zx.basic_optimization(c_opt)

            opt_score = c_score(c_opt)
            # g_opt = c_opt.to_graph()
        elif METHOD == "qiskit":
            c_opt = QISKIT_OPT._optimize(c.copy())

            opt_score = c_score(c_opt)
        elif METHOD == "tket":
            c_opt = TKET_OPT._optimize(c.copy())

            opt_score = c_score(c_opt)
        elif METHOD == "SA" or METHOD == "GA":
            # obtain the initial simplified ZX-diagram
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

            opt_score = g_score(g_opt)
        else:
            raise RuntimeError(f"Invalid method: {METHOD}")

        reduction = (init_score - opt_score) / init_score * 100
        reductions[f] = reduction
    print(reductions)
