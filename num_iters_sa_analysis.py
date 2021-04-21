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

from anneal import pivot_anneal
from utilities import to_graph_like

if __name__ == "__main__":
    cs = list()
    for _ in range(3):
        for n_qubits in range(4, 12, 2):
            for gates_per_qubit in range(10, 25, 5):
                cs.append(zx.generate.CNOT_HAD_PHASE_circuit(qubits=n_qubits,
                                                             depth=n_qubits * gates_per_qubit,
                                                             clifford=False))

    N_ITERS = 5000
    INTERVAL = 250
    improvement_after = { x: list() for x in range(0, N_ITERS, INTERVAL) }

    REDUCE_METHOD = "TR"

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

        _, scores = pivot_anneal(g_simp, iters=N_ITERS)
        final_score = scores[-1]
        for x in range(0, N_ITERS, INTERVAL):
            if final_score < scores[x]:
                improvement_after[x].append(1)
            else:
                improvement_after[x].append(0)


    pdb.set_trace()

    improvement_after_probs = {k: np.mean(v) for k, v in improvement_after.items()}
    xs = list(improvement_after_probs.keys())
    ys = list(improvement_after_probs.values())
    plt.rcParams.update({'font.size': 14})
    plt.scatter(xs, ys)
    plt.xlabel("Steps")
    plt.ylabel("Probability of Future Improvement")
    plt.title("Likelihood of Improvement Throughout SA")
    plt.show()


    pdb.set_trace()
    print("HI")
