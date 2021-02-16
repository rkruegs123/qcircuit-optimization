from abc import ABC, abstractmethod
from os import listdir
from os.path import isfile, join
from tqdm import tqdm
import matplotlib.pyplot as plt
import pdb
import seaborn as sns
import pandas as pd

import sys
sys.path.append('../pyzx')
import pyzx as zx

# def full_reduce_tracker(g, quiet=True, stats=None):
# tracker = {'tcount' = list(), 'twoqubitcount': list(), 'total': list()}

"""
zx.generate.CNOT_HAD_PHASE_circuit(qubits=qubits, depth=depth, clifford=False)
zx.generate.cliffortT(qubits=qubits, depth=depth, backend=None)
zx.generate.clifford(qubits=qubits, depth=depth, backend=None)
zx.generate.cnots(qubits=qubits, depth=depth, backend=None)
"""

if __name__ == "__main__":
    N_QUBITS = 10
    DEPTH = 300

    c = zx.generate.CNOT_HAD_PHASE_circuit(qubits=N_QUBITS, depth=DEPTH, clifford=False)
    g = c.to_graph()

    reduce_tracker = zx.full_reduce_tracker(g, quiet=True)
    reduce_tracker['tcount'].insert(0, c.tcount())
    reduce_tracker['twoqubitcount'].insert(0, c.twoqubitcount())
    reduce_tracker['total'].insert(0, len(c.gates))
    reduce_tracker['edges'].insert(0, c.to_graph().num_edges())
    reduce_tracker['cliffordcount'].insert(0, c.cliffordcount())

    c_opt = zx.extract_circuit(g)
    print("-----FULL_REDUCE-----")
    print(f"T Count: {c.tcount()} -> {c_opt.tcount()}")
    print(f"2-qubit Count: {c.twoqubitcount()} -> {c_opt.twoqubitcount()}")
    print(f"Total Count: {len(c.gates)} -> {len(c_opt.gates)}")


    n_steps = len(reduce_tracker['total'])
    xs = list(range(n_steps))

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(xs, reduce_tracker['tcount'], color='blue', label='T Count')
    ax1.plot(xs, reduce_tracker['total'], color='green', label='Total')
    ax1.plot(xs, reduce_tracker['twoqubitcount'], color='red', label='2-qubit count')
    ax1.plot(xs, reduce_tracker['edges'], color='purple', label='edges')
    ax1.plot(xs, reduce_tracker['cliffordcount'], color='orange', label='Clifford count')

    plt.axvline(x=0.5, alpha=0.5)
    for x in range(3, n_steps - 1, 4):
        plt.axvline(x=x + 0.5, alpha=0.5)

    plt.title("full_reduce for randomized circuit")
    plt.xlabel("optimization steps\n(0th is original circuit, first 2 in full reduce, then 4 steps = one loop")
    plt.legend()
    plt.show()
