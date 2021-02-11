import pyzx as zx
import os
import pdb
import argparse
from os import listdir
from os.path import isfile, join
from tqdm import tqdm
import numpy as np


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate a directory of random circuits for benchmarking optimization methods')
    parser.add_argument('-m', '--method', type=str, choices=['CNOT_HAD_PHASE', 'cliffordT', 'cliffords', 'cnots'],
                        help='Method for randomly sampling circuits', required=True)
    parser.add_argument('-d', '--depth', nargs='+', type=int, help='Possible circuit depths', required=True)
    parser.add_argument('-qb', '--qubits', nargs='+', type=int, help='Possible number of qubits', required=True)
    parser.add_argument('-n', '--n_circuits', type=int, default=100, help='Number of circuits to randomly generate')
    args = parser.parse_args()

    print(args)

    method = args.method

    def gen_rand_circ(qubits, depth):
        if method == "CNOT_HAD_PHASE":
            return zx.generate.CNOT_HAD_PHASE_circuit(qubits=qubits, depth=depth, clifford=False)
        elif method == "cliffordT":
            return zx.generate.cliffortT(qubits=qubits, depth=depth, backend=None)
        elif method == "cliffords":
            return zx.generate.clifford(qubits=qubits, depth=depth, backend=None)
        elif method == "cnots":
            return zx.generate.cnots(qubits=qubits, depth=depth, backend=None)
        else:
            raise RuntimeError(f"[gen_rand_circ]: Invalid method {method}")


    ds = args.depth
    qbs = args.qubits
    n = args.n_circuits
    basedir = f"circuits/{method}"
    dname = f"qubits_{'-'.join([str(qb) for qb in qbs])}_depth_{'-'.join([str(d) for d in ds])}"
    odir = join(basedir, dname)

    if os.path.exists(odir):
        sys.exit("ERROR: output directory already exists")

    os.makedirs(odir)

    qubit_depth_pairs = [(qb, d) for qb in qbs for d in ds]
    assert(len(qubit_depth_pairs) <= n) # want at least one circuit for each pair

    # each pair will have approximately the same number of circuits
    chunks = np.array_split(list(range(n)), len(qubit_depth_pairs))

    pbar = tqdm(total=n, desc="Generating random circuits...")
    for (chunk, (qubits, depth)) in zip(chunks, qubit_depth_pairs):
        for (i, id) in enumerate(chunk):
            # fname = f"{id}_qubits-{qubits}_depth-{depth}_{i}"
            fname = f"qubits-{qubits}_depth-{depth}_{i}.qasm"
            circ = gen_rand_circ(qubits, depth)
            with open(join(odir, fname), 'w') as f:
                f.write(circ.to_qasm())
            pbar.update(1)
    pbar.close()
