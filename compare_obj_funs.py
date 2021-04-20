from tqdm import tqdm
import itertools
import pdb
import numpy as np
import networkx as nx
import seaborn as sns
import pandas as pd
from scipy.stats import pearsonr

import sys
sys.path.append('../pyzx')
import pyzx as zx
from anneal import c_score, edge_count, density, centrality

if __name__ == "__main__":
    N_CS = 500
    N_QUBITS = 10
    DEPTH = 100
    gs = [zx.generate.CNOT_HAD_PHASE_circuit(qubits=N_QUBITS, depth=DEPTH, clifford=False).to_graph()
          for _ in tqdm(range(N_CS), desc="Generating...")]
    comps = [c_score(g) for g in tqdm(gs, desc="Complexity...")]
    n_edges = [edge_count(g) for g in gs]
    densities = [density(g) for g in gs]
    centralities = [centrality(g) for g in gs]
    print(f"Num edges: {pearsonr(comps, n_edges)}")
    print(f"Density: {pearsonr(comps, densities)}")
    print(f"Centrality: {pearsonr(comps, centralities)}")
