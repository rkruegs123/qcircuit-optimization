import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm

import sys
sys.path.append('../pyzx')
import pyzx as zx


def pyzx2nx(zx_graph):
    nx_graph = nx.DiGraph()
    nx_graph.add_nodes_from([v for v in zx_graph.vertices()])
    nx_graph.add_edges_from(zx_graph.edges())
    return nx_graph

# TODO: Should enforce a graph being graph-like
def enforce_graph_like(g):
    pass

if __name__ == "__main__":
    N_QUBITS = 10
    DEPTH = 300

    c = zx.generate.CNOT_HAD_PHASE_circuit(qubits=N_QUBITS, depth=DEPTH, clifford=False)
    g = c.to_graph()


    nx_g = pyzx2nx(g)
    nx.draw(nx_g)
    plt.show()
    """
    Potentially useful networkx functions:
    - nx.edge_load_centrality : maybe a good ranking function for pivoting
    - nx.load_centrality : maybe good for local complementation
    - See Algorithms -> Centrality in networkx manual
    """



    # The below shows that it is basically never worth trying to get the circuit from an arbitrary graph
    """
    try:
        c_orig = zx.Circuit.from_graph(g)
        print("Can convert from original graph back to circuit")
    except:
        print("Can NOT convert from original graph back to circuit")

    successes = 0
    for _ in tqdm(range(1000)):
        c = zx.generate.CNOT_HAD_PHASE_circuit(qubits=N_QUBITS, depth=DEPTH, clifford=False)
        g = c.to_graph()
        zx.full_reduce(g)
        try:
            c_opt = zx.Circuit.from_graph(g)
            successes += 1
        except:
            continue
    print(f"Number of successes: {successes}")
    """
