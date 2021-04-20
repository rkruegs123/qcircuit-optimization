import random
import math
from fractions import Fraction
from tqdm import tqdm
import matplotlib.pyplot as plt
import itertools
import pdb
import numpy as np

import sys
sys.path.append('../pyzx')
from pyzx.utils import VertexType, EdgeType
import pyzx as zx

from utilities import is_graph_like, to_graph_like, uniform_weights
from congruences_ak import pivot_cong_ak, is_pivot_edge_ak


###### Utilities

# note: modifies graph in-place
def toggle_edge(g, v1, v2):
    if g.connected(v1, v2):
        # print(f"REMOVED: {v1}, {v2}")
        g.remove_edge(g.edge(v1, v2))
    else:
        # print(f"ADDED: {v1}, {v2}")
        g.add_edge(g.edge(v1, v2), edgetype=zx.EdgeType.HADAMARD)


def toggle_subset_connectivity(g, vs1, vs2):
    for v1 in vs1:
        for v2 in vs2:
            toggle_edge(g, v1, v2)

def unfuse(g, v):
    """
    For a Z-spider with a phase and neighbors that are both BOUNDARY and Z, unfuse the phase in the form of a new spider that holds the connectivity to the boundaries.
    Note that v will maintain its connectivity to any Z spiders (just not boundaries).
    """
    v_phase = g.phase(v)
    bs = [v for v in g.neighbors(v) if g.type(v) == VertexType.BOUNDARY]
    # we could enforce the following to ensure graph-likeness, but not necessary
    # assert(len(bs) < 2)
    new_v = g.add_vertex(
        ty=VertexType.Z,
        phase=v_phase)

    # unfuse the phase
    g.set_phase(v, phase=0)
    g.add_edge(g.edge(v, new_v), edgetype=zx.EdgeType.SIMPLE)

    # transfer over the boundary connectivity to the new spider
    for b in bs:
        e = g.edge(v, b)
        b_edge_type = g.edge_type(e)
        g.remove_edge(e)
        g.add_edge(g.edge(new_v, b), edgetype=b_edge_type)

    # return the reference to the new vertex
    return new_v


###### Pivoting

# TODO: May want to add some additional cases when we know it's not useful (as in the LC case)
def is_pivot_edge(g, e):
    v1, v2 = g.edge_st(e)
    return g.type(v1) == VertexType.Z and g.type(v2) == VertexType.Z

def pivot_cong(g, v1, v2):
    # get the three subsets
    nhd1 = list(g.neighbors(v1))
    nhd2 = list(g.neighbors(v2))
    assert(all([g.type(v) in [VertexType.Z, VertexType.BOUNDARY] for v in nhd1]))
    assert(all([g.type(v) in [VertexType.Z, VertexType.BOUNDARY] for v in nhd2]))
    vs1 = [v for v in nhd1 if g.type(v) == VertexType.Z]
    bs1 = [v for v in nhd1 if g.type(v) == VertexType.BOUNDARY]
    vs2 = [v for v in nhd2 if g.type(v) == VertexType.Z]
    bs2 = [v for v in nhd2 if g.type(v) == VertexType.BOUNDARY]

    shared_ns = list(set(vs1) & set(vs2))
    nhd1_only = list(set(vs1) - set(vs2 + [v2]))
    nhd2_only = list(set(vs2) - set(vs1 + [v1]))

    # toggle connectivity between the three subsets
    toggle_subset_connectivity(g, shared_ns, nhd1_only)
    toggle_subset_connectivity(g, shared_ns, nhd2_only)
    toggle_subset_connectivity(g, nhd1_only, nhd2_only)

    # add pi to all the shared neighbors
    for n in shared_ns:
        g.add_to_phase(n, 1)

    # for v1 and v2, pull out phases along hadamards and swap (while maintaining any I/O)
    new_v1 = unfuse(g, v1)
    new_v2 = unfuse(g, v2)

    g.remove_edge(g.edge(v1, new_v1))
    g.remove_edge(g.edge(v2, new_v2))
    g.add_edge(g.edge(v1, new_v2), edgetype=EdgeType.HADAMARD)
    g.add_edge(g.edge(v2, new_v1), edgetype=EdgeType.HADAMARD)




###### Local Complementation

def is_lc_vertex(g, v):
    # don't want to apply LC to a single-degree spider because it will only result in a chain growing
    if g.vertex_degree(v) < 2 or g.type(v) != VertexType.Z:
        return False

    # not doing I/O vertices for now. note taht they are included in degree count
    for n in g.neighbors(v):
        if g.type(n) == VertexType.BOUNDARY:
            return False

    return True


# note: assumes that v is a green spider
def lc_cong(g, v):
    ns = [n for n in g.neighbors(v) if g.type(n) == VertexType.Z]
    # print(f"Neighbors: {ns}")

    # swap edges between neighbors
    # TODO: Should really be using edge table (e.g., add_edge_table)
    for n1, n2 in itertools.combinations(ns, 2):
        toggle_edge(g, n1, n2)


    # add pi/2 to all neighbors
    for n in ns:
        g.add_to_phase(n, Fraction(1, 2))

    # add gadget (and transfer over the phase)
    new_v = g.add_vertex(
        ty=zx.VertexType.Z,
        phase=g.phase(v) + Fraction(1, 2),
        qubit=g.qubit(v)-1,
        row=g.row(v)
    )
    g.set_phase(v, phase=Fraction(1, 2))
    g.add_edge(g.edge(v, new_v), edgetype=zx.EdgeType.HADAMARD)


    # FIXME: not gracefully handling if on boundary. If on boundary, sohuld just add on same qubit rather than add a gadget




def apply_rand_pivot(g, weight_func=uniform_weights):
    # assumes len(candidates) != 0
    candidates = [e for e in g.edges() if is_pivot_edge(g, e)]
    weights = weight_func(g, candidates)
    e_idx = np.random.choice(len(candidates), 1, p=weights)[0]
    e = candidates[e_idx]
    v1, v2 = g.edge_st(e)
    pivot_cong(g, v1, v2)

def apply_rand_lc(g, weight_func=uniform_weights):
    lc_vs = [v for v in g.vertices() if is_lc_vertex(g, v)]
    weights = weight_func(g, lc_vs)
    lc_v = np.random.choice(lc_vs, 1, p=weights)[0]
    lc_cong(g, lc_v)



if __name__ == "__main__":

    CONG_METHOD = "PIVOT"
    # CONG_METHOD = "LC"

    # generate circuit to test on
    N_QUBITS = 3
    DEPTH = 35
    c = zx.generate.CNOT_HAD_PHASE_circuit(qubits=N_QUBITS, depth=DEPTH, clifford=False)
    g = c.to_graph()
    # note: full_reduce doesn't leave it in a graph-like state!!
    zx.full_reduce(g)
    if not is_graph_like(g):
        print("Putting to graph like...")
        to_graph_like(g)
    g1 = g.copy()
    zx.draw(g1, labels=True)
    # plt.show()


    if CONG_METHOD == "LC":
        lc_vs = [v for v in g1.vertices() if is_lc_vertex(g1, v)]
        lc_v = random.choice(lc_vs)
        print(f"Local complementation on vertex: {lc_v}")
        lc_cong(g1, lc_v)
    elif CONG_METHOD == "PIVOT":
        candidates = [e for e in g1.edges() if is_pivot_edge(g1, e)]
        e = candidates[random.randint(0, len(candidates)-1)]
        v1, v2 = g1.edge_st(e)
        print(f"Pivoting on vertices: {v1}, {v2}")
        pivot_cong(g1, v1, v2)
    else:
        raise RuntimeError(f"Unrecognized congruence method: {CONG_METHOD}")

    print(f"Is graph-like after {CONG_METHOD}: {is_graph_like(g1)}")
    zx.draw(g1, labels=True)

    # try to extract the new circuit, applying full_reduce to put in normal form if necessary
    # note: even when graph-likeness is preserved, we have to full_reduce to extract a circuit. why?
    c1 = None
    try:
        c1 = zx.extract_circuit(g1.copy())
        print("Successfully extracted circuit (directly after congruence)")
    except:
        try:
            zx.full_reduce(g1)
            # zx.draw(g1, labels=True) # additionally plots the reduced circuit
            c1 = zx.extract_circuit(g1.copy())
            print("Successfully extracted circuit (full_reduce required)")
        except Exception as e:
            print(f"Circuit extraction failed (after full_reduce): {e}")

    # print(f"compare_tensors: {zx.compare_tensors(g, g1)}")
    if c1 is not None:
        print(f"verify_equality: {c.verify_equality(c1)}")

    plt.show()
    print("DONE")
