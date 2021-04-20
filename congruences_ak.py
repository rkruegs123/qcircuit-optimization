import random
import math
from fractions import Fraction

import sys
sys.path.append('../pyzx')
from pyzx.utils import VertexType, EdgeType
import pyzx as zx


###### Pivoting

def is_gadget(g, v):
    # RK: Is this different than PHASE gadget? Why do we not enforce Hadamard edge?
    return g.vertex_degree(v) == 1 and g.type(v) == VertexType.Z
def is_boundary(g, v):
    return g.type(v) == VertexType.BOUNDARY
def adj_gadget(g, v):
    for n in g.neighbors(v):
        if is_gadget(g, n): return (True, n)
    return None
def adj_boundary(g,v):
    for n in g.neighbors(v):
        if is_boundary(g, n): return (True, n)
    return None

def unfuse(g,v):
    v1 = g.add_vertex(1,
        qubit=g.qubit(v)-1,
        row=g.row(v),
        phase=g.phase(v))
    g.set_phase(v, 0)
    g.add_edge((v,v1), 1)
    return (True, v1)

def simp_pair(g, v1, v2):
    if g.type(v2) == VertexType.Z:
        if g.edge_type(g.edge(v1,v2)) == EdgeType.SIMPLE:
            g.add_to_phase(v1, g.phase(v2))
            g.remove_vertex(v2)
        elif g.phase(v1) == Fraction(1):
            g.set_phase(v1, 0)
            g.set_phase(v2, -g.phase(v2))
    else:
        # TODO: handle boundaries properly. Currently just swallows the phase
        g.set_phase(v1, 0)

def toggle_edge_type(g, v1, v2):
    e = g.edge(v1,v2)
    et = 1+(2-g.edge_type(e))
    g.set_edge_type(e, et)
    return et

def is_pivot_edge_ak(g, e):
    v1, v2 = g.edge_st(e)
    return (g.type(v1) == VertexType.Z and g.type(v2) == VertexType.Z and
            not is_gadget(g, v1) and not is_gadget(g, v2)) # RK: what if only one of them has degree 1?

def pivot_cong_ak(g, v1, v2):
    x,adj1 = adj_boundary(g, v1) or adj_gadget(g,v1) or unfuse(g, v1)
    x,adj2 = adj_boundary(g, v2) or adj_gadget(g,v2) or unfuse(g, v2)

    toggle_edge_type(g, v1, adj1)
    toggle_edge_type(g, v2, adj2)

    etab = {}

    nhd1 = list(g.neighbors(v1))
    nhd1.remove(adj1)
    nhd1.append(v1)

    for n in nhd1:
        if g.type(n) == VertexType.BOUNDARY:
            raise ValueError("got boundary in nhd1")

    nhd2 = list(g.neighbors(v2))
    nhd2.remove(adj2)
    nhd2.append(v2)

    for n in nhd2:
        if g.type(n) == VertexType.BOUNDARY:
            raise ValueError("got boundary in nhd2")

    for n1 in nhd1:
        for n2 in nhd2:
            if ((n1 == v1 and n2 == v2) or
                (n1 == v2 and n2 == v1)): continue
            elif n1 == n2:
                g.add_to_phase(n1, Fraction(1))
            else:
                e = (n1,n2) if n1 < n2 else (n2,n1)
                x,he = etab.get(e,(0,0))
                etab[e] = (0,he+1)
    g.add_edge_table(etab)

    simp_pair(g, v1, adj1)
    simp_pair(g, v2, adj2)
