import random
import math
from fractions import Fraction
from tqdm import tqdm
import matplotlib.pyplot as plt
import itertools
import pdb
import numpy as np
import networkx as nx
import seaborn as sns
import pandas as pd
from functools import partial
from math import floor, ceil

import sys
sys.path.append('../pyzx')
from pyzx.utils import VertexType, EdgeType
import pyzx as zx

from congruences import is_pivot_edge, pivot_cong, is_lc_vertex, lc_cong, apply_rand_pivot, \
    apply_rand_lc
from congruences_ak import is_gadget
from utilities import to_graph_like, pyzx2nx, uniform_weights, c_score, g_score



def size(g):
    return (
        sum(1 for e in g.edges()) -
        sum(1 if is_gadget(g, v) else 0 for v in g.vertices())
    )


def edge_count(g):
    return g.num_edges()

def density(g):
    nx_g = pyzx2nx(g.copy())
    return nx.density(nx_g)

def centrality(g):
    nx_g = pyzx2nx(g.copy())
    return nx.estrada_index(nx_g)

def get_central_nodes(g, allowed, n=20, method="load_centrality"):
    nx_g = pyzx2nx(g.copy())

    if method == "load_centrality":
        centralities = nx.load_centrality(nx_g)
    elif method == "neg_load_centrality":
        centralities = {k: -v for (k, v) in nx.load_centrality(nx_g).items()}
    elif method == "degree_centrality":
        centralities = nx.degree_centrality(nx_g)
    elif method == "katz_centrality":
        centralities = nx.katz_centrality(nx_g)
    elif method == "betweenness_centrality":
        centralities = nx.betweenness_centrality(nx_g)
    elif method == "current_flow_betweenness_centrality":
        centralities = nx.current_flow_betweenness_centrality(nx_g)
    elif method == "harmonic_centrality":
        centralities = nx.harmonic_centrality(nx_g)
    elif method == "second_order_centrality":
        centralities = nx.second_order_centrality(nx_g)
    elif method == "degree":
        centralities = nx_g.degree()
    elif method == "avg_neighb_degree":
        centralities = nx.algorithms.assortativity.average_neighbor_degree(nx_g)
    elif method == "sum_neighb_degree":
        dgs = nx_g.degree()
        avg_n_degrees = nx.algorithms.assortativity.average_neighbor_degree(nx_g)
        centralities = {v: dg * avg_n_degrees[v] for (v, dg) in dgs}
    else:
        raise RuntimeError(f"[get_central_nodes] Unrecognized method {method}")

    # Relies on order being preserved in dictionary
    allowed_centralities = { v: centralities[v] for v in allowed }
    sum_allowed = sum(allowed_centralities.values())
    weights = [float(v)/sum_allowed for v in allowed_centralities.values()]

    # ranked_nodes = sorted(allowed_centralities.items(), key=lambda item: -item[1])
    # top_n = [k for (k, _) in ranked_nodes[:n]]
    # return top_n

    return weights



def get_central_edges(g, allowed, n=20, method="load_centrality"):
    nx_g = pyzx2nx(g.copy())

    if method == "load_centrality":
        centralities = nx.edge_load_centrality(nx_g)
    elif method == "betweenness_centrality":
        centralities = nx.edge_betweenness_centrality(nx_g)
    elif method == "pos_dispersion":
        centralities = { (s, t): nx.dispersion(nx_g, u=s, v=t) for (s, t) in g.edges()}
    elif method == "neg_dispersion":
        centralities = { (s, t): -nx.dispersion(nx_g, u=s, v=t) for (s, t) in g.edges()}
    elif method == "degree":
        dgs = nx_g.degree()
        centralities = { (s, t): dgs[s] + dgs[t] for (s, t) in g.edges() }
    elif method == "sum_neighb_degree":
        dgs = nx_g.degree()
        all_neighbors = { (s, t): list(set(list(nx_g.neighbors(s)) + list(nx_g.neighbors(t)))) for (s, t) in g.edges() }
        centralities = { (s, t): sum([dgs[n] for n in ns]) for ((s, t), ns) in all_neighbors.items() }
    elif method == "avg_neighb_degree":
        dgs = nx_g.degree()
        all_neighbors = { (s, t): list(set(list(nx_g.neighbors(s)) + list(nx_g.neighbors(t)))) for (s, t) in g.edges() }
        centralities = { (s, t): np.mean([dgs[n] for n in ns]) for ((s, t), ns) in all_neighbors.items() }
    else:
        raise RuntimeError(f"[get_central_edges] Unrecognized method {method}")

    allowed_centralities = { e: centralities[e] for e in allowed }
    sum_allowed = sum(allowed_centralities.values())
    weights = [float(v)/sum_allowed for v in allowed_centralities.values()]

    # ranked_edges = sorted(allowed_centralities.items(), key=lambda item: -item[1])
    # top_n = [k for (k, _) in ranked_edges[:n]]
    # return top_n

    return weights


# simulated annealing
def pivot_anneal(g, iters=1000, temp=25, cool=0.005, score=g_score, cong_ps=[0.5, 0.5],
                 lc_select=uniform_weights,
                 pivot_select=uniform_weights,
                 full_reduce_prob=0.1, reset_prob=0.0):
    g_best = g.copy()
    sz = score(g_best)
    sz_best = sz

    best_scores = list()

    for i in tqdm(range(iters), desc="annealing..."):

        g1 = g.copy()

        cong_method = np.random.choice(["LC", "PIVOT"], 1, p=cong_ps)[0]

        if cong_method == "PIVOT":
            apply_rand_pivot(g1, weight_func=pivot_select)
        else:
            apply_rand_lc(g1, weight_func=lc_select)

        # probabilistically full_reduce:
        if random.uniform(0, 1) < full_reduce_prob:
            zx.full_reduce(g1)
        sz1 = score(g1)

        best_scores.append(sz_best)

        if temp != 0: temp *= 1.0 - cool
        # if i % 50 == 0:
            # print(i)
            # print(temp)

        if sz1 < sz or \
            (temp != 0 and random.random() < math.exp((sz - sz1)/temp)):
            # if temp != 0: temp *= 1.0 - cool

            sz = sz1
            g = g1.copy()
            if sz < sz_best:
                # print("NEW BEST")
                g_best = g.copy()
                sz_best = sz
        elif random.uniform(0, 1) < reset_prob:
            g = g_best.copy()


    return g_best, best_scores





if __name__ == "__main__":
    N_QUBITS = 5
    DEPTH = 100
    c = zx.generate.CNOT_HAD_PHASE_circuit(qubits=N_QUBITS, depth=DEPTH, clifford=False)
    print("----- initial -----")
    print(c.stats())

    # zx.draw(c)
    plt.show()
    plt.close('all')

    g = c.to_graph()

    g_fr = g.copy()
    zx.full_reduce(g_fr)
    c_fr = zx.extract_circuit(g_fr.copy()).to_basic_gates()
    c_fr = zx.basic_optimization(c_fr)
    # note: we don't reset g_fr here because for any future annealing, we'd really optimize after the graph produced by full_reduce, rather than something resulting from extraction
    print("\n----- full_reduce + basic_optimization -----")
    print(c_fr.stats())


    g_just_tr = g.copy()
    zx.teleport_reduce(g_just_tr)
    c_just_tr = zx.Circuit.from_graph(g_just_tr.copy()).to_basic_gates()
    print("\n----- teleport_reduce -----")
    print(c_just_tr.stats())


    g_tr = g.copy()
    zx.teleport_reduce(g_tr)
    c_tr = zx.Circuit.from_graph(g_tr.copy()).to_basic_gates()
    # c_opt = zx.full_optimize(c_opt)
    c_tr = zx.basic_optimization(c_tr)
    g_tr = c_tr.to_graph()
    print("\n----- teleport_reduce + basic_optimization -----")
    print(c_tr.stats())
    to_graph_like(g_tr)


    g_tr_extract = g.copy()
    zx.teleport_reduce(g_tr_extract)
    c_tr_extract = zx.extract_circuit(g_tr_extract.copy()).to_basic_gates()
    c_tr_extract = zx.basic_optimization(c_tr_extract)
    print("\n----- teleport_reduce (extract) + basic_optimization -----")
    print(c_tr_extract.stats())


    g_tr2 = g.copy()
    zx.teleport_reduce(g_tr2)
    c_tr2 = zx.Circuit.from_graph(g_tr2.copy()).to_basic_gates()
    c_tr2 = zx.full_optimize(c_tr2)
    g_tr2 = c_tr2.to_graph()
    print("\n----- teleport_reduce + full_optimize -----")
    print(c_tr2.stats())
    to_graph_like(g_tr2)


    # For annealing over output of full_reduce
    """
    # to_graph_like(g_fr)
    g_anneal, _ = pivot_anneal(g_fr, iters=100, score=g_score)
    zx.full_reduce(g_anneal)
    c_anneal = zx.extract_circuit(g_anneal.copy()).to_basic_gates()
    c_anneal = zx.basic_optimization(c_anneal)
    print("\n----- anneal (fr, LR + PIVOT) -----")
    print(c_anneal.stats())

    print(f"\nverify_equality: {c.verify_equality(c_anneal)}")
    """



    # For annealing over output of teleport_reduce

    # to_graph_like(g_tr)
    """
    g_anneal, _ = pivot_anneal(g_tr, iters=1000, score=g_score, full_reduce_prob=0.1, reset_prob=0.0)
    zx.full_reduce(g_anneal)
    c_anneal = zx.extract_circuit(g_anneal.copy()).to_basic_gates()
    c_anneal = zx.basic_optimization(c_anneal)
    print("\n----- anneal (tr, LR + PIVOT) -----")
    print(c_anneal.stats())
    """


    # For testing effect of only LC or PIVOT
    """
    g_anneal_lc, _ = pivot_anneal(g_tr.copy(), score=g_score, cong_ps=[1.0, 0.0])
    c_anneal_lc = zx.extract_circuit(g_anneal_lc.copy())
    print("\n----- + anneal (LC only) -----")
    print(c_anneal_lc.stats())

    g_anneal_pivot, _ = pivot_anneal(g_tr.copy(), score=g_score, cong_ps=[0.0, 1.0])
    c_anneal_pivot = zx.extract_circuit(g_anneal_pivot.copy())
    print("\n----- + anneal (PIVOT only) -----")
    print(c_anneal_pivot.stats())
    """


    # for measuring the consistency of pivot_anneal
    """
    reductions = list()
    N_TRIALS = 10
    # to_graph_like(g_tr)

    init_score = c_score(c_tr) # 10 * c_tr.twoqubitcount() + c_tr.tcount()
    for i in tqdm(range(N_TRIALS)):
        g_anneal, _ = pivot_anneal(g_tr.copy(), iters=100, score=g_score)
        zx.full_reduce(g_anneal)
        c_anneal = zx.extract_circuit(g_anneal.copy()).to_basic_gates()
        c_anneal = zx.basic_optimization(c_anneal)

        trial_score = c_score(c_anneal) # 10 * c_anneal.twoqubitcount() + c_anneal.tcount()
        reduction = (init_score - trial_score) / init_score * 100
        reductions.append(reduction)

        print(f"\n----- trial {i} -----")
        print(c_anneal.stats())


    plt.hist(reductions)
    plt.xlabel("Reduction")
    plt.ylabel("Frequency")

    plt.show()
    print("DONE")
    """

    # For plotting score throughout annealing process (for multiple trials)

    # to_graph_like(g_tr)

    init_score = c_score(c_tr) # 10 * c_tr.twoqubitcount() + c_tr.tcount()
    for _ in range(1):
        g_anneal, tracker = pivot_anneal(g_tr.copy(), iters=2000, score=g_score, full_reduce_prob=0.1, reset_prob=0.0)
        zx.full_reduce(g_anneal)
        c_anneal = zx.extract_circuit(g_anneal.copy()).to_basic_gates()
        c_anneal = zx.basic_optimization(c_anneal)
        xs = list(range(len(tracker)))
        plt.plot(xs, tracker)
        # print("\n----- anneal (tr, LR + PIVOT) -----")
        print(c_anneal.stats())

    plt.xlabel("Iterations")
    plt.ylabel("Score")
    plt.axhline(y=init_score, color='r', linestyle='--', label="tr + basic")
    plt.legend()
    plt.show()



    # For measuring difference between LC + PIVOT, LC only, PIVOT only

    # to_graph_like(g_tr)
    """
    d = {'reduction': list(), 'method': list(), 'complexity': list()}
    init_score = c_score(c_tr) # 10 * c_tr.twoqubitcount() + c_tr.tcount()
    for _ in tqdm(range(10), desc='trials'):
        for (ps, method) in [([1.0, 0.0], "LC only"), ([0.75, 0.25], "75% LC, 25% Pivot"), ([0.5, 0.5], "Both"), ([0.25, 0.75], "25% LC, 75% Pivot"), ([0.0, 1.0], "Pivot only")]:
        # for (ps, method) in [([0.5, 0.5], "Both"), ([1.0, 0.0], "LC only")]:
            g_anneal, _ = pivot_anneal(g_tr.copy(), score=g_score, cong_ps=ps, full_reduce_prob=0.1)
            zx.full_reduce(g_anneal)
            c_anneal = zx.extract_circuit(g_anneal.copy()).to_basic_gates()
            c_anneal = zx.basic_optimization(c_anneal)
            trial_score = c_score(c_anneal) # 10 * c_anneal.twoqubitcount() + c_anneal.tcount()
            reduction = (init_score - trial_score) / init_score * 100
            d['reduction'].append(reduction)
            d['method'].append(method)
            d['complexity'].append(trial_score)
    df = pd.DataFrame(data=d)
    # sns.boxplot(x="method", y="reduction", data=df)
    plt.rcParams.update({'font.size': 16})
    sns.boxplot(x="method", y="complexity", data=df)
    # plt.title("Congruence Sampling")
    plt.ylabel("Complexity")
    plt.xlabel("Congruence Sampling Method")

    plt.show()
    """



    # First test if full_reduce is deterministic
    """
    fr_scores = list()
    for _ in tqdm(range(100)):
        g_tmp = g.copy()
        zx.full_reduce(g_tmp)
        c_tmp = zx.extract_circuit(g_tmp.copy()).to_basic_gates()
        c_tmp = zx.basic_optimization(c_tmp)
        trial_score = c_score(c_tmp) # 10 * c_tmp.twoqubitcount() + c_tmp.tcount()
        fr_scores.append(trial_score)
    plt.hist(fr_scores)
    plt.ylabel("Frequency")
    plt.xlabel("Score")
    plt.show()
    """


    # For measuring the benefit of prepending congruence(s) to full_reduce
    """
    init_fr_score = c_score(c_fr) # 10 * c_fr.twoqubitcount() + c_fr.tcount()
    to_graph_like(g)
    trial_scores = list()
    for _ in tqdm(range(1000)):

        g_tmp = g.copy()

        apply_cong = True

        while apply_cong:

            cong_method = np.random.choice(["LC", "PIVOT"], 1, p=[0.5, 0.5])[0]
            if cong_method == "PIVOT":
                candidates = [e for e in g_tmp.edges() if is_pivot_edge(g_tmp, e)]
                e = random.choice(candidates)
                v1, v2 = g_tmp.edge_st(e)
                pivot_cong(g_tmp, v1, v2)
            else:
                lc_vs = [v for v in g_tmp.vertices() if is_lc_vertex(g_tmp, v)]
                lc_v = random.choice(lc_vs)
                lc_cong(g_tmp, lc_v)

            # coin flip to keep applying congruences
            apply_cong = True if random.uniform(0, 1) < 0.5 else False

        zx.full_reduce(g_tmp)
        c_opt = zx.extract_circuit(g_tmp.copy()).to_basic_gates()
        c_opt = zx.basic_optimization(c_opt)
        trial_score = c_score(c_opt) # 10 * c_opt.twoqubitcount() + c_opt.tcount()
        trial_scores.append(trial_score)

    plt.hist(trial_scores)
    plt.axvline(x=init_fr_score, color='r', linestyle='--', label="Original full_reduce score")
    plt.xlabel("Scores")
    plt.legend()
    plt.show()
    """

    # Evaluate different vertex selection methods
    """
    load_cent = partial(get_central_nodes, method="load_centrality")
    neg_load_cent = partial(get_central_nodes, method="neg_load_centrality")
    degree_cent = partial(get_central_nodes, method="degree_centrality")
    # katz_cent = partial(get_central_nodes, method="katz_centrality")
    btw_cent = partial(get_central_nodes, method="betweenness_centrality")
    dg = partial(get_central_nodes, method="degree")
    n_deg_avg = partial(get_central_nodes, method="avg_neighb_degree")
    n_deg_sum = partial(get_central_nodes, method="sum_neighb_degree")

    d = {'reduction': list(), 'method': list(), 'complexity': list()}
    init_score = c_score(c_tr) # 10 * c_tr.twoqubitcount() + c_tr.tcount()
    for _ in tqdm(range(10), desc="Trial"):
        for (lc_select, method) in tqdm([(uniform_weights, "Uniform"),
                                         (load_cent, "Load Centrality"),
                                         ## (neg_load_cent, "-load_centrality"),
                                         ## (degree_cent, "Degree Centrality"),
                                         # (btw_cent, "Betweenness Centrality"),
                                         (dg, "Degree"),
                                         (n_deg_avg, "Neighb. Degrees (avg)"),
                                         (n_deg_sum, "Neighb. Degrees (sum)")],
                                        desc="Method"):
            g_anneal, _ = pivot_anneal(g_tr.copy(), score=g_score, cong_ps=[1.0, 0.0],
                                       lc_select=lc_select)
            zx.full_reduce(g_anneal)
            c_anneal = zx.extract_circuit(g_anneal.copy()).to_basic_gates()
            c_anneal = zx.basic_optimization(c_anneal)
            trial_score = c_score(c_anneal) # 10 * c_anneal.twoqubitcount() + c_anneal.tcount()
            reduction = (init_score - trial_score) / init_score * 100
            d['reduction'].append(reduction)
            d['complexity'].append(trial_score)
            d['method'].append(method)
    df = pd.DataFrame(data=d)
    plt.rcParams.update({'font.size': 14})
    sns.boxplot(x="method", y="complexity", data=df)
    plt.xlabel("LC Spider Selection Method")
    plt.ylabel("Complexity")
    plt.show()
    """


    # Evaluate different edge selection methods
    """
    load_cent = partial(get_central_edges, method="load_centrality")
    btw_cent = partial(get_central_edges, method="betweenness_centrality")
    # pos_disp = partial(get_central_edges, method="pos_dispersion")
    # neg_disp = partial(get_central_edges, method="neg_dispersion")

    dg = partial(get_central_edges, method="degree")
    n_deg_avg = partial(get_central_edges, method="avg_neighb_degree")
    n_deg_sum = partial(get_central_edges, method="sum_neighb_degree")

    d = {'reduction': list(), 'method': list(), 'complexity': list()}
    init_score = c_score(c_tr) # 10 * c_tr.twoqubitcount() + c_tr.tcount()
    for _ in tqdm(range(10), desc="Trial"):
        for (pivot_select, method) in tqdm([(uniform_weights, "Uniform"),
                                            # (load_cent, "load_centrality"),
                                            (dg, "Degree"),
                                            (btw_cent, "Betweenness Centrality"),
                                            (n_deg_sum, "Neighb. Degrees (sum)"),
                                            (n_deg_avg, "Neighb. Degrees (avg)")
                                            # (pos_disp, "dispersion"),
                                            # (neg_disp, "-dispersion")
        ],
                                        desc="Method"):
            g_anneal, _ = pivot_anneal(g_tr.copy(), score=g_score, cong_ps=[0.0, 1.0],
                                       pivot_select=pivot_select)
            zx.full_reduce(g_anneal)
            c_anneal = zx.extract_circuit(g_anneal.copy()).to_basic_gates()
            c_anneal = zx.basic_optimization(c_anneal)
            trial_score = c_score(c_anneal) # 10 * c_anneal.twoqubitcount() + c_anneal.tcount()
            reduction = (init_score - trial_score) / init_score * 100
            d['reduction'].append(reduction)
            d['complexity'].append(trial_score)
            d['method'].append(method)
    df = pd.DataFrame(data=d)
    sns.boxplot(x="method", y="complexity", data=df)
    plt.xlabel("Pivot Edge Selection Method")
    plt.ylabel("Complexity")
    plt.show()
    """


    # Compare different probabilities of full_reducing
    """
    d = {'reduction': list(), 'method': list()}
    init_score = c_score(c_tr) # 10 * c_tr.twoqubitcount() + c_tr.tcount()
    for _ in tqdm(range(10), desc="Trial"):
        for fr_prob in tqdm([0.01, 0.1, 0.5], desc="Method"):
            method = f"{fr_prob * 100}%"

            g_anneal, _ = pivot_anneal(g_tr.copy(), score=g_score, full_reduce_prob=fr_prob)
            zx.full_reduce(g_anneal)
            c_anneal = zx.extract_circuit(g_anneal.copy()).to_basic_gates()
            c_anneal = zx.basic_optimization(c_anneal)
            trial_score = c_score(c_anneal) # 10 * c_anneal.twoqubitcount() + c_anneal.tcount()
            reduction = (init_score - trial_score) / init_score * 100
            d['reduction'].append(reduction)
            d['method'].append(method)
    df = pd.DataFrame(data=d)
    sns.boxplot(x="method", y="reduction", data=df)
    plt.xlabel("Full Reduce Likelihood")
    plt.ylabel("Reduction")
    plt.show()
    """

    # Compare different scoring methods
    """
    d = {'reduction': list(), 'method': list()}
    init_score = c_score(c_tr) # 10 * c_tr.twoqubitcount() + c_tr.tcount()
    for _ in tqdm(range(20), desc="Trial"):
        for (score, method) in tqdm([(g_score, "default"), (edge_count, "# edges"), (density, "density"), (centrality, "centrality")], desc="Method"):

            g_anneal, _ = pivot_anneal(g_tr.copy(), score=score)
            zx.full_reduce(g_anneal)
            c_anneal = zx.extract_circuit(g_anneal.copy()).to_basic_gates()
            c_anneal = zx.basic_optimization(c_anneal)
            trial_score = c_score(c_anneal) # 10 * c_anneal.twoqubitcount() + c_anneal.tcount()
            reduction = (init_score - trial_score) / init_score * 100
            d['reduction'].append(reduction)
            d['method'].append(method)
    df = pd.DataFrame(data=d)
    sns.boxplot(x="method", y="reduction", data=df)
    plt.xlabel("Score Function")
    plt.ylabel("Reduction")
    plt.show()
    """

    # Compare different input graphs
    """
    d = {'score': list(), 'method': list()}
    init_score = c_score(c_tr) # 10 * c_tr.twoqubitcount() + c_tr.tcount()
    for _ in tqdm(range(20), desc="Trial"):
        for method in tqdm(["tr + basic", "tr + full", "fr"], desc="Method"):

            if method == "tr + basic":
                g_anneal, _ = pivot_anneal(g_tr.copy())
            elif method == "tr + full":
                g_anneal, _ = pivot_anneal(g_tr2.copy())
            elif method == "fr":
                g_anneal, _ = pivot_anneal(g_fr.copy())
            else:
                raise RuntimeError(f"Invalid method {method}")

            zx.full_reduce(g_anneal)
            c_anneal = zx.extract_circuit(g_anneal.copy()).to_basic_gates()
            c_anneal = zx.basic_optimization(c_anneal)
            trial_score = c_score(c_anneal) # 10 * c_anneal.twoqubitcount() + c_anneal.tcount()
            reduction = (init_score - trial_score) / init_score * 100
            d['score'].append(trial_score)
            d['method'].append(method)
    df = pd.DataFrame(data=d)
    sns.boxplot(x="method", y="score", data=df)
    plt.xlabel("Input Graph")
    plt.ylabel("Total Score")
    plt.show()
    """

    # for actually testing SA
    """
    tr_scores = list()
    fr_scores = list()
    N_TRIALS = 20
    N_ITERS = 1000

    original_score = c_score(c) # 10 * c.twoqubitcount() + c.tcount()
    init_score = c_score(c_tr) # 10 * c_tr.twoqubitcount() + c_tr.tcount()
    init_score2 = c_score(c_tr2) # 10 * c_tr2.twoqubitcount() + c_tr2.tcount()
    init_fr_score = c_score(c_fr) # 10 * c_fr.twoqubitcount() + c_fr.tcount()
    init_just_tr = c_score(c_just_tr) # 10 * c_just_tr.twoqubitcount() + c_just_tr.tcount()
    to_graph_like(g_tr)

    for i in tqdm(range(N_TRIALS)):
        g_anneal, _ = pivot_anneal(g_tr.copy(), iters=N_ITERS, score=g_score)
        zx.full_reduce(g_anneal)
        c_anneal = zx.extract_circuit(g_anneal.copy()).to_basic_gates()
        c_anneal = zx.basic_optimization(c_anneal)

        trial_score = c_score(c_anneal) # 10 * c_anneal.twoqubitcount() + c_anneal.tcount()
        tr_scores.append(trial_score)

        print(f"\n----- trial {i} -----")
        print(c_anneal.stats())

    for i in tqdm(range(N_TRIALS)):
        g_anneal, _ = pivot_anneal(g_fr.copy(), iters=N_ITERS, score=g_score)
        zx.full_reduce(g_anneal)
        c_anneal = zx.extract_circuit(g_anneal.copy()).to_basic_gates()
        c_anneal = zx.basic_optimization(c_anneal)

        trial_score = c_score(c_anneal) # 10 * c_anneal.twoqubitcount() + c_anneal.tcount()
        fr_scores.append(trial_score)

        print(f"\n----- trial {i} -----")
        print(c_anneal.stats())

    all_vals = tr_scores + fr_scores + [original_score, init_score, init_score2, init_fr_score, init_just_tr]
    min_val = min(all_vals) - 5
    max_val = max(all_vals) + 5
    n_bins = ceil((max_val - min_val) / 2)
    range_ = (min_val, max_val)
    print(f"Range: {range_}")
    print(f"# Bins: {n_bins}")

    # plt.hist(reductions)
    # plt.xlabel("Reduction")
    plt.hist(tr_scores, bins=n_bins, range=range_, label="input: tr + basic", alpha=0.5)
    plt.hist(fr_scores, bins=n_bins, range=range_, label="input: fr", alpha=0.5)
    plt.xlabel("Score")
    plt.ylabel("Frequency")

    print(f"Full reduce score: {init_fr_score}")
    print(f"TR + Basic score: {init_score}")
    print(f"TR + Full score: {init_score2}")
    print(f"TR score: {init_just_tr}")
    print(f"Original score: {original_score}")


    plt.axvline(x=init_fr_score, color='r', linestyle='--', label=f"full_reduce ({init_fr_score})", alpha=0.5)
    plt.axvline(x=init_score, color='purple', linestyle='--', label=f"tr + basic ({init_score})", alpha=0.5)
    plt.axvline(x=init_score2, color='green', linestyle='--', label=f"tr + full ({init_score2})", alpha=0.5)
    plt.axvline(x=init_just_tr, color='orange', linestyle='--', label=f"tr ({init_just_tr})", alpha=0.5)
    plt.axvline(x=original_score, color='yellow', linestyle='--', label=f"original ({original_score})", alpha=0.5)
    plt.legend(loc="upper right")
    plt.title(f"{N_QUBITS} qubits, {DEPTH} depth, {N_ITERS} iters, n={N_TRIALS}")
    plt.show()
    print("DONE")
    """
