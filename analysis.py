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

from utilities import to_graph_like

# import optimizers
from baseline_optimizers import QiskitTranspiler, TketMinCX, TketFullPeephole, FullOptimize, \
    FullReduce, TeleportReduce, TketBasic
from sa_optimizers import AleksSA
import ga_optimizers as ga
import anneal as sa

MAX_TCOUNT = 100
MAX_2QUBIT_COUNT = 100
MAX_QUBITS = 20
MAX_NGATES = 100

def is_small(c):
    return c.qubits <= MAX_QUBITS and c.tcount() <= MAX_TCOUNT \
        and c.twoqubitcount() <= MAX_2QUBIT_COUNT and len(c.gates) <= MAX_NGATES


def plot_circ_info(css):

    NBINS = 10
    ALPHA = 0.5
    plt.rcParams['savefig.facecolor'] = "0.8"
    plt.close('all')
    fig = plt.figure()
    fig.set_figheight(10)
    fig.set_figwidth(10)


    ax1 = plt.subplot2grid((2, 2), (0, 0))
    ax2 = plt.subplot2grid((2, 2), (0, 1), colspan=1)
    ax3 = plt.subplot2grid((2, 2), (1, 0), colspan=1, rowspan=1)
    ax4 = plt.subplot2grid((2, 2), (1, 1), rowspan=1)

    for (cs_name, cs) in css:
        twoqubitcounts = [c.twoqubitcount() for c in cs]
        ax1.hist(twoqubitcounts, alpha=ALPHA, label=cs_name)

        tcounts = [c.tcount() for c in cs]
        ax2.hist(tcounts, alpha=ALPHA, label=cs_name)

        qubits = [c.qubits for c in cs]
        ax3.hist(qubits, alpha=ALPHA, label=cs_name)

        gates = [len(c.gates) for c in cs]
        ax4.hist(gates, alpha=ALPHA, label=cs_name)

    # set titles and locator params
    def config_ax(ax, nbins, xlabel="", ylabel="", title="", fontsize=12):
        ax.locator_params(nbins=nbins)
        ax.set_xlabel(xlabel, fontsize=fontsize)
        ax.set_ylabel(ylabel, fontsize=fontsize)
        ax.set_title(title, fontsize=fontsize)
        ax.legend(loc='upper right')

    config_ax(ax1, NBINS, title="2-qubit counts")
    config_ax(ax2, NBINS, title="T-counts")
    config_ax(ax3, NBINS, title="No. qubits")
    config_ax(ax4, NBINS, title="No. gates")

    plt.tight_layout()


def compute_reductions(original_cs, css, ax, val=("2-qubit-counts", lambda c: c.twoqubitcount())):

    val_name, val_func = val

    original_vals = [val_func(c) for c in original_cs]
    d = {'reduction': list(), 'method': list()}
    for (cs_name, cs) in css:
        cs_vals = [val_func(c) for c in cs]
        reductions = [(orig_val - opt_val) / orig_val * 100
                      for (orig_val, opt_val) in zip(original_vals, cs_vals)]
        d['reduction'] += reductions
        d['method'] += [cs_name] * len(cs)
    df = pd.DataFrame(data=d)

    sns.boxplot(x="method", y="reduction", data=df, ax=ax)
    ax.set(ylim=(-100, 50))
    ax.set_title(f"{val_name} reductions")


# Note: Could do evolutionary approach with qiskit "passes" as well
if __name__ == "__main__":

    """

    # cs_dir = "circuits/CNOT_HAD_PHASE/qubits_10-20_depth_50-100"
    cs_dir = "circuits/bench"
    fs = [f for f in listdir(cs_dir) if isfile(join(cs_dir, f))]
    # fs = fs[:10]
    cs = [zx.Circuit.load(join(cs_dir, f)).to_basic_gates() for f in fs]
    print(f"Loaded {len(cs)} circuits...")


    test_cs = [c for c in cs if is_small(c)]
    print(f"Loaded {len(test_cs)} small circuits...")
    # Currently not including TketFullPeephole()
    optimizers = [FullOptimize(), FullReduce(), TketBasic(), TketMinCX(), \
                  QiskitTranspiler(opt_level=2), TeleportReduce()]
    # NOTE: pyzx qasm parser doesn't appropriately handle -pi, so opt_level=3 fails
    # optimizers = [AleksSA()]

    css = [(opt.name, [opt.optimize(c) for c in tqdm(test_cs)]) for opt in tqdm(optimizers)]
    # plot_circ_info([('original', test_cs)] + css)
    # plt.close('all')

    plt.rcParams['font.size'] = 8
    fig = plt.figure()
    fig.set_figheight(10)
    fig.set_figwidth(20)
    ax1 = plt.subplot2grid((2, 2), (0, 0))
    ax2 = plt.subplot2grid((2, 2), (0, 1))
    ax3 = plt.subplot2grid((2, 2), (1, 0), colspan=2)

    compute_reductions(test_cs, css, ax1, val=("T count", lambda c: c.tcount()))
    compute_reductions(test_cs, css, ax2, val=("Total Gates", lambda c: len(c.gates)))
    compute_reductions(test_cs, css, ax3)
    plt.tight_layout()
    plt.show()
    """



    # Compare GA vs SA


    N_QUBITS = 9
    DEPTH = 100

    c = zx.generate.CNOT_HAD_PHASE_circuit(qubits=N_QUBITS, depth=DEPTH, clifford=False)
    g = c.to_graph()

    g_tr = g.copy()
    zx.teleport_reduce(g_tr)
    c_tr = zx.Circuit.from_graph(g_tr.copy())
    c_tr = zx.basic_optimization(c_tr).to_basic_gates()
    g_tr = c_tr.to_graph()
    to_graph_like(g_tr)

    # fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(5, 3))
    fig, axes = plt.subplots(nrows=1, ncols=2)


    ## GA part

    N_MUTANTS = 100
    N_GENS = 100
    actions = [ga.rand_pivot, ga.rand_lc, ga.do_nothing]
    # actions = [rand_lc, rand_pivot]
    ga_opt = ga.GeneticOptimizer(actions, n_generations=N_GENS, n_mutants=N_MUTANTS, quiet=False)
    orig_score = ga.default_score(ga.Mutant(c_tr, g_tr))
    print(f"Original score: {orig_score}")
    scores, c_opt = ga_opt.evolve(c_tr.copy())
    reductions = [(orig_score - gen_score) / orig_score * 100 for gen_score in scores]

    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # ax.plot(list(range(len(scores))), scores)
    # ax.axhline(orig_score, label="original", alpha=0.5, linestyle="--")
    ax.plot(list(range(len(reductions))), reductions)
    plt.xlabel("Generation")
    plt.ylabel("Reduction")
    plt.title(f"GA: {N_QUBITS} qubits, {DEPTH} depth, {N_MUTANTS} mutants, {N_GENS} generations")
    plt.legend()
    plt.show()
    plt.close('all')
    """

    axes[0].plot(list(range(len(reductions))), reductions)
    axes[0].set_xlabel("Generation")
    axes[0].set_ylabel("Reduction")
    axes[0].set_title(f"GA: {N_MUTANTS} mutants, {N_GENS} generations")
    axes[0].legend()



    ## SA part

    reductions = list()
    N_TRIALS = 10
    N_ITERS = 1000

    init_score = 4 * c_tr.twoqubitcount() + c_tr.tcount()
    for i in tqdm(range(N_TRIALS)):
        g_anneal, _ = sa.pivot_anneal(g_tr.copy(), iters=N_ITERS, score=sa.c_score)
        zx.full_reduce(g_anneal)
        c_anneal = zx.extract_circuit(g_anneal.copy()).to_basic_gates()
        c_anneal = zx.basic_optimization(c_anneal)

        trial_score = 4 * c_anneal.twoqubitcount() + c_anneal.tcount()
        reduction = (init_score - trial_score) / init_score * 100
        reductions.append(reduction)

        # print(f"\n----- trial {i} -----")
        # print(c_anneal.stats())


    """
    plt.hist(reductions)
    plt.xlabel("Reduction")
    plt.ylabel("Frequency")
    plt.title(f"SA: {N_TRIALS} trials, {N_ITERS} iters")
    """


    axes[1].hist(reductions)
    axes[1].set_xlabel("Reduction")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title(f"SA: {N_TRIALS} trials, {N_ITERS} iters")
    plt.suptitle(f"{N_QUBITS} qubits, {DEPTH} depth") # , fontsize=14)
    plt.show()

    print("DONE")
