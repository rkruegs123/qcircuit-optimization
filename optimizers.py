from abc import ABC, abstractmethod
import pyzx as zx
from os import listdir
from os.path import isfile, join
from tqdm import tqdm
import matplotlib.pyplot as plt
import pdb
from pytket.extensions.pyzx import pyzx_to_tk, tk_to_pyzx
from pytket.passes import RemoveRedundancies
import seaborn as sns
import pandas as pd

class Optimizer(ABC):

    def __init__(self):
        super().__init__()

    def optimize(self, c):
        assert(isinstance(c, zx.Circuit))
        c_opt = self._optimize(c)
        assert(isinstance(c_opt, zx.Circuit))
        assert(c.verify_equality(c_opt))
        return c_opt

    @abstractmethod
    def _optimize(self, c):
        pass

    @property
    @abstractmethod
    def name(self):
        pass

    @property
    @abstractmethod
    def desc(self):
        pass

# FIXME: Not functional
class QiskitTranspiler(Optimizer):

    def __init__(self, opt_level):
        self.opt_level = opt_level
        super().__init__()

    def _optimize(self, c):
        return c
        # qiskit.transpile(c, opt_level=self.opt_level) # FIXME: just for show

    @property
    def name(self):
        return f"Qiskit Transpiler (level={self.opt_level})"

    @property
    def desc(self):
        return "Qiskit's default transpiler. Makes use of various individual Transpiler Passes."

class TketBasic(Optimizer):
    def _optimize(self, c):
        c_tket = pyzx_to_tk(c)
        RemoveRedundancies().apply(c_tket)
        c_opt = tk_to_pyzx(c_tket)
        return c_opt

    @property
    def name(self):
        return f"RemoveRedundancies (tket)"

    @property
    def desc(self):
        return "Tket's most basic optimization method"

class FullCircuitOptimize(Optimizer):

    def _optimize(self, c):
        return zx.full_optimize(c)

    @property
    def name(self):
        return f"full_optimize (PyZX)"

    @property
    def desc(self):
        return "PyZX's most general circuit optimization procedure"


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


def compute_reductions(original_cs, css, val=("2-qubit-counts", lambda c: c.twoqubitcount())):

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

    plt.close('all')
    ax = sns.boxplot(x="method", y="reduction", data=df)
    ax.set_title(f"{val_name} reductions")

# Note: Could do evolutionary approach with qiskit "passes" as well
if __name__ == "__main__":
    # cs_dir = "circuits/bench"
    cs_dir = "circuits/CNOT_HAD_PHASE/qubits_10-20_depth_50-100"
    fs = [f for f in listdir(cs_dir) if isfile(join(cs_dir, f))]
    cs = [zx.Circuit.load(join(cs_dir, f)).to_basic_gates() for f in fs]
    print(f"Loaded {len(cs)} circuits...")


    test_cs = [c for c in cs if is_small(c)]
    print(f"Loaded {len(test_cs)} small circuits...")
    optimizers = [FullCircuitOptimize(), TketBasic()]

    css = [(opt.name, [opt.optimize(c) for c in tqdm(test_cs)]) for opt in tqdm(optimizers)]
    # plot_circ_info([('original', test_cs)] + css)
    compute_reductions(test_cs, css)

    plt.show()
