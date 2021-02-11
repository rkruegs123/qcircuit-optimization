from abc import ABC, abstractmethod
import pyzx as zx
from os import listdir
from os.path import isfile, join
from tqdm import tqdm
import matplotlib.pyplot as plt
import pdb

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

class FullCircuitOptimize(Optimizer):

    def _optimize(self, c):
        return zx.full_optimize(c)

    @property
    def name(self):
        return f"full_optimize (PyZX)"

    @property
    def desc(self):
        return "PyZX's most general circuit optimization procedure"


# Basic circuit optimization
OPTIMIZERS = [
    QiskitTranspiler(opt_level=0),
    QiskitTranspiler(opt_level=1)
]



MAX_TCOUNT = 50
MAX_2QUBIT_COUNT = 50
MAX_QUBITS = 10
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






# Note: Could do evolutionary approach with qiskit "passes" as well
if __name__ == "__main__":
    benchdir = "bench"
    fs = [f for f in listdir(benchdir) if isfile(join(benchdir, f))]
    cs = [zx.Circuit.load(join(benchdir, f)).to_basic_gates() for f in fs]
    print(f"Loaded {len(cs)} circuits...")

    # plot_circ_info([('all', cs)])
    # plt.show()


    small_cs = [c for c in cs if is_small(c)]
    print(f"Loaded {len(small_cs)} small circuits...")
    full_circ_opt = FullCircuitOptimize()

    css = [('original', small_cs)]
    opts = [full_circ_opt.optimize(c) for c in small_cs]
    css.append((full_circ_opt.name, opts))
    plot_circ_info(css)
    plt.show()
