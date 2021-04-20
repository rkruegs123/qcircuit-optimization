from base_optimizer import Optimizer

# congruences
from anneal import pivot_anneal


class AleksSA(Optimizer):
    def _optimize(self, c):
        g = c.to_graph()
        zx.full_reduce(g, quiet=True)
        g.normalize()
        gs = pivot_anneal(g, iters=5, cool=0.0025, score=self.twoq_score)
        c_opt = zx.extract_circuit(gs)
        c_opt = self.post_process(c_opt)
        return c_opt

    def twoq_score(self, g):
        c = zx.extract_circuit(g.copy())
        c = self.post_process(c)
        return c.twoqubitcount()

    def post_process(self, c):
        c = c.split_phase_gates().to_basic_gates()
        return zx.optimize.basic_optimization(c).to_basic_gates()

    @property
    def name(self):
        return f"anneal (AK, pivot)"

    @property
    def desc(self):
        return "Aleks' original annealing implementation"
