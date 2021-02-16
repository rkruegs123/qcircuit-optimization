from abc import ABC, abstractmethod

import sys
sys.path.append('../pyzx')
import pyzx as zx

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
