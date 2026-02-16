from abc import ABC, abstractmethod
from typing import override

import numpy as np


class AssuranceScheme(ABC):
    @abstractmethod
    def handle_co_failed(self): ...

    @abstractmethod
    def handle_co_succeeded(self): ...

    @abstractmethod
    def modify_co_probability(self, p: np.float64) -> np.float64: ...

    @abstractmethod
    def reset(self): ...


class LastBinWinsAssurance(AssuranceScheme):
    num_bins: int
    num_failed_COs: int

    def __init__(self, num_bins: int):
        self.num_bins = num_bins
        self.num_failed_COs = 0

    @override
    def handle_co_failed(self):
        self.num_failed_COs += 1

    @override
    def handle_co_succeeded(self):
        self.reset()

    @override
    def modify_co_probability(self, p: np.float64) -> np.float64:
        return p if self.num_failed_COs < self.num_bins - 1 else np.float64(1.0)

    @override
    def reset(self):
        self.num_failed_COs = 0


class LinearPotentialAssurance(AssuranceScheme):
    num_bins: int
    num_failed_COs: int

    def __init__(self, num_bins: int):
        self.num_bins = num_bins
        self.num_failed_COs = 0

    @override
    def handle_co_failed(self):
        self.num_failed_COs += 1

    @override
    def handle_co_succeeded(self):
        self.reset()

    @override
    def modify_co_probability(self, p: np.float64) -> np.float64:
        # return p + (self.num_failed_COs / (self.num_bins * 10))
        return p + (self.num_failed_COs / self.num_bins)

    @override
    def reset(self):
        self.num_failed_COs = 0
