import numpy as np
from math import exp, gamma, log

from numpy import tanh

def poisson(x: float, lam: float) -> float:
    return ((lam ** x) * exp(-1 * lam)) / gamma(x + 1)

def haldane_input_cM(x: float, p: int = 2) -> float:
    return (1 - np.exp(-1 * p * x)) / p

def haldane_input_rf(x: float, p: int = 2) -> float:
    return -1 * (1 / p) * log(1 - p * x)

def kosambi(x: float) -> float:
    return tanh(2 * x) / 2
