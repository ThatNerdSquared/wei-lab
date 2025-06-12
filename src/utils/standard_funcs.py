from math import exp, gamma, log

from numpy import tanh

def poisson(x, lam) -> float:
    return ((lam ** x) * exp(-1 * lam)) / gamma(x + 1)

def haldane_input_cM(x, p: int = 2) -> float:
    return (1 - exp(-1 * p * x)) / p

def haldane_input_rf(x, p: int = 2) -> float:
    return -1 * (1 / p) * log(1 - p * x)

def kosambi(x: float) -> float:
    return tanh(2 * x) / 2
