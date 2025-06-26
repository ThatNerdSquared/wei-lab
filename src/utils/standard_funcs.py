import numpy as np
from scipy.special import loggamma

def poisson(x: float, lam: float) -> float:
    return np.exp(x * np.log(lam) - lam - loggamma(x + 1))

def haldane_input_cM(x: float, p: int = 2) -> float:
    return (1 - np.exp(-1 * p * x)) / p

def haldane_input_rf(x: float, p: int = 2) -> float:
    return -1 * (1 / p) * np.log(1 - p * x)

def kosambi(x: float) -> float:
    return np.tanh(2 * x) / 2
