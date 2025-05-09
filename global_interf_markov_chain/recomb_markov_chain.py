import numpy as np
import math

class RecombMarkovChain:

    def __init__(self, n_states: int = 1):
        self._transition_matrix: np.ndarray[np.ndarray[np.float64]] = np.zeros((n_states, n_states))

    def __str__(self):
        return str(self._transition_matrix)

    def reset_to_poisson(self, p_zero_to_zero: float):
        lam: float = -1 * math.log(p_zero_to_zero)
        for k in range(len(self._transition_matrix)):
            if k == 0:
                self._transition_matrix[k][k] = p_zero_to_zero
            else:
                prev_transition_product = self._superdiagonal_partial_product(k)
                p_k_to_k = (((lam ** k) * p_zero_to_zero) / math.factorial(k)) * (1 / prev_transition_product)
                self._transition_matrix[k][k] = p_k_to_k
            try:
                self._transition_matrix[k][k + 1] = 1 - self._transition_matrix[k][k]
            except IndexError:
                pass

    def _superdiagonal_partial_product(self, k: int) -> np.float64:
        if k == 0:
            return np.float64(0.0)
        product = np.float64(1.0)
        for i in reversed(range(k)):
            product *= self._transition_matrix[i][i+1]
        return product
