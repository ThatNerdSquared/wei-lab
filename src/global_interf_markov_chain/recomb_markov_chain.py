import numpy as np
import math
from collections.abc import Callable
from textwrap import dedent
from typing import Optional

class RecombMarkovChain:

    def __init__(self, n_states: int = 1):
        self._transition_matrix: np.ndarray[np.ndarray[np.float64]] = np.zeros((n_states, n_states))

    def __str__(self) -> str:
        segments: list[str] = []
        for state in range(len(self._transition_matrix)):
            segments.append(dedent(f'''
                ┌──────┐
                │  {str(state).rjust(2)}  │
                └──────┘
                     ^──
            '''))
            if state is not len(self._transition_matrix) - 1:
                segments.append(dedent(f'''
                      {str(round(self._transition_matrix[state][state + 1], 6)).center(8, ' ')}  
                    ───────────>
                    ─┐          
                    ─┘ {str(round(self._transition_matrix[state][state], 6)).center(8, ' ')} 
                '''))
            else:
                segments.append(dedent(f'''


                    ─┐          
                    ─┘ {str(round(self._transition_matrix[state][state], 6)).center(8, ' ')} 
                '''))
        return "\n".join(map(''.join, zip(*[x.split('\n') for x in segments])))

    def reset_to_poisson(self, p_zero_to_zero: float):
        lam: float = -1 * math.log(p_zero_to_zero)
        for k in range(len(self._transition_matrix)):
            if k == 0:
                self._transition_matrix[k][k] = p_zero_to_zero
            elif k == len(self._transition_matrix) - 1:
                self._transition_matrix[k][k] = 1
            else:
                prev_transition_product = self._superdiagonal_partial_product(k)
                p_k_to_k = (((lam ** k) * p_zero_to_zero) / math.factorial(k)) * (1 / prev_transition_product)
                self._transition_matrix[k][k] = p_k_to_k
            try:
                self._transition_matrix[k][k + 1] = 1 - self._transition_matrix[k][k]
            except IndexError:
                pass

    def _superdiagonal_partial_product(self, k: int) -> np.float64:
        product = np.float64(1.0)
        for i in reversed(range(k)):
            product *= self._transition_matrix[i][i+1]
        return product

    def get_p_cos(self, k: int) -> np.float64:
        return self._transition_matrix[k][k] * self._superdiagonal_partial_product(k)

    def perturb_transition_probability(self, start: int, end: int, delta: float):
        if end - start > 1:
            raise ValueError("Given start and end states are not connected!")
        if (
            self._transition_matrix[start][end] + delta > 1 or
            self._transition_matrix[start][end] + delta < 0
        ):
            raise ValueError("Delta results in out-of-bounds probability!")
        self._transition_matrix[start][end] += delta
        if start == end:
            self._transition_matrix[start][end + 1] -= delta
        else:
            self._transition_matrix[start][end - 1] -= delta

    def scale_transition_probability(self, start: int, end: int, multiplier: float):
        if end - start > 1:
            raise ValueError("Given start and end states are not connected!")
        if start == end:
            raise NotImplementedError(
                "This method doesn't yet support scaling k->k transitions. Please try scaling a k->k+1 transition instead."
            )
        self._transition_matrix[start][end] *= multiplier
        self._transition_matrix[start][end - 1] = 1 -self._transition_matrix[start][end]

    def on_all_onward_transitions(
        self,
        fn: Callable[[int, int, float], None],
        arg: float,
        start: Optional[int] = None
    ):
        for k in range(start if start else 0, len(self._transition_matrix)):
            if k == len(self._transition_matrix) - 1:
                return
            fn(k, k+1, arg)
