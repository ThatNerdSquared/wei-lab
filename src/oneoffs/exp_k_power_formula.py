import collections
import functools
import operator
from typing import cast

import numpy as np


def p_co_total(p_single_co: float) -> float:
    return (1 / (1 - p_single_co)) - 1


def generate_poisson(rf: float) -> np.ndarray:
    rng = np.random.default_rng()
    return rng.poisson(lam=rf, size=100)


avg_p_co_total = 0.3

sims: list[dict[str, float]] = []
num_sims = 1000
num_sims_w_2cos = 0

for i in range(num_sims):
    chromosomes = generate_poisson(avg_p_co_total)
    counter = collections.Counter(chromosomes)
    oneco_chance = counter[1] / len(chromosomes)
    twoco_chance = counter[2] / len(chromosomes)
    p_2_percent_diff = 0.0
    if twoco_chance != 0:
        est_p_2 = oneco_chance**2
        p_2_percent_diff = abs(est_p_2 - twoco_chance) / twoco_chance
        num_sims_w_2cos += 1
    sim_res = {
        'est_p_co_total': p_co_total(oneco_chance),
        'est_p_2_diff': p_2_percent_diff,
    }
    sims.append(sim_res)


summed_sims = cast(
    dict[str, float], functools.reduce(operator.add, map(collections.Counter, sims))
)
estimated_avg = summed_sims['est_p_co_total'] / num_sims
percent_diff_p_co_total = abs(avg_p_co_total - estimated_avg) / avg_p_co_total
p_2_diff = summed_sims['est_p_2_diff'] / num_sims_w_2cos
print(f'Actual avg P(co)-total: {avg_p_co_total}')
print(f'Estimated avg P(co)-total: {estimated_avg}')
print(f'% diff: {percent_diff_p_co_total * 100}%')
print(f'Avg % diff between actual and estimated P(2CO): {p_2_diff * 100}%')
