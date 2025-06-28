import numpy as np
import collections
from random import randrange


def generate_poisson(rf: float) -> np.ndarray:
    rng = np.random.default_rng()
    return rng.poisson(lam=rf, size=100)


avg_p_co_total = 0.5
CO_distribution: dict[int, int] = {}
totals: dict[int, int] = {}
totals_only_2COs: dict[int, int] = {}

for i in range(1000):
    chromosomes = generate_poisson(avg_p_co_total)
    chromosomes_with_windows = []
    for chrom in chromosomes:
        this_chrom = []
        for i in range(chrom):
            window = randrange(10)
            this_chrom.append(window)
        chromosomes_with_windows.append(this_chrom)

    for c in chromosomes_with_windows:
        try:
            CO_distribution[len(c)] += 1
        except KeyError:
            CO_distribution[len(c)] = 0

        for window in c:
            try:
                totals[window] += 1
            except KeyError:
                totals[window] = 0

        if len(c) == 2:
            for window in c:
                try:
                    totals_only_2COs[window] += 1
                except KeyError:
                    totals_only_2COs[window] = 0

CO_distribution = dict(sorted(CO_distribution.items()))
totals = dict(sorted(totals.items()))
totals_only_2COs = dict(sorted(totals_only_2COs.items()))

print(f'Distribution of COs per chromosome: {CO_distribution}')
print(f'Num of COs per window: {totals}')
print(f'Num of COs per window, for chromosmes with 2 COs: {totals_only_2COs}')
