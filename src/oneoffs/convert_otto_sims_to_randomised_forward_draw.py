import numpy as np
import matplotlib.pyplot as plt

from typing import Callable
from src.utils.chromosome import Chromosome

# config here is set to match otto's simulation
NUM_CHROMOSOMES = 100000
BINS_PER_CHROMOSOME = 1000
# NEW: standardised map length
AVG_COS_PER_CHROMOSOME = 0.5

# rand = np.random.default_rng(77777)
rand = np.random.default_rng(54321)
starting_p_hit = AVG_COS_PER_CHROMOSOME / BINS_PER_CHROMOSOME


def simulate_bins(react_to_co: Callable[[int, list, list[int]], None]):
    chroms = []
    for i in range(NUM_CHROMOSOMES):
        print(i)
        chrom = Chromosome(BINS_PER_CHROMOSOME)
        bins = np.zeros(BINS_PER_CHROMOSOME)
        bins += starting_p_hit

        bin_idxs = list(range(BINS_PER_CHROMOSOME))
        # bin_idxs = np.arange(BINS_PER_CHROMOSOME)
        rand.shuffle(bin_idxs)

        for loc in bin_idxs:
            if rand.binomial(1, bins[loc]):
                react_to_co(loc, bins, bin_idxs)
                chrom.insert_crossover(loc)

        chroms.append(chrom)

    return chroms

no_interf_chroms = simulate_bins(lambda x, y, z: None)

# basic interference
def interf_scaling_factor(bin_index, co_location) -> float:
    return 1 - np.exp(-1 * np.abs(
        (bin_index / BINS_PER_CHROMOSOME) - (co_location / BINS_PER_CHROMOSOME)
    ))


def basic_interf_reaction(co_loc, bins, bin_idxs):
    for idx in bin_idxs:
        bins[idx] *= interf_scaling_factor(idx, co_loc)


interf_chroms = simulate_bins(basic_interf_reaction)

fig, ax = plt.subplots(sharex=True, sharey=True, figsize=(9.0, 7.0))
total_cos_per_location__default = np.zeros(BINS_PER_CHROMOSOME)
total_cos_per_location__interf = np.zeros(BINS_PER_CHROMOSOME)
x_range = list(range(BINS_PER_CHROMOSOME))

for x in x_range:
    total_cos_per_location__default[x] = sum([1 if y.has_crossover_at(x) else 0 for y in no_interf_chroms]) / NUM_CHROMOSOMES
    total_cos_per_location__interf[x] = sum([1 if y.has_crossover_at(x) else 0 for y in interf_chroms]) / NUM_CHROMOSOMES

ax.plot(x_range, total_cos_per_location__default, label='no interf')
ax.plot(x_range, total_cos_per_location__interf, label='interf')
ax.set_ylim((0, 1))
# ax.set_ylim([0, 1])
plt.show()
