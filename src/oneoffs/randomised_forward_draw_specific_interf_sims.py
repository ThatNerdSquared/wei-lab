import numpy as np
import matplotlib.pyplot as plt

from src.randomised_forward_draw.rfd_chromosome import BinArray, induce_COs_via_random_forward_draw
from src.utils.chromosome_set import ChromosomeSet, gather_co_per_chrom_dist_data, gather_interevent_dist_data, gather_recomb_curve_data, gather_rf_cM_curve_data
from src.utils.standard_funcs import haldane_input_cM

NUM_CHROMOSOMES = 100000
BINS_PER_CHROMOSOME = 1000
# AVG_COS_PER_CHROMOSOME = 0.5
AVG_COS_PER_CHROMOSOME = 1
STARTING_P_HIT = AVG_COS_PER_CHROMOSOME / BINS_PER_CHROMOSOME
RAND_SEED = 54321

def no_interf(_: int, bins: BinArray):
    return bins

def basic_interf(co_loc: int, bins: BinArray):
    bin_locs = np.arange(len(bins))
    return bins * (1 - np.exp(-1 * np.abs(
        (bin_locs / BINS_PER_CHROMOSOME) - (co_loc / BINS_PER_CHROMOSOME)
    )))

no_interf_chroms = ChromosomeSet(
    num_chroms=NUM_CHROMOSOMES,
    chrom_length=BINS_PER_CHROMOSOME,
    rand_seed=RAND_SEED
)

basic_interf_chroms = ChromosomeSet(
    num_chroms=NUM_CHROMOSOMES,
    chrom_length=BINS_PER_CHROMOSOME,
    rand_seed=RAND_SEED
)

induce_COs_via_random_forward_draw(
    no_interf_chroms,
    lambda x: np.zeros(len(x)) + STARTING_P_HIT,
    no_interf
)

induce_COs_via_random_forward_draw(
    basic_interf_chroms,
    lambda x: np.zeros(len(x), dtype=np.float64) + STARTING_P_HIT,
    basic_interf
)

# no_interf_recomb_curve = gather_recomb_curve_data(no_interf_chroms)
# basic_interf_recomb_curve = gather_recomb_curve_data(basic_interf_chroms)
# no_interf_co_per_chrom_dist = gather_co_per_chrom_dist_data(no_interf_chroms)
# basic_interf_co_per_chrom_dist = gather_co_per_chrom_dist_data(
#     basic_interf_chroms
# )
# no_interf_totals, no_interf_groups = gather_interevent_dist_data(no_interf_chroms)
# basic_interf_totals, basic_interf_groups = gather_interevent_dist_data(basic_interf_chroms)
no_interf_cM, no_interf_rf = gather_rf_cM_curve_data(no_interf_chroms)
basic_interf_cM, basic_interf_rf = gather_rf_cM_curve_data(basic_interf_chroms)
max_y = np.max([
    # np.max(no_interf_recomb_curve),
    # np.max(basic_interf_recomb_curve)
    # np.max(no_interf_co_per_chrom_dist),
    # np.max(basic_interf_co_per_chrom_dist)
    # np.max(no_interf_totals),
    # np.max(basic_interf_totals)
    np.max(no_interf_rf),
    np.max(basic_interf_rf)
])
max_y = max_y * 1.1

fig, ax = plt.subplots(sharex=True, sharey=True, figsize=(9.0, 7.0))
# ax.plot(no_interf_recomb_curve, label='no interf')
# ax.plot(basic_interf_recomb_curve, label='interf')
# ax.stairs(no_interf_co_per_chrom_dist, label='no interf')
# ax.stairs(basic_interf_co_per_chrom_dist, label='interf')
# ax.stairs(no_interf_totals, no_interf_groups, label='no interf')
# ax.stairs(basic_interf_totals, basic_interf_groups, label='interf')
x = np.linspace(0, 2, 1000)
ax.plot(x, [haldane_input_cM(n) for n in x], label='haldane')
ax.plot(no_interf_cM, no_interf_rf, label='no interf')
ax.plot(basic_interf_cM, basic_interf_rf, label='interf')
# ax.plot(
#     no_interf_interevent_dist.keys(),
#     no_interf_interevent_dist.values(),
#     label='no interf'
# )
# ax.plot(
#     basic_interf_interevent_dist.keys(),
#     basic_interf_interevent_dist.values(),
#     label='interf'
# )
# ax.set_ylim((0, max_y))
ax.set_ylim((0, 1))
plt.legend()
plt.show()
