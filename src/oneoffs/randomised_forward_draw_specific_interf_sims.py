from typing import cast
import numpy as np

from src.randomised_forward_draw.rfd_chromosome import induce_COs_via_random_forward_draw
from src.specific_interf_sims.plotter import plot_cM_rf_curve, plot_co_per_chrom_dist, plot_interevent_dist, plot_recomb_curve
from src.specific_interf_sims.simulator import BinArray, SimulationQueue, queue_sim, run_sims

NUM_CHROMOSOMES = 100000
BINS_PER_CHROMOSOME = 1000
AVG_COS_PER_CHROMOSOME = 1
STARTING_P_HIT = np.float64(AVG_COS_PER_CHROMOSOME / BINS_PER_CHROMOSOME)
RAND_SEED = 54321

def no_interf(_: int, bins: BinArray) -> BinArray:
    return bins

def local_interf(co_loc: int, bins: BinArray) -> BinArray:
    bin_locs = np.arange(len(bins))
    return bins * (1 - np.exp(-1 * np.abs(
        (bin_locs / BINS_PER_CHROMOSOME) - (co_loc / BINS_PER_CHROMOSOME)
    )))

def global_interf(co_loc: int, bins: BinArray) -> BinArray:
    return bins * 0.7

def equalised_global_interf(co_loc: int, bins: BinArray) -> BinArray:
    new_total_p_co = np.sum(local_interf(co_loc, bins))
    current_total_p_co = np.sum(bins)
    return bins * new_total_p_co / current_total_p_co

def regional_interf(co_loc: int, bins: BinArray) -> BinArray:
    bin_locs = np.arange(len(bins))
    return bins * (
        np.where(bin_locs <= co_loc, 0, 1)
        if co_loc < len(bins) / 2 else
        np.where(bin_locs >= co_loc, 0, 1)
    )


default_bins = cast(
    BinArray,
    np.zeros(BINS_PER_CHROMOSOME, dtype=np.float64) + STARTING_P_HIT
)
queue = SimulationQueue(
    induce_COs_via_random_forward_draw,
    default_bins,
    rand_seed=RAND_SEED,
    chroms_per_sim=NUM_CHROMOSOMES
)
print('setting up chromosomes...')
no_interf_chroms = queue_sim(queue, 'no interf', no_interf)
local_interf_chroms = queue_sim(queue, 'local interf', local_interf)
global_interf_chroms = queue_sim(queue, 'global interf', global_interf)
regional_interf_chroms = queue_sim(queue, 'regional interf', regional_interf)

run_sims(queue)

plot_recomb_curve(
    no_interf_chroms,
    local_interf_chroms,
    global_interf_chroms,
    regional_interf_chroms
)

plot_co_per_chrom_dist(
    no_interf_chroms,
    local_interf_chroms,
    global_interf_chroms,
    regional_interf_chroms
)

plot_interevent_dist(
    no_interf_chroms,
    local_interf_chroms,
    global_interf_chroms,
    regional_interf_chroms
)

plot_cM_rf_curve(
    no_interf_chroms,
    local_interf_chroms,
    global_interf_chroms,
    regional_interf_chroms
)
