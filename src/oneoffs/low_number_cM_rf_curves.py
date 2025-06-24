import numpy as np

from src.randomised_forward_draw.rfd_chromosome import induce_COs_via_random_forward_draw
from src.specific_interf_sims.interf_schemes import fixed_regional_interf, global_interf, local_interf, no_interf
from src.specific_interf_sims.plotter import plot_cM_rf_curve, plot_co_per_chrom_dist, plot_interevent_dist, plot_recomb_curve
from src.specific_interf_sims.simulator import SimulationQueue, queue_sim, run_sims

CHROM_LEN = 1000
AVG_COS_PER_CHROMOSOME = 1
STARTING_P_HIT = AVG_COS_PER_CHROMOSOME / CHROM_LEN

queue = SimulationQueue(
    induce_COs_via_random_forward_draw,
    np.zeros(CHROM_LEN) + STARTING_P_HIT,
    rand_seed=54321
)

hundred_chrom_sets = [
    queue_sim(queue, 'no interf', no_interf, chrom_count=100),
    queue_sim(queue, 'local interf', local_interf, chrom_count=100),
    queue_sim(queue, 'global interf', global_interf, chrom_count=100),
    queue_sim(queue, 'fixed regional interf', fixed_regional_interf, chrom_count=100)
]

thousand_chrom_sets = [
    queue_sim(queue, 'no interf', no_interf, chrom_count=1000),
    queue_sim(queue, 'local interf', local_interf, chrom_count=1000),
    queue_sim(queue, 'global interf', global_interf, chrom_count=1000),
    queue_sim(queue, 'fixed regional interf', fixed_regional_interf, chrom_count=1000)
]

ten_thousand_chrom_sets = [
    queue_sim(queue, 'no interf', no_interf, chrom_count=10000),
    queue_sim(queue, 'local interf', local_interf, chrom_count=10000),
    queue_sim(queue, 'global interf', global_interf, chrom_count=10000),
    queue_sim(queue, 'fixed regional interf', fixed_regional_interf, chrom_count=10000)
]

hundred_thousand_chrom_sets = [
    queue_sim(queue, 'no interf', no_interf, chrom_count=100000),
    queue_sim(queue, 'local interf', local_interf, chrom_count=100000),
    queue_sim(queue, 'global interf', global_interf, chrom_count=100000),
    queue_sim(queue, 'fixed regional interf', fixed_regional_interf, chrom_count=100000)
]

run_sims(queue)

for chrom_sets in [
    hundred_chrom_sets,
    thousand_chrom_sets,
    ten_thousand_chrom_sets,
    hundred_thousand_chrom_sets
]:
    plot_cM_rf_curve(*chrom_sets)
