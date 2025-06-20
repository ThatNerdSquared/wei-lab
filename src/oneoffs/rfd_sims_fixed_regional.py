import numpy as np

from src.randomised_forward_draw.rfd_chromosome import induce_COs_via_random_forward_draw
from src.specific_interf_sims.interf_schemes import fixed_regional_interf, global_interf, local_interf, no_interf
from src.specific_interf_sims.plotter import plot_cM_rf_curve, plot_co_per_chrom_dist, plot_interevent_dist, plot_recomb_curve
from src.specific_interf_sims.simulator import SimulationQueue, queue_sim, run_sims

NUM_CHROMS = 100000
CHROM_LEN = 1000
AVG_COS_PER_CHROMOSOME = 1
STARTING_P_HIT = AVG_COS_PER_CHROMOSOME / CHROM_LEN

queue = SimulationQueue(
    induce_COs_via_random_forward_draw,
    np.zeros(CHROM_LEN) + STARTING_P_HIT,
    rand_seed=54321,
    chroms_per_sim=NUM_CHROMS
)
no_interf_chroms = queue_sim(queue, 'no interf', no_interf)
local_interf_chroms = queue_sim(queue, 'local interf', local_interf)
global_interf_chroms = queue_sim(queue, 'global interf', global_interf)
fixed_regional_interf_chroms = queue_sim(queue, 'fixed regional interf', fixed_regional_interf)

run_sims(queue)

for plot_fn in [
    plot_recomb_curve,
    plot_co_per_chrom_dist,
    plot_interevent_dist,
    plot_cM_rf_curve
]:
    plot_fn(
        no_interf_chroms,
        local_interf_chroms,
        global_interf_chroms,
        fixed_regional_interf_chroms
    )
