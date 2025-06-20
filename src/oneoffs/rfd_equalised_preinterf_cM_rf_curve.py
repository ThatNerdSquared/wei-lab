import numpy as np

from src.randomised_forward_draw.rfd_chromosome import induce_COs_via_random_forward_draw
from src.specific_interf_sims.interf_schemes import equalised_fixed_regional_interf, equalised_global_interf, local_interf, no_interf
from src.specific_interf_sims.plotter import plot_preinterf_cM_rf_curve
from src.specific_interf_sims.simulator import SimulationQueue, queue_sim, run_sims

CHROM_LEN = 1000
AVG_COS_PER_CHROMOSOME = 1
STARTING_P_HIT = AVG_COS_PER_CHROMOSOME / CHROM_LEN

starting_bins = np.zeros(CHROM_LEN) + STARTING_P_HIT
queue = SimulationQueue(
    induce_COs_via_random_forward_draw,
    default_starting_bins=starting_bins,
    rand_seed=54321,
    chroms_per_sim=100000
)
no_interf_chroms = queue_sim(queue, 'no interf', no_interf)
local_chroms = queue_sim(queue, 'local', local_interf)
equalised_global_chroms = queue_sim(queue, 'equalised global', equalised_global_interf)
equalised_fixed_regional_chroms = queue_sim(queue, 'equalised fixed regional', equalised_fixed_regional_interf)

run_sims(queue)

plot_preinterf_cM_rf_curve(
    starting_bins,
    no_interf_chroms,
    local_chroms,
    equalised_global_chroms,
    equalised_fixed_regional_chroms
)
