import numpy as np

from src.specific_interf_sims.models import induce_COs_via_random_forward_draw
from src.specific_interf_sims.interf_schemes import (
    equalised_fixed_regional_interf,
    equalised_global_interf,
    local_interf,
    no_interf,
)
from src.specific_interf_sims.plotter import (
    plot_cM_rf_curve,
    plot_co_per_chrom_dist,
    plot_interevent_dist,
    plot_recomb_curve,
)
from src.specific_interf_sims.simulator import SimulationQueue, queue_sim, run_sims

NUM_CHROMS = 100000
CHROM_LEN = 1000
AVG_COS_PER_CHROMOSOME = 1
STARTING_P_HIT = AVG_COS_PER_CHROMOSOME / CHROM_LEN

queue = SimulationQueue(
    induce_COs_via_random_forward_draw,
    np.zeros(CHROM_LEN) + STARTING_P_HIT,
    rand_seed=54321,
    chroms_per_sim=NUM_CHROMS,
)
no_interf_chroms = queue_sim(queue, 'no interf', no_interf)
local_chroms = queue_sim(queue, 'local', local_interf)
equalised_global_chroms = queue_sim(queue, 'equalised global', equalised_global_interf)
equalised_fixed_regional_chroms = queue_sim(
    queue, 'equalised fixed regional', equalised_fixed_regional_interf
)

run_sims(queue)

for plot_fn in [
    plot_recomb_curve,
    plot_co_per_chrom_dist,
    plot_interevent_dist,
    plot_cM_rf_curve,
]:
    plot_fn(
        no_interf_chroms,
        local_chroms,
        equalised_global_chroms,
        equalised_fixed_regional_chroms,
    )
