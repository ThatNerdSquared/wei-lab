import numpy as np
from croissant.chromosomes import make_default_chromset

from src.specific_interf_sims.data_collectors import collect_interevent_dist_data
from src.specific_interf_sims.data_transformers import transform_interevent_dist_data
from src.specific_interf_sims.interf_schemes import (
    local_interf,
    no_interf,
)
from src.specific_interf_sims.models import induce_COs_via_random_forward_draw
from src.specific_interf_sims.plotter import (
    save_plot,
    standard_plotter,
)
from src.specific_interf_sims.simulator import SimulationQueue, queue_sim, run_sims

NUM_CHROMS = 100000
CHROM_LEN = 1000
AVG_COS_PER_CHROMOSOME = 2
STARTING_P_HIT = AVG_COS_PER_CHROMOSOME / CHROM_LEN

queue = SimulationQueue(
    induce_COs_via_random_forward_draw,
    np.zeros(CHROM_LEN) + STARTING_P_HIT,
    rand_seed=54321,
    chroms_per_sim=NUM_CHROMS,
    default_chromset_maker=make_default_chromset,
)
no_interf_chroms = queue_sim(queue, 'no interf', no_interf)
local_interf_chroms = queue_sim(queue, 'local interf', local_interf)

run_sims(queue)

chrom_sets = [
    no_interf_chroms,
    local_interf_chroms,
]

interevent_dist_data = collect_interevent_dist_data(
    transform_interevent_dist_data, chrom_sets
)
interevent_dist_plot = standard_plotter(interevent_dist_data)
_ = interevent_dist_plot.suptitle('Distribution of COs by interevent distance')
ax = interevent_dist_plot.get_axes()[0]
no_interf_line, local_interf_line = ax.patches
no_interf_line.set_color('black')
local_interf_line.set_color('red')
_ = ax.legend()
save_plot(interevent_dist_plot, fmt='svg')
