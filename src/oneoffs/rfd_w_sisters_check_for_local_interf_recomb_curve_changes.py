import numpy as np

from src.specific_interf_sims.data_collectors import (
    collect_cM_rf_curve_data,
    collect_co_per_chrom_dist_data,
    collect_interevent_dist_data,
    collect_recomb_curve_data,
)
from src.specific_interf_sims.data_transformers import (
    transform_cM_rf_curve_data,
    transform_co_per_chrom_dist_data,
    transform_interevent_dist_data,
    transform_mean_cos_per_chrom_data,
    transform_mean_cos_per_chrom_in_pool_data,
    transform_recomb_curve_data,
)
from src.specific_interf_sims.interf_schemes import (
    faster_rising_local_interf,
    local_interf,
    no_interf,
    slower_rising_local_interf,
)
from src.specific_interf_sims.models import induce_COs_via_random_forward_draw
from src.specific_interf_sims.plotter import (
    plot_canonical_map_funcs,
    save_plot,
    standard_plotter,
)
from src.specific_interf_sims.simulator import SimulationQueue, queue_sim, run_sims
from src.utils.chromosome_set import make_default_chromset
from src.utils.cli_utils import mapThenPrint

NUM_CHROMS = 200000
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
local_interf_chroms = queue_sim(queue, 'local', local_interf)
slower_rising_local_interf_chroms = queue_sim(
    queue, 'slower rising local', slower_rising_local_interf
)
faster_rising_local_interf_chroms = queue_sim(
    queue, 'faster rising local', faster_rising_local_interf
)

run_sims(queue)

chrom_sets = [
    no_interf_chroms,
    local_interf_chroms,
    slower_rising_local_interf_chroms,
    faster_rising_local_interf_chroms,
]

mapThenPrint(transform_mean_cos_per_chrom_data, chrom_sets)
mapThenPrint(transform_mean_cos_per_chrom_in_pool_data, chrom_sets)

recomb_curve_data = collect_recomb_curve_data(transform_recomb_curve_data, chrom_sets)
recomb_curve_plot = standard_plotter(recomb_curve_data)
save_plot(recomb_curve_plot)

co_per_chrom_dist_data = collect_co_per_chrom_dist_data(
    transform_co_per_chrom_dist_data, chrom_sets
)
co_per_chrom_dist_plot = standard_plotter(co_per_chrom_dist_data)
save_plot(co_per_chrom_dist_plot)

interevent_dist_data = collect_interevent_dist_data(
    transform_interevent_dist_data, chrom_sets
)
interevent_dist_plot = standard_plotter(interevent_dist_data)
save_plot(interevent_dist_plot)

cM_rf_curve_data = collect_cM_rf_curve_data(transform_cM_rf_curve_data, chrom_sets)
cM_rf_curve_plot = standard_plotter(cM_rf_curve_data)
cM_rf_ax = cM_rf_curve_plot.get_axes()[0]
plot_canonical_map_funcs(cM_rf_ax)
save_plot(cM_rf_curve_plot)
