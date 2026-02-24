import functools

import numpy as np
from croissant.chromosomes import (
    haplotypeSwitchingCOsOnlySelector,
    make_chromset_custom_sister_selectors,
    selectMostRecombinantForPool,
    selectOnlyRecombinantsForPool,
    selectSingleSisterForPool,
)

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
from src.specific_interf_sims.interf_schemes import no_interf
from src.specific_interf_sims.models import induce_COs_via_random_forward_draw
from src.specific_interf_sims.plotter import (
    plot_canonical_map_funcs,
    save_plot,
    standard_plotter,
)
from src.specific_interf_sims.simulator import SimulationQueue, queue_sim, run_sims
from src.utils.cli_utils import mapThenPrint

NUM_CHROMS = 100000
CHROM_LEN = 1000
AVG_COS_PER_CHROMOSOME = 2
STARTING_P_HIT = AVG_COS_PER_CHROMOSOME / CHROM_LEN

_make_chromset = functools.partial(
    make_chromset_custom_sister_selectors,
    num_chroms=NUM_CHROMS,
    chrom_length=CHROM_LEN,
    co_selector=haplotypeSwitchingCOsOnlySelector,
)

queue = SimulationQueue(
    induce_COs_via_random_forward_draw,
    np.zeros(CHROM_LEN) + STARTING_P_HIT,
    rand_seed=54321,
    chroms_per_sim=NUM_CHROMS,
)

no_drive_chroms = _make_chromset(
    label='no drive', pool_selector=selectSingleSisterForPool
)
partial_drive_chroms = _make_chromset(
    label='partial drive', pool_selector=selectOnlyRecombinantsForPool
)
max_drive_chroms = _make_chromset(
    label='max drive', pool_selector=selectMostRecombinantForPool
)

chrom_sets = [no_drive_chroms, partial_drive_chroms, max_drive_chroms]
for chrom_set in chrom_sets:
    _ = queue_sim(queue, chrom_set.label, no_interf, provided_chroms=chrom_set)
run_sims(queue)


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
