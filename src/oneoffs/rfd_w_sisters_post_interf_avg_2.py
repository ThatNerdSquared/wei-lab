import argparse
from typing import cast

import numpy as np

from src.specific_interf_sims.data_collectors import (
    collect_cM_rf_curve_data,
    collect_co_per_chrom_dist_data,
    collect_interevent_dist_data,
    collect_recomb_curve_data,
    collect_single_refregion_coc_curve_data,
    collect_sliding_compareregion_coc_curve_data,
)
from src.specific_interf_sims.data_transformers import (
    Region,
    transform_cM_rf_curve_data_haplotype_switching_only,
    transform_co_per_chrom_dist_data,
    transform_interevent_dist_data,
    transform_mean_cos_per_chrom_data,
    transform_recomb_curve_data,
    transform_single_refregion_coc_curve_data,
    transform_sliding_compare_region_coc_curve_data,
)
from src.specific_interf_sims.interf_schemes import (
    equalised_fixed_regional_interf,
    equalised_global_interf,
    local_interf,
    no_interf,
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

parser = argparse.ArgumentParser()
_ = parser.add_argument(
    '--alt-seed-coc',
    help='use 93719 instead of 54321 as random seed, and only run CoC sims',
    action='store_true',
)
args = parser.parse_args()

NUM_CHROMS = 100000
CHROM_LEN = 1000
AVG_COS_PER_CHROMOSOME = 7.5
STARTING_P_HIT = AVG_COS_PER_CHROMOSOME / CHROM_LEN

queue = SimulationQueue(
    induce_COs_via_random_forward_draw,
    np.zeros(CHROM_LEN) + STARTING_P_HIT,
    rand_seed=54321 if not cast(bool, args.alt_seed_coc) else 93719,
    chroms_per_sim=NUM_CHROMS,
    default_chromset_maker=make_default_chromset,
)
no_interf_chroms_avg_2 = queue_sim(
    queue,
    'no interf (avg=2)',
    no_interf,
    starting_bins=np.zeros(CHROM_LEN) + (2 / CHROM_LEN),
)
no_interf_chroms = queue_sim(queue, 'no interf', no_interf)
local_chroms = queue_sim(queue, 'local', local_interf)
equalised_global_chroms = queue_sim(queue, 'equalised global', equalised_global_interf)
equalised_fixed_regional_chroms = queue_sim(
    queue, 'equalised fixed regional', equalised_fixed_regional_interf
)

run_sims(queue)

chrom_sets = [
    no_interf_chroms,
    local_chroms,
    equalised_global_chroms,
    equalised_fixed_regional_chroms,
    no_interf_chroms_avg_2,
]

mapThenPrint(transform_mean_cos_per_chrom_data, chrom_sets)

if not cast(bool, args.alt_seed_coc):
    recomb_curve_data = collect_recomb_curve_data(
        transform_recomb_curve_data, chrom_sets
    )
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

    cM_rf_curve_data = collect_cM_rf_curve_data(
        transform_cM_rf_curve_data_haplotype_switching_only, chrom_sets
    )
    cM_rf_curve_plot = standard_plotter(cM_rf_curve_data)
    cM_rf_ax = cM_rf_curve_plot.get_axes()[0]
    plot_canonical_map_funcs(cM_rf_ax)
    _ = cM_rf_ax.set_xlim(right=1.5)
    save_plot(cM_rf_curve_plot)

expanding_coc_curve_data = collect_single_refregion_coc_curve_data(
    transform_single_refregion_coc_curve_data, chrom_sets
)
expanding_coc_plot = standard_plotter(expanding_coc_curve_data)
save_plot(expanding_coc_plot)

sliding_coc_curve_data = collect_sliding_compareregion_coc_curve_data(
    lambda x: transform_sliding_compare_region_coc_curve_data(
        x, ref_region=Region(0, 75), sliding_region_size=75
    ),
    chrom_sets,
    testregion_size=75,
)
sliding_coc_plot = standard_plotter(sliding_coc_curve_data)
save_plot(sliding_coc_plot)
