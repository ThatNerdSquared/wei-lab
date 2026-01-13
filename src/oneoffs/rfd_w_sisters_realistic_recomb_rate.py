import argparse
from typing import cast

import numpy as np
from scipy.integrate import quad

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
    '--alt-seed',
    help='use 93719 instead of 54321 as random seed, and only run CoC sims',
    action='store_true',
)
args = parser.parse_args()

NUM_CHROMS = 100000
CHROM_LEN = 1000
RAND_SEED = 54321 if not cast(bool, args.alt_seed) else 93719

# starting with a rough guesstimate: chromosome has 20/50 left arm, 10/50
# centromere region, 20/50 right arm.
SIZE_OF_BIN_IN_MB = 50 / CHROM_LEN
bin_locs = np.arange(CHROM_LEN)
scaled_bin_locs = bin_locs * SIZE_OF_BIN_IN_MB
left_arm = np.copy(scaled_bin_locs)
right_arm = np.copy(scaled_bin_locs)
for i in bin_locs:
    # quad returns 2-tuple w 2nd element being abserr
    left_arm[i] = quad(
        lambda x: 2.4313 + 0.4474 * x - 0.0312 * (x**2),
        left_arm[i],
        left_arm[i] + SIZE_OF_BIN_IN_MB,
    )[0]
    right_arm[i] = quad(
        lambda x: -1.3784 + 0.7318 * x - 0.0258 * (x**2),
        right_arm[i],
        right_arm[i] + SIZE_OF_BIN_IN_MB,
    )[0]
combined_arms = np.concatenate(
    [(left_arm[0:400] / 100), np.zeros(200), (right_arm[0:400]) / 100]
)
STARTING_BINS = np.maximum.reduce([combined_arms, np.zeros(CHROM_LEN)])

queue = SimulationQueue(
    induce_COs_via_random_forward_draw,
    STARTING_BINS,
    rand_seed=RAND_SEED,
    chroms_per_sim=NUM_CHROMS,
    default_chromset_maker=make_default_chromset,
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
]

mapThenPrint(transform_mean_cos_per_chrom_data, chrom_sets)

recomb_curve_data = collect_recomb_curve_data(transform_recomb_curve_data, chrom_sets)
recomb_curve_plot = standard_plotter(recomb_curve_data)
save_plot(recomb_curve_plot, additional_annotation=f'seed{RAND_SEED}')

co_per_chrom_dist_data = collect_co_per_chrom_dist_data(
    transform_co_per_chrom_dist_data, chrom_sets
)
co_per_chrom_dist_plot = standard_plotter(co_per_chrom_dist_data)
save_plot(co_per_chrom_dist_plot, additional_annotation=f'seed{RAND_SEED}')

interevent_dist_data = collect_interevent_dist_data(
    transform_interevent_dist_data, chrom_sets
)
interevent_dist_plot = standard_plotter(interevent_dist_data)
save_plot(interevent_dist_plot, additional_annotation=f'seed{RAND_SEED}')

cM_rf_curve_data = collect_cM_rf_curve_data(
    transform_cM_rf_curve_data_haplotype_switching_only, chrom_sets
)
cM_rf_curve_plot = standard_plotter(cM_rf_curve_data)
cM_rf_ax = cM_rf_curve_plot.get_axes()[0]
plot_canonical_map_funcs(cM_rf_ax)
_ = cM_rf_ax.set_xlim(right=1.5)
save_plot(cM_rf_curve_plot, additional_annotation=f'seed{RAND_SEED}')

expanding_coc_curve_data = collect_single_refregion_coc_curve_data(
    lambda x: transform_single_refregion_coc_curve_data(
        x, ref_region=Region(0, 50), min_testregion_size=50
    ),
    chrom_sets,
)
expanding_coc_plot = standard_plotter(expanding_coc_curve_data)
save_plot(expanding_coc_plot, additional_annotation=f'seed{RAND_SEED}')

sliding_coc_curve_data = collect_sliding_compareregion_coc_curve_data(
    lambda x: transform_sliding_compare_region_coc_curve_data(
        x, ref_region=Region(0, 75), sliding_region_size=75
    ),
    chrom_sets,
    testregion_size=75,
)
sliding_coc_plot = standard_plotter(sliding_coc_curve_data)
save_plot(sliding_coc_plot, additional_annotation=f'seed{RAND_SEED}')
