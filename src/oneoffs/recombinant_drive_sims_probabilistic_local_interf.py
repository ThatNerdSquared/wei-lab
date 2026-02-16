import functools

import numpy as np

from src.specific_interf_sims.data_collectors import (
    collect_cM_rf_curve_data,
    collect_co_per_chrom_dist_data,
)
from src.specific_interf_sims.data_transformers import (
    transform_cM_rf_curve_data,
    transform_co_per_chrom_dist_data,
    transform_mean_cos_per_chrom_data,
    transform_mean_cos_per_chrom_in_pool_data,
)
from src.specific_interf_sims.interf_schemes import local_interf
from src.specific_interf_sims.models import induce_COs_via_random_forward_draw
from src.specific_interf_sims.plotter import (
    plot_canonical_map_funcs,
    save_plot,
    standard_plotter,
)
from src.specific_interf_sims.simulator import SimulationQueue, queue_sim, run_sims
from src.utils.chromosome_set import make_chromset_custom_sister_selectors
from src.utils.chromosome_with_sister_chromatids import (
    haplotypeSwitchingCOsOnlySelector,
    probabilistic_selector,
    selectMostRecombinantForPool,
    selectOnlyRecombinantsForPool,
    selectSingleSisterForPool,
)
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

RAND_SEED = 54321
queue = SimulationQueue(
    induce_COs_via_random_forward_draw,
    np.zeros(CHROM_LEN) + STARTING_P_HIT,
    rand_seed=RAND_SEED,
    chroms_per_sim=NUM_CHROMS,
)

RAND_FOR_SELECTORS = np.random.default_rng(RAND_SEED)
no_drive_chroms = _make_chromset(
    label='no drive', pool_selector=selectSingleSisterForPool
)

partial_drive_100_pct_chroms = _make_chromset(
    label='partial drive (100%)',
    pool_selector=lambda x: probabilistic_selector(
        x,
        RAND_FOR_SELECTORS,
        selectOnlyRecombinantsForPool,
        selectSingleSisterForPool,
        1.0,
    ),
)
partial_drive_75_pct_chroms = _make_chromset(
    label='partial drive (75%)',
    pool_selector=lambda x: probabilistic_selector(
        x,
        RAND_FOR_SELECTORS,
        selectOnlyRecombinantsForPool,
        selectSingleSisterForPool,
        0.75,
    ),
)
partial_drive_50_pct_chroms = _make_chromset(
    label='partial drive (50%)',
    pool_selector=lambda x: probabilistic_selector(
        x,
        RAND_FOR_SELECTORS,
        selectOnlyRecombinantsForPool,
        selectSingleSisterForPool,
        0.5,
    ),
)
partial_drive_25_pct_chroms = _make_chromset(
    label='partial drive (25%)',
    pool_selector=lambda x: probabilistic_selector(
        x,
        RAND_FOR_SELECTORS,
        selectOnlyRecombinantsForPool,
        selectSingleSisterForPool,
        0.25,
    ),
)

max_drive_100_pct_chroms = _make_chromset(
    label='max drive (100%)',
    pool_selector=lambda x: probabilistic_selector(
        x,
        RAND_FOR_SELECTORS,
        selectMostRecombinantForPool,
        selectSingleSisterForPool,
        1.0,
    ),
)
max_drive_75_pct_chroms = _make_chromset(
    label='max drive (75%)',
    pool_selector=lambda x: probabilistic_selector(
        x,
        RAND_FOR_SELECTORS,
        selectMostRecombinantForPool,
        selectSingleSisterForPool,
        0.75,
    ),
)
max_drive_50_pct_chroms = _make_chromset(
    label='max drive (50%)',
    pool_selector=lambda x: probabilistic_selector(
        x,
        RAND_FOR_SELECTORS,
        selectMostRecombinantForPool,
        selectSingleSisterForPool,
        0.5,
    ),
)
max_drive_25_pct_chroms = _make_chromset(
    label='max drive (25%)',
    pool_selector=lambda x: probabilistic_selector(
        x,
        RAND_FOR_SELECTORS,
        selectMostRecombinantForPool,
        selectSingleSisterForPool,
        0.25,
    ),
)

partial_drive_chromsets = [
    partial_drive_25_pct_chroms,
    partial_drive_50_pct_chroms,
    partial_drive_75_pct_chroms,
    partial_drive_100_pct_chroms,
]

max_drive_chromsets = [
    max_drive_25_pct_chroms,
    max_drive_50_pct_chroms,
    max_drive_75_pct_chroms,
    max_drive_100_pct_chroms,
]

chrom_sets = (
    [
        no_drive_chroms,
    ]
    + partial_drive_chromsets
    + max_drive_chromsets
)

for chrom_set in chrom_sets:
    _ = queue_sim(queue, chrom_set.label, local_interf, provided_chroms=chrom_set)
run_sims(queue)


mapThenPrint(transform_mean_cos_per_chrom_data, chrom_sets)
mapThenPrint(transform_mean_cos_per_chrom_in_pool_data, chrom_sets)

partialdrive_co_per_chrom_dist_data = collect_co_per_chrom_dist_data(
    transform_co_per_chrom_dist_data, [chrom_sets[0]] + partial_drive_chromsets
)
partialdrive_co_per_chrom_dist_plot = standard_plotter(
    partialdrive_co_per_chrom_dist_data
)
save_plot(partialdrive_co_per_chrom_dist_plot)

maxdrive_co_per_chrom_dist_data = collect_co_per_chrom_dist_data(
    transform_co_per_chrom_dist_data, [chrom_sets[0]] + max_drive_chromsets
)
maxdrive_co_per_chrom_dist_plot = standard_plotter(maxdrive_co_per_chrom_dist_data)
save_plot(maxdrive_co_per_chrom_dist_plot)

partialdrive_cM_rf_curve_data = collect_cM_rf_curve_data(
    transform_cM_rf_curve_data, [chrom_sets[0]] + partial_drive_chromsets
)
partialdrive_cM_rf_curve_plot = standard_plotter(partialdrive_cM_rf_curve_data)
partialdrive_cM_rf_ax = partialdrive_cM_rf_curve_plot.get_axes()[0]
plot_canonical_map_funcs(partialdrive_cM_rf_ax)
save_plot(partialdrive_cM_rf_curve_plot)

maxdrive_cM_rf_curve_data = collect_cM_rf_curve_data(
    transform_cM_rf_curve_data, [chrom_sets[0]] + max_drive_chromsets
)
maxdrive_cM_rf_curve_plot = standard_plotter(maxdrive_cM_rf_curve_data)
maxdrive_cM_rf_ax = maxdrive_cM_rf_curve_plot.get_axes()[0]
plot_canonical_map_funcs(maxdrive_cM_rf_ax)
save_plot(maxdrive_cM_rf_curve_plot)
