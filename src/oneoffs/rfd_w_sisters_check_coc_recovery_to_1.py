import argparse
from typing import cast

import numpy as np
from croissant.chromosomes import make_default_chromset

from src.specific_interf_sims.data_collectors import (
    collect_single_refregion_coc_curve_data,
    collect_sliding_compareregion_coc_curve_data,
)
from src.specific_interf_sims.data_transformers import (
    Region,
    transform_mean_cos_per_chrom_data,
    transform_single_refregion_coc_curve_data,
    transform_sliding_compare_region_coc_curve_data,
)
from src.specific_interf_sims.interf_schemes import (
    faster_rising_local_interf,
    local_interf,
    no_interf,
    very_fast_rising_local_interf,
)
from src.specific_interf_sims.models import induce_COs_via_random_forward_draw
from src.specific_interf_sims.plotter import (
    save_plot,
    standard_plotter,
)
from src.specific_interf_sims.simulator import SimulationQueue, queue_sim, run_sims
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

RAND_SEED = 54321 if not cast(bool, args.alt_seed_coc) else 93719
queue = SimulationQueue(
    induce_COs_via_random_forward_draw,
    np.zeros(CHROM_LEN) + STARTING_P_HIT,
    rand_seed=RAND_SEED,
    chroms_per_sim=NUM_CHROMS,
    default_chromset_maker=make_default_chromset,
)
no_interf_chroms = queue_sim(queue, 'no interf', no_interf)
local_chroms = queue_sim(queue, 'local', local_interf)
faster_local_chroms = queue_sim(
    queue, 'faster-rising local', faster_rising_local_interf
)
very_fast_local_chroms = queue_sim(
    queue, 'very fast-rising local', very_fast_rising_local_interf
)

run_sims(queue)

chrom_sets = [
    no_interf_chroms,
    local_chroms,
    faster_local_chroms,
    very_fast_local_chroms,
]

mapThenPrint(transform_mean_cos_per_chrom_data, chrom_sets)

expanding_coc_curve_data = collect_single_refregion_coc_curve_data(
    lambda x: transform_single_refregion_coc_curve_data(
        x,
        ref_region=Region(0, 50),
        min_testregion_size=32,
    ),
    chrom_sets,
    refregion_size=50,
)
expanding_coc_plot = standard_plotter(expanding_coc_curve_data)
save_plot(
    expanding_coc_plot,
    additional_annotation=f'seed{RAND_SEED}',
)

sliding_coc_curve_data = collect_sliding_compareregion_coc_curve_data(
    lambda x: transform_sliding_compare_region_coc_curve_data(
        x, ref_region=Region(0, 75), sliding_region_size=75
    ),
    chrom_sets,
    testregion_size=75,
)
sliding_coc_plot = standard_plotter(sliding_coc_curve_data)
save_plot(
    sliding_coc_plot,
    additional_annotation=f'seed{RAND_SEED}',
)
