import argparse
from typing import cast

import numpy as np
from croissant.chromosomes import make_default_chromset

from src.specific_interf_sims.data_collectors import (
    collect_single_refregion_coc_curve_data,
)
from src.specific_interf_sims.data_transformers import (
    Region,
    transform_mean_cos_per_chrom_data,
    transform_single_refregion_coc_curve_data,
)
from src.specific_interf_sims.interf_schemes import (
    equalised_fixed_regional_interf,
    equalised_global_interf,
    local_interf,
    no_interf,
)
from src.specific_interf_sims.models import induce_COs_via_random_forward_draw
from src.specific_interf_sims.plotter import (
    PlotGridDimensions,
    SingleFigPlotter,
    save_plot,
)
from src.specific_interf_sims.simulator import SimulationQueue, queue_sim, run_sims
from src.utils.cli_utils import mapThenPrint

parser = argparse.ArgumentParser()
_ = parser.add_argument(
    '--alt-seed', help='use 93719 as seed instead of 54321', action='store_true'
)
_ = parser.add_argument(
    '--post-interf-2',
    help='use a post-interf avg COs/chrom = 2, instead of pre-interf',
    action='store_true',
)
args = parser.parse_args()

NUM_CHROMS = 100000
CHROM_LEN = 1000
AVG_COS_PER_CHROMOSOME = 2 if not cast(bool, args.post_interf_2) else 7.5
STARTING_P_HIT = AVG_COS_PER_CHROMOSOME / CHROM_LEN
RAND_SEED = 54321 if not cast(bool, args.alt_seed) else 93719

queue = SimulationQueue(
    induce_COs_via_random_forward_draw,
    np.zeros(CHROM_LEN) + STARTING_P_HIT,
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

# for refregion_size in [10, 25, 32, 40, 50, 64, 75]:
#     for min_testregion_size in [10, 25, 32, 40, 50, 64, 75]:
plotter = SingleFigPlotter(
    PlotGridDimensions(4, 4),
    'expanding_testregions_diff_sizes -- '
    + f'{NUM_CHROMS / 1000}k chroms @ {CHROM_LEN / 1000}k bins / chrom',
    subplot_size=(5, 4),
)
for refregion_size in [10, 32, 50, 75]:
    for min_testregion_size in [10, 32, 50, 75]:
        coc_curve_data = collect_single_refregion_coc_curve_data(
            lambda x: transform_single_refregion_coc_curve_data(
                x,
                ref_region=Region(0, refregion_size),
                min_testregion_size=min_testregion_size,
            ),
            chrom_sets,
            refregion_size=refregion_size,
        )
        plotter.add_plot(
            coc_curve_data, f'{refregion_size}ref, {min_testregion_size}min_test'
        )

save_plot(
    plotter.fig,
    additional_annotation=f'seed{RAND_SEED}',
)
