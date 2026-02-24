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


# REGION_SIZE = 75
# STEP_SIZE = 37
# REGION_SIZE = 75
# STEP_SIZE = 185

# REGION_SIZE = 120
# STEP_SIZE = 20

# STEP_SIZE = 110
REGION_SIZE = 150
STEP_SIZE = 50
plotter = SingleFigPlotter(
    PlotGridDimensions(
        1, int(((chrom_sets[0].chrom_length - REGION_SIZE) / STEP_SIZE))
    ),
    'sliding_expanding_testregion_coc -- '
    + f'{NUM_CHROMS / 1000}k chroms @ {CHROM_LEN / 1000}k bins / chrom',
    subplot_size=(5, 4),
)
for start_loc in range(0, chrom_sets[0].chrom_length - REGION_SIZE, STEP_SIZE):
    expanding_coc_curve_data = collect_single_refregion_coc_curve_data(
        lambda x: transform_single_refregion_coc_curve_data(
            x,
            ref_region=Region(start_loc, start_loc + REGION_SIZE),
            min_testregion_size=32,
            calculate_inside_refregion=True,
        ),
        chrom_sets,
    )
    plotter.add_plot(expanding_coc_curve_data, f'{start_loc},{start_loc + REGION_SIZE}')
    print(f'{start_loc},{start_loc + REGION_SIZE}')

save_plot(
    plotter.fig,
    additional_annotation=f'seed{RAND_SEED}',
)
