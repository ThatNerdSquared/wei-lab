import argparse
from typing import cast

import numpy as np
from croissant.chromosomes import make_default_chromset

from src.specific_interf_sims.data_collectors import (
    collect_3d_expanding_testregion_coc_curve_data,
)
from src.specific_interf_sims.data_transformers import (
    transform_mean_cos_per_chrom_data,
    transform_sliding_refregion_coc_curve_data,
)
from src.specific_interf_sims.interf_schemes import (
    equalised_fixed_regional_interf,
    equalised_global_interf,
    local_interf,
    no_interf,
)
from src.specific_interf_sims.models import induce_COs_via_random_forward_draw
from src.specific_interf_sims.plotter import save_plot, standard_plotter_threedim
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


for chrom_set in chrom_sets:
    coc_curve_data = collect_3d_expanding_testregion_coc_curve_data(
        lambda x: transform_sliding_refregion_coc_curve_data(x, region_size=75),
        chrom_set,
        refregion_size=75,
    )
    coc_curve_threedim_plot = standard_plotter_threedim(coc_curve_data)
    save_plot(
        coc_curve_threedim_plot,
        additional_annotation=f'seed{RAND_SEED}',
    )
