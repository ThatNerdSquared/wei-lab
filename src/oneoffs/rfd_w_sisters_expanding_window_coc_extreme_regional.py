import argparse
from typing import cast

import numpy as np

from src.specific_interf_sims.data_collectors import (
    collect_single_refregion_coc_curve_data,
)
from src.specific_interf_sims.data_transformers import (
    transform_mean_cos_per_chrom_data,
    transform_single_refregion_coc_curve_data,
)
from src.specific_interf_sims.interf_schemes import (
    extreme_fixed_regional_interf,
    no_interf,
)
from src.specific_interf_sims.models import induce_COs_via_random_forward_draw
from src.specific_interf_sims.plotter import save_plot, standard_plotter
from src.specific_interf_sims.simulator import SimulationQueue, queue_sim, run_sims
from src.utils.chromosome_set import make_default_chromset
from src.utils.cli_utils import mapThenPrint

parser = argparse.ArgumentParser()
_ = parser.add_argument(
    '--large', help='simulate on 1mil chromosomes instead of 300k', action='store_true'
)
args = parser.parse_args()

NUM_CHROMS = 300000 if not cast(bool, args.large) else 1000000
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
extreme_regional_interf_chroms = queue_sim(
    queue, 'extreme regional', extreme_fixed_regional_interf
)

run_sims(queue)

chrom_sets = [
    no_interf_chroms,
    extreme_regional_interf_chroms,
]

mapThenPrint(transform_mean_cos_per_chrom_data, chrom_sets)

coc_curve_data = collect_single_refregion_coc_curve_data(
    transform_single_refregion_coc_curve_data, chrom_sets
)
coc_plot = standard_plotter(coc_curve_data)
save_plot(coc_plot)
