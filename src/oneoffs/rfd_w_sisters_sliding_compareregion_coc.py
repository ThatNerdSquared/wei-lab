import argparse

import numpy as np
from croissant.chromosomes import make_default_chromset

from src.specific_interf_sims.data_collectors import (
    collect_sliding_compareregion_coc_curve_data,
)
from src.specific_interf_sims.data_transformers import (
    Region,
    transform_mean_cos_per_chrom_data,
    transform_sliding_compare_region_coc_curve_data,
)
from src.specific_interf_sims.interf_schemes import (
    equalised_fixed_regional_interf,
    equalised_global_interf,
    local_interf,
    no_interf,
)
from src.specific_interf_sims.models import induce_COs_via_random_forward_draw
from src.specific_interf_sims.plotter import save_plot, standard_plotter
from src.specific_interf_sims.simulator import SimulationQueue, queue_sim, run_sims
from src.utils.cli_utils import mapThenPrint


def sim(large: bool, alt_seed: bool, best_bin_size_only: bool):
    NUM_CHROMS = 100000 if not large else 1000000
    CHROM_LEN = 1000
    AVG_COS_PER_CHROMOSOME = 2
    STARTING_P_HIT = AVG_COS_PER_CHROMOSOME / CHROM_LEN

    queue = SimulationQueue(
        induce_COs_via_random_forward_draw,
        np.zeros(CHROM_LEN) + STARTING_P_HIT,
        rand_seed=54321 if not alt_seed else 93719,
        chroms_per_sim=NUM_CHROMS,
        default_chromset_maker=make_default_chromset,
    )
    no_interf_chroms = queue_sim(queue, 'no interf', no_interf)
    local_chroms = queue_sim(queue, 'local', local_interf)
    equalised_global_chroms = queue_sim(
        queue, 'equalised global', equalised_global_interf
    )
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

    for region_size in [10, 25, 32, 40, 50, 64, 75] if not best_bin_size_only else [75]:
        coc_curve_data = collect_sliding_compareregion_coc_curve_data(
            lambda x: transform_sliding_compare_region_coc_curve_data(
                x, ref_region=Region(0, region_size), sliding_region_size=region_size
            ),
            chrom_sets,
            testregion_size=region_size,
        )
        coc_plot = standard_plotter(coc_curve_data)
        save_plot(coc_plot)


parser = argparse.ArgumentParser()
_ = parser.add_argument(
    '--large', help='simulate on 1mil chromosomes instead of 100k', action='store_true'
)
_ = parser.add_argument(
    '--alt-seed',
    help='use 93719 as the random seed instead of 54321',
    action='store_true',
)
_ = parser.add_argument(
    '--best-bin-size-only',
    help='only generate plots for testregion_size=75',
    action='store_true',
)
args = parser.parse_args()
sim(args.large, args.alt_seed, args.best_bin_size_only)  # pyright: ignore[reportAny]
# plot generation: no flags, --alt-seed --best-bin-size-only,
# --large --best-bin-size-only, --large --alt-seed --best-bin-size-only
