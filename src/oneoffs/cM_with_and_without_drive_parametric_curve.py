import argparse
from typing import cast

import numpy as np
from croissant.chromosomes import (
    ChromosomeSet,
    ChromosomeTetrad,
    haplotypeSwitchingCOsOnlySelector,
    make_chromset_custom_sister_selectors,
    selectMostRecombinantForPool,
    selectOnlyRecombinantsForPool,
    selectSingleSisterForPool,
)
from croissant.outputs.cM_drive_diff_parametric_curve import cMDriveDiffParametricCurve
from croissant.recomb_sim import downsample

from src.specific_interf_sims.interf_schemes import (
    no_interf,
    very_fast_rising_local_interf,
)
from src.specific_interf_sims.models import induce_COs_via_random_forward_draw
from src.specific_interf_sims.simulator import SimulationQueue, queue_sim, run_sims

NUM_CHROMS = 100000
CHROM_LEN = 1000


def _make_chromset():
    return make_chromset_custom_sister_selectors(
        label='chromosomes to sim',
        num_chroms=NUM_CHROMS,
        chrom_length=CHROM_LEN,
        co_selector=haplotypeSwitchingCOsOnlySelector,
        pool_selector=selectSingleSisterForPool,
    )


parser = argparse.ArgumentParser()
_ = parser.add_argument(
    '--interf',
    help='simulate with interf',
    action='store_true',
)
args = parser.parse_args()
WITH_INTERF = cast(bool, args.interf)

chrom_sets: list[ChromosomeSet[ChromosomeTetrad]] = []
queue = SimulationQueue(
    induce_COs_via_random_forward_draw, rand_seed=54321, chroms_per_sim=NUM_CHROMS
)
for AVG_COS_PER_CHROM in np.linspace(1, 10, 100):
    chroms_to_sim = _make_chromset()
    _ = queue_sim(
        queue,
        f'chromosomes to sim (rate={AVG_COS_PER_CHROM})',
        no_interf if not WITH_INTERF else very_fast_rising_local_interf,
        starting_bins=np.zeros(CHROM_LEN) + (AVG_COS_PER_CHROM / CHROM_LEN),
        provided_chroms=chroms_to_sim,
    )
    chrom_sets.append(chroms_to_sim)
run_sims(queue)

no_drive_data = list(
    map(
        lambda x: downsample(x, selectSingleSisterForPool, new_label='no drive'),
        chrom_sets,
    )
)
partial_drive_data = list(
    map(
        lambda x: downsample(
            x, selectOnlyRecombinantsForPool, new_label='partial drive'
        ),
        chrom_sets,
    )
)
max_drive_data = list(
    map(
        lambda x: downsample(x, selectMostRecombinantForPool, new_label='max drive'),
        chrom_sets,
    )
)

partial_drive_diff_plot = cMDriveDiffParametricCurve(with_interf=WITH_INTERF)
partial_drive_diff_plot.build(
    {'reference': no_drive_data, 'experimental': partial_drive_data}
)
max_drive_diff_plot = cMDriveDiffParametricCurve(with_interf=WITH_INTERF)
max_drive_diff_plot.build({'reference': no_drive_data, 'experimental': max_drive_data})
