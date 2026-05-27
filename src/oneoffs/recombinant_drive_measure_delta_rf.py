import argparse
import functools
import itertools
from typing import cast

import numpy as np
from croissant.chromosomes import (
    ChromosomeSet,
    ChromosomeTetrad,
    haplotypeSwitchingCOsOnlySelector,
    make_chromset_custom_sister_selectors,
    probabilistic_selector,
    selectMostRecombinantForPool,
    selectOnlyRecombinantsForPool,
    selectSingleSisterForPool,
)
from croissant.chromosomes.chromosome_tetrad import PoolSisterSelector
from croissant.outputs.delta_rf_curve import DeltaRFCurve

from src.specific_interf_sims.interf_schemes import (
    no_interf,
    very_fast_rising_local_interf,
)
from src.specific_interf_sims.models import induce_COs_via_random_forward_draw
from src.specific_interf_sims.simulator import SimulationQueue, queue_sim, run_sims

parser = argparse.ArgumentParser()
_ = parser.add_argument(
    '--interf',
    help='simulate with interf and post-interf avg of 2',
    action='store_true',
)
args = parser.parse_args()
WITH_INTERF = cast(bool, args.interf)

NUM_CHROMS = 100000
CHROM_LEN = 1000
AVG_COS_PER_CHROMOSOME = 2 if not WITH_INTERF else 7
STARTING_P_HIT = AVG_COS_PER_CHROMOSOME / CHROM_LEN
RAND_SEED = 54321
RAND_FOR_SELECTORS = np.random.default_rng(RAND_SEED)


_make_chromset = functools.partial(
    make_chromset_custom_sister_selectors,
    num_chroms=NUM_CHROMS,
    chrom_length=CHROM_LEN,
    co_selector=haplotypeSwitchingCOsOnlySelector,
)


def _make_probabilistic_drive_sims(
    drive_type: str, drive_selector: PoolSisterSelector
) -> list[ChromosomeSet[ChromosomeTetrad]]:
    res: list[ChromosomeSet[ChromosomeTetrad]] = []
    for level in [0.25, 0.5, 0.75, 1.0]:
        res.append(
            _make_chromset(
                label=f'{drive_type} ({int(level * 100)}%)',
                pool_selector=lambda x: probabilistic_selector(
                    x,
                    RAND_FOR_SELECTORS,
                    drive_selector,
                    selectSingleSisterForPool,
                    level,
                ),
            )
        )
    return res


no_drive_chroms = _make_chromset(
    label='no drive', pool_selector=selectSingleSisterForPool
)
partial_drive_chromsets = _make_probabilistic_drive_sims(
    'partial drive', selectOnlyRecombinantsForPool
)
max_drive_chromsets = _make_probabilistic_drive_sims(
    'max drive', selectMostRecombinantForPool
)

queue = SimulationQueue(
    induce_COs_via_random_forward_draw,
    np.zeros(CHROM_LEN) + STARTING_P_HIT,
    rand_seed=RAND_SEED,
    chroms_per_sim=NUM_CHROMS,
)
for chrom_set in itertools.chain(
    [no_drive_chroms], partial_drive_chromsets, max_drive_chromsets
):
    _ = queue_sim(
        queue,
        chrom_set.label,
        no_interf if not WITH_INTERF else very_fast_rising_local_interf,
        provided_chroms=chrom_set,
    )
run_sims(queue)

partialdrive_deltarf_plot = DeltaRFCurve(with_interf=WITH_INTERF)
partialdrive_deltarf_plot.build(
    {'reference': [no_drive_chroms], 'experimental': partial_drive_chromsets}
)
maxdrive_deltarf_plot = DeltaRFCurve(with_interf=WITH_INTERF)
maxdrive_deltarf_plot.build(
    {'reference': [no_drive_chroms], 'experimental': max_drive_chromsets}
)
