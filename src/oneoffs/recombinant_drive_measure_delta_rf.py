import argparse
import functools
from typing import cast

import numpy as np
from croissant.chromosomes import (
    haplotypeSwitchingCOsOnlySelector,
    make_chromset_custom_sister_selectors,
    selectMostRecombinantForPool,
    selectOnlyRecombinantsForPool,
    selectSingleSisterForPool,
)
from croissant.outputs.delta_rf_curve import DeltaRFCurve
from croissant.recomb_drive import make_default_drive_downsamples
from croissant.recomb_sim import downsample

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


no_drive_chroms = _make_chromset(
    label='no drive', pool_selector=selectSingleSisterForPool
)
queue = SimulationQueue(
    induce_COs_via_random_forward_draw,
    np.zeros(CHROM_LEN) + STARTING_P_HIT,
    rand_seed=RAND_SEED,
    chroms_per_sim=NUM_CHROMS,
)
_ = queue_sim(
    queue,
    no_drive_chroms.label,
    no_interf if not WITH_INTERF else very_fast_rising_local_interf,
    provided_chroms=no_drive_chroms,
)
run_sims(queue)

ref_chromatids = downsample(
    no_drive_chroms, selectSingleSisterForPool, new_label='no drive'
)
partial_drive_chromatids = make_default_drive_downsamples(
    no_drive_chroms, 'partial drive', selectOnlyRecombinantsForPool
)
max_drive_chromatids = make_default_drive_downsamples(
    no_drive_chroms, 'max drive', selectMostRecombinantForPool
)


partialdrive_deltarf_plot = DeltaRFCurve(with_interf=WITH_INTERF)
partialdrive_deltarf_plot.build(
    {'reference': [ref_chromatids], 'experimental': partial_drive_chromatids}
)
maxdrive_deltarf_plot = DeltaRFCurve(with_interf=WITH_INTERF)
maxdrive_deltarf_plot.build(
    {'reference': [ref_chromatids], 'experimental': max_drive_chromatids}
)
