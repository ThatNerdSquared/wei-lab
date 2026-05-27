import argparse
import functools

import numpy as np
from croissant.chromosomes import (
    ChromosomeSet,
    haplotypeSwitchingCOsOnlySelector,
    make_chromset_custom_sister_selectors,
    probabilistic_selector,
    selectMostRecombinantForPool,
    selectOnlyRecombinantsForPool,
    selectSingleSisterForPool,
)
from croissant.chromosomes.core import ChromosomeLike
from matplotlib.axes import Axes

from src.specific_interf_sims.data_collectors import collect_cM_rf_curve_data
from src.specific_interf_sims.data_packet import DataPacket
from src.specific_interf_sims.data_transformers import transform_cM_rf_curve_data
from src.specific_interf_sims.interf_schemes import (
    no_interf,
    very_fast_rising_local_interf,
)
from src.specific_interf_sims.models import induce_COs_via_random_forward_draw
from src.specific_interf_sims.plotter import save_plot, standard_plotter
from src.specific_interf_sims.simulator import SimulationQueue, queue_sim, run_sims
from src.utils.standard_funcs import haldane_input_cM, kosambi

parser = argparse.ArgumentParser()
_ = parser.add_argument(
    '--interf',
    help='simulate with interf and post-interf avg of 2',
    action='store_true',
)
args = parser.parse_args()

NUM_CHROMS = 100000
CHROM_LEN = 1000
AVG_COS_PER_CHROMOSOME = 2 if not args.interf else 7
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
    _ = queue_sim(
        queue,
        chrom_set.label,
        no_interf if not args.interf else very_fast_rising_local_interf,
        provided_chroms=chrom_set,
    )
run_sims(queue)

CurrPacket = DataPacket[tuple[list[float], list[float]]]


def _subtract_nodrive_data_from_packet(packet: CurrPacket):
    _, nodrive_rf_data = packet.data['no drive']
    for label, (x_coords, rf_data) in packet.data.items():
        if label == 'no drive':
            continue
        delta_rf_data = list(map(float.__sub__, rf_data, nodrive_rf_data))
        packet.data[label] = (x_coords, delta_rf_data)
    packet.ylabel = '∆RF'


def _plot_canonical_map_func_delta_rf(packet: CurrPacket, ax: Axes):
    _, nodrive_rf_data = packet.data['no drive']
    x = np.linspace(0, 1, int(1000 * 1))
    _ = ax.plot(
        x,
        list(map(float.__sub__, haldane_input_cM(x), nodrive_rf_data)),
        label='haldane',
    )
    _ = ax.plot(x, list(map(float.__sub__, x, nodrive_rf_data)), label='morgan')
    _ = ax.plot(
        x, list(map(float.__sub__, kosambi(x), nodrive_rf_data)), label='kosambi'
    )
    _ = ax.legend()


def _plot_delta_rf_for_chromsets[C: ChromosomeLike](chrom_sets: list[ChromosomeSet[C]]):
    cM_rf_curve_data = collect_cM_rf_curve_data(transform_cM_rf_curve_data, chrom_sets)
    _subtract_nodrive_data_from_packet(cM_rf_curve_data)
    cM_rf_curve_plot = standard_plotter(cM_rf_curve_data)
    cM_rf_ax = cM_rf_curve_plot.get_axes()[0]
    _plot_canonical_map_func_delta_rf(cM_rf_curve_data, cM_rf_ax)
    _ = cM_rf_ax.set_xlim(right=2.0)
    _ = cM_rf_ax.set_ylim(auto=True)
    save_plot(cM_rf_curve_plot)


_plot_delta_rf_for_chromsets([chrom_sets[0]] + partial_drive_chromsets)
_plot_delta_rf_for_chromsets([chrom_sets[0]] + max_drive_chromsets)
