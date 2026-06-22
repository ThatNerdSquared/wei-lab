import random
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from croissant.chromosomes import ChromosomeSet, ChromsetMaker
from croissant.chromosomes.core import ChromosomeLike

from src.specific_interf_sims.interf_schemes import BinArray, InterferenceScheme
from src.utils.cli_utils import elapsed_time_printer


@dataclass(kw_only=True)
class InterferenceSim:
    interf_scheme: InterferenceScheme
    starting_bins: BinArray
    chrom_set: ChromosomeSet[ChromosomeLike]
    rand: np.random.Generator
    completed: bool = False


type SimModel = Callable[
    [
        ChromosomeSet[ChromosomeLike],
        BinArray,
        InterferenceScheme,
        np.random.Generator,
    ],
    None,
]


@dataclass
class SimulationQueue:
    model: SimModel
    default_starting_bins: BinArray | None = None
    sims: list[InterferenceSim] = field(default_factory=list)
    rand_seed: int | None = None
    chroms_per_sim: int | None = None
    default_chromset_maker: ChromsetMaker | None = None

    def __post_init__(self):
        random.seed(self.rand_seed)


def queue_sim(
    queue: SimulationQueue,
    label: str,
    interf_scheme: InterferenceScheme,
    chrom_count: int | None = None,
    starting_bins: BinArray | None = None,
    provided_chroms: ChromosomeSet[ChromosomeLike] | None = None,
) -> ChromosomeSet[ChromosomeLike]:
    if any(x.chrom_set.label == label for x in queue.sims):
        raise ValueError(
            f'Label {label} already in queue - please use a unique label name instead.'
        )
    bins = starting_bins if starting_bins is not None else queue.default_starting_bins
    num_chroms = chrom_count or queue.chroms_per_sim
    assert bins is not None
    assert num_chroms is not None

    assert provided_chroms or queue.default_chromset_maker
    chrom_set = (
        queue.default_chromset_maker(
            label=label, num_chroms=num_chroms, chrom_length=len(bins)
        )
        if queue.default_chromset_maker
        else provided_chroms
    )
    assert chrom_set is not None

    sim = InterferenceSim(
        interf_scheme=interf_scheme,
        starting_bins=bins,
        chrom_set=chrom_set,
        rand=np.random.default_rng(queue.rand_seed),
    )
    queue.sims.append(sim)
    return chrom_set


def run_sims(queue: SimulationQueue):
    total_sims = len(queue.sims)
    print('simulating crossovers...')
    for queue_num in range(total_sims):
        sim = queue.sims[queue_num]
        if sim.completed:
            print('op complete', f'[{queue_num + 1}/{total_sims}]:')
            print('restored from previous completion.')
            continue
        with elapsed_time_printer(
            op_name=sim.chrom_set.label,
            op_number=(queue_num + 1, total_sims),
            chrom_set=sim.chrom_set,
        ):
            queue.model(sim.chrom_set, sim.starting_bins, sim.interf_scheme, sim.rand)
            sim.completed = True
    print('all queued sims completed.')
