import random
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from src.specific_interf_sims.interf_schemes import BinArray, InterferenceScheme
from src.utils.chromosome_set import ChromosomeLike, ChromosomeSet, ChromsetMaker
from src.utils.cli_utils import elapsed_time_printer


@dataclass(kw_only=True)
class InterferenceSim[C: ChromosomeLike]:
    interf_scheme: InterferenceScheme
    starting_bins: BinArray
    chrom_set: ChromosomeSet[C]
    rand: np.random.Generator


@dataclass()
class SimulationQueue[C: ChromosomeLike]:
    model: Callable[
        [ChromosomeSet[C], BinArray, InterferenceScheme, np.random.Generator], None
    ]
    default_starting_bins: BinArray | None = None
    sims: list[InterferenceSim[C]] = field(default_factory=list)
    rand_seed: int | None = None
    chroms_per_sim: int | None = None
    default_chromset_maker: ChromsetMaker[C] | None = None

    def __post_init__(self):
        random.seed(self.rand_seed)


def queue_sim[C: ChromosomeLike](
    queue: SimulationQueue[C],
    label: str,
    interf_scheme: InterferenceScheme,
    chrom_count: int | None = None,
    starting_bins: BinArray | None = None,
    provided_chroms: ChromosomeSet[C] | None = None,
) -> ChromosomeSet[C]:
    bins = starting_bins or queue.default_starting_bins
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


def run_sims[C: ChromosomeLike](queue: SimulationQueue[C]):
    total_sims = len(queue.sims)
    print('simulating crossovers...')
    for queue_num in range(total_sims):
        sim = queue.sims[queue_num]
        with elapsed_time_printer(
            op_name=sim.chrom_set.label,
            op_number=(queue_num + 1, total_sims),
            chrom_set=sim.chrom_set,
        ):
            queue.model(sim.chrom_set, sim.starting_bins, sim.interf_scheme, sim.rand)
    print('all queued sims completed.')
