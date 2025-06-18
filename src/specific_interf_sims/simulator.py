import numpy as np
import numpy.typing as npt
from dataclasses import dataclass, field
from typing import Callable, Protocol

from src.utils.chromosome_set import ChromosomeSet
from src.utils.cli_utils import elapsed_time_printer

type BinArray = npt.NDArray[np.float64]

class InterferenceStrategy(Protocol):
    def __call__(self, co_loc: int, bins: BinArray) -> BinArray: ...

@dataclass(kw_only=True)
class InterferenceSim:
    label: str
    interf_strategy: InterferenceStrategy
    starting_bins: BinArray
    chrom_set: ChromosomeSet
    rand: np.random.Generator

@dataclass()
class SimulationQueue:
    model: Callable[[ChromosomeSet, BinArray, InterferenceStrategy, np.random.Generator], None]
    default_starting_bins: BinArray | None = None
    sims: list[InterferenceSim] = field(default_factory=list)
    rand_seed: int | None = None
    chroms_per_sim: int | None = None


def queue_sim(
    queue: SimulationQueue,
    label: str,
    interf_strategy: InterferenceStrategy,
    chrom_count: int | None = None,
    starting_bins: BinArray | None = None,
    provided_chroms: ChromosomeSet | None = None,
) -> ChromosomeSet:
    bins = starting_bins or queue.default_starting_bins
    num_chroms=chrom_count or queue.chroms_per_sim
    assert bins is not None
    assert num_chroms is not None

    chrom_set = provided_chroms or ChromosomeSet(
        num_chroms=num_chroms,
        chrom_length=len(bins)
    )

    sim = InterferenceSim(
        label=label,
        interf_strategy=interf_strategy,
        starting_bins=bins,
        chrom_set=chrom_set,
        rand=np.random.default_rng(queue.rand_seed)
    )
    queue.sims.append(sim)
    return chrom_set

def run_sims(queue: SimulationQueue):
    total_sims = len(queue.sims)
    print('simulating crossovers...')
    for queue_num in range(total_sims):
        sim = queue.sims[queue_num]
        with elapsed_time_printer(
            op_name=sim.label,
            op_number=(queue_num + 1, total_sims),
            chrom_set=sim.chrom_set
        ):
            queue.model(
                sim.chrom_set,
                sim.starting_bins,
                sim.interf_strategy,
                sim.rand
            )
    print('all queued sims completed.')
