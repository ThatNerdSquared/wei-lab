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
    starting_bins: BinArray | None = None
    interf_strategy: InterferenceStrategy
    chrom_set: ChromosomeSet

@dataclass()
class SimulationQueue:
    model: Callable[[ChromosomeSet, BinArray, InterferenceStrategy], None]
    default_starting_bins: BinArray
    sims: list[InterferenceSim] = field(default_factory=list)
    # rand_seed: int


def queue_sim(queue: SimulationQueue, sim: InterferenceSim):
    queue.sims.append(sim)

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
                sim.starting_bins or queue.default_starting_bins,
                sim.interf_strategy
            )
