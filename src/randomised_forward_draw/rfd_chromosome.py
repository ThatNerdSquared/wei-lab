from dataclasses import dataclass
import numpy as np
import numpy.typing as npt

from typing import Callable

from src.utils.chromosome import Chromosome


NUM_CHROMOSOMES = 100
BINS_PER_CHROMOSOME = 300000
AVG_COS_PER_CHROMOSOME = 0.5
STARTING_P_HIT = AVG_COS_PER_CHROMOSOME / BINS_PER_CHROMOSOME
rand = np.random.default_rng(54321)


def simulate_bins(react_to_co: Callable[
    [int, npt.NDArray[np.float64], list[int]],
    None
]) -> list[Chromosome]:
    chroms: list[Chromosome] = []
    for i in range(NUM_CHROMOSOMES):
        print(i)
        chrom = Chromosome(BINS_PER_CHROMOSOME)
        bins = np.zeros(BINS_PER_CHROMOSOME)
        bins += STARTING_P_HIT

        bin_idxs = list(range(BINS_PER_CHROMOSOME))
        rand.shuffle(bin_idxs)

        for loc in bin_idxs:
            if rand.binomial(1, bins[loc]):
                react_to_co(loc, bins, bin_idxs)
                chrom.insert_crossover(loc)

        chroms.append(chrom)

    return chroms


# (location: int, origin: int, probability: float) -> new_probability: float
type ModifierCallback = Callable[[int, int, float], float]

@dataclass
class InterfModifier:
    interval_start: int
    interval_end: int
    originator_loc: int
    modifier: ModifierCallback

def simulate_bins_lazy(
    react_to_co: Callable[[int], InterfModifier]
) -> list[Chromosome]:
    chroms: list[Chromosome] = []
    interf_modifiers: list[InterfModifier] = []

    for i in range(NUM_CHROMOSOMES):
        print(i)
        chrom = Chromosome(BINS_PER_CHROMOSOME)
        bin_idxs = list(range(BINS_PER_CHROMOSOME))
        rand.shuffle(bin_idxs)

        for loc in bin_idxs:
            p_hit = compute_p_hit(loc, interf_modifiers)
            if rand.binomial(1, p_hit):
                interf_modifiers.append(
                    react_to_co(loc)
                )
                chrom.insert_crossover(loc)

        chroms.append(chrom)

    return chroms

def compute_p_hit(loc: int, interf_modifiers: list[InterfModifier]) -> float:
    result = STARTING_P_HIT
    for mod in interf_modifiers:
        if loc >= mod.interval_start and loc <= mod.interval_end:
            result = mod.modifier(loc, mod.originator_loc, result)

    return result
