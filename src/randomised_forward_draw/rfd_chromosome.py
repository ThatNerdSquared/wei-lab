from dataclasses import dataclass, field
import numpy as np
from numpy._core.numerictypes import float64, int64
import numpy.typing as npt

from typing import Callable, Optional

from src.utils.chromosome import Chromosome

type BinArray = npt.NDArray[float64]
type BinIdxArray = npt.NDArray[int64]

class RandomForwardDrawChromosome(Chromosome):
    def __init__(
        self,
        length: int,
        rng: np.random.Generator,
        base_p_hit: Callable[[int], float64],
        interf: Callable[[int, BinArray, BinIdxArray], BinArray]
    ):
        self.bin_idxs: BinIdxArray = np.arange(length)
        self.bins: BinArray = base_p_hit(self.bin_idxs)
        self.rand: np.random.Generator = rng
        self.interf = interf
        super().__init__(length)

    def perform_randomised_forward_draw(self):
        order = rand.permutation(self.bin_idxs)
        for loc in order:
            if rand.binomial(1, self.bins[loc]):
                self.bins = self.interf(loc, self.bins, self.bin_idxs)
                self.insert_crossover(loc)

@dataclass(kw_only=True, slots=True)
class ChromosomeSet():
    num_chroms: int
    chrom_length: int
    rand_seed: int | None = None
    rand: np.random.Generator = field(init=False)
    chroms: list[Chromosome] = field(init=False)

    def __post_init__(self):
        self.rand = np.random.default_rng(self.rand_seed)
        self.chroms = [
            Chromosome(self.chrom_length) for _ in range(self.num_chroms)
        ]

    def __iter__(self):
        return self.chroms.__iter__()


def induce_COs_via_random_forward_draw(
    chrom_set: ChromosomeSet,
    recomb_rate_init: Callable[[BinIdxArray], BinArray],
    interf: Callable[[int, BinArray], BinArray]
) -> None:
    bin_locs = np.arange(chrom_set.chrom_length)

    counter = 0
    for chrom in chrom_set:
        bins: BinArray = recomb_rate_init(bin_locs)
        rand.shuffle(bin_locs)
        for loc in bin_locs:
            if rand.binomial(1, bins[loc]):
                bins = interf(loc, bins)
                chrom.insert_crossover(loc)

        print(counter)
        counter += 1

def gather_recomb_curve_data(
    chrom_set: ChromosomeSet,
    proportional: bool = True
) -> npt.NDArray[float64]:
    recomb_curve_data = np.zeros(chrom_set.chrom_length)

    for loc in range(chrom_set.chrom_length):
        recomb_curve_data[loc] = sum(
            [1 if x.has_crossover_at(loc) else 0 for x in chrom_set]
        )

    if proportional:
        recomb_curve_data = recomb_curve_data / chrom_set.num_chroms

    return recomb_curve_data


NUM_CHROMOSOMES = 1000
BINS_PER_CHROMOSOME = 300000
AVG_COS_PER_CHROMOSOME = 0.5
STARTING_P_HIT = AVG_COS_PER_CHROMOSOME / BINS_PER_CHROMOSOME
rand = np.random.default_rng(54321)


def simulate_bins(react_to_co: Callable[
    [int, npt.NDArray[np.float64], list[int]],
    None
]) -> list[Chromosome]:
    chroms: list[Chromosome] = []
    for _ in range(NUM_CHROMOSOMES):
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
