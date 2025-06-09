import numpy as np
import numpy.typing as npt

from dataclasses import dataclass, field

from src.utils.chromosome import Chromosome


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

def gather_recomb_curve_data(
    chrom_set: ChromosomeSet,
    proportional: bool = True
) -> npt.NDArray[np.float64]:
    recomb_curve_data = np.zeros(chrom_set.chrom_length)

    for loc in range(chrom_set.chrom_length):
        recomb_curve_data[loc] = sum(
            [1 if x.has_crossover_at(loc) else 0 for x in chrom_set]
        )

    if proportional:
        recomb_curve_data = recomb_curve_data / chrom_set.num_chroms

    return recomb_curve_data
