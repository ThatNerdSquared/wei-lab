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

def gather_co_per_chrom_dist_data(
    chrom_set: ChromosomeSet,
    proportional: bool = True
) -> npt.NDArray[np.float64]:
    data_dict: dict[int, float] = {}

    for chrom in chrom_set:
        try:
            data_dict[chrom.get_num_crossovers()] += 1.0
        except KeyError:
            data_dict[chrom.get_num_crossovers()] = 1.0

    dist_data = np.array([
        data_dict.get(x, 0) for x in range(max(data_dict))
    ])

    if proportional:
        dist_data = dist_data / chrom_set.num_chroms

    return dist_data

def gather_interevent_dist_data(
    chrom_set: ChromosomeSet,
    proportional: bool = True
) -> dict[int, np.float64]:
    dist_data: dict[int, float] = {}

    for chrom in chrom_set:
        co_locs = chrom.get_crossover_locations()

        for loc_idx in range(1, len(co_locs)):
            interevent_dist = co_locs[loc_idx] - co_locs[loc_idx - 1] - 1
            try:
                dist_data[interevent_dist] += 1.0
            except KeyError:
                dist_data[interevent_dist] = 1.0

    if proportional:
        for k, v in dist_data.items():
            dist_data[k] = v / chrom_set.num_chroms

    return dict(sorted(dist_data.items()))

