import numpy as np
import numpy.typing as npt

from dataclasses import dataclass, field

from src.specific_interf_sims.interf_schemes import BinArray
from src.utils.chromosome import Chromosome


@dataclass(kw_only=True, slots=True)
class ChromosomeSet:
    label: str
    num_chroms: int
    chrom_length: int
    chroms: list[Chromosome] = field(init=False)

    def __post_init__(self):
        self.chroms = [Chromosome(self.chrom_length) for _ in range(self.num_chroms)]

    def __iter__(self):
        return self.chroms.__iter__()


def gather_recomb_curve_data(
    chrom_set: ChromosomeSet, proportional: bool = True
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
    chrom_set: ChromosomeSet, proportional: bool = True
) -> npt.NDArray[np.float64]:
    data_dict: dict[int, float] = {}

    for chrom in chrom_set:
        try:
            data_dict[chrom.get_num_crossovers()] += 1.0
        except KeyError:
            data_dict[chrom.get_num_crossovers()] = 1.0

    dist_data = np.array([data_dict.get(x, 0) for x in range(max(data_dict))])

    if proportional:
        dist_data = dist_data / chrom_set.num_chroms

    return dist_data


def gather_interevent_dist_data(
    chrom_set: ChromosomeSet, proportional: bool = True
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    dist_data: list[float] = []

    for chrom in chrom_set:
        co_locs = chrom.get_crossover_locations()

        for loc_idx in range(1, len(co_locs)):
            interevent_dist = co_locs[loc_idx] - co_locs[loc_idx - 1] - 1
            dist_data.append(interevent_dist)

    totals, groups = np.histogram(dist_data, bins='auto')

    if proportional:
        totals = totals / chrom_set.num_chroms

    return (totals, groups)


def gather_cM_rf_curve_data(
    chrom_set: ChromosomeSet,
) -> tuple[list[float], list[float]]:
    dataset_cM: list[float] = []
    dataset_rf: list[float] = []

    total_cos_so_far = 0
    for x in range(chrom_set.chrom_length):
        visible_COs = 0
        for chrom in chrom_set:
            if chrom.has_crossover_at(x):
                total_cos_so_far += 1
            if chrom.has_visible_crossover(0, x):
                visible_COs += 1

        dataset_cM.append(total_cos_so_far / chrom_set.num_chroms)
        dataset_rf.append(visible_COs / chrom_set.num_chroms)

    return (dataset_cM, dataset_rf)


def gather_preinterf_cM_rf_curve_data(
    chrom_set: ChromosomeSet, pre_interf_dist: BinArray
) -> tuple[list[float], list[float]]:
    dataset_cM: list[float] = []
    dataset_rf: list[float] = []

    cM_so_far = 0
    for x in range(chrom_set.chrom_length):
        cM_so_far += pre_interf_dist[x]
        visible_COs = 0
        for chrom in chrom_set:
            if chrom.has_visible_crossover(end=x):
                visible_COs += 1

        dataset_cM.append(cM_so_far)
        dataset_rf.append(visible_COs / chrom_set.num_chroms)

    return (dataset_cM, dataset_rf)
