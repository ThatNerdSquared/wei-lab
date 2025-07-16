from typing import Callable

import numpy as np
import numpy.typing as npt

from src.utils.chromatid import (
    get_crossover_locations,
    get_num_crossovers,
    has_crossover_at,
    has_visible_crossover,
)
from src.utils.chromosome_set import ChromosomeLike, ChromosomeSet, pool_iter

type DataTransformer[T, C: ChromosomeLike] = Callable[[ChromosomeSet[C]], tuple[str, T]]


def transform_recomb_curve_data[C: ChromosomeLike](
    chrom_set: ChromosomeSet[C], proportional: bool = True
) -> tuple[str, npt.NDArray[np.float64]]:
    recomb_curve_data = np.zeros(chrom_set.chrom_length)

    for loc in range(chrom_set.chrom_length):
        recomb_curve_data[loc] = sum(
            [1 if has_crossover_at(x, loc) else 0 for x in pool_iter(chrom_set)]
        )

    if proportional:
        recomb_curve_data = recomb_curve_data / chrom_set.num_chroms

    return (chrom_set.label, recomb_curve_data)


def transform_co_per_chrom_dist_data[C: ChromosomeLike](
    chrom_set: ChromosomeSet[C], proportional: bool = True
) -> tuple[str, npt.NDArray[np.float64]]:
    data_dict: dict[int, float] = {}

    for chrom in pool_iter(chrom_set):
        try:
            data_dict[get_num_crossovers(chrom)] += 1.0
        except KeyError:
            data_dict[get_num_crossovers(chrom)] = 1.0

    dist_data = np.array([data_dict.get(x, 0) for x in range(max(data_dict))])

    if proportional:
        dist_data = dist_data / chrom_set.num_chroms

    result = chrom_set.label, dist_data
    print(result)
    return result


def transform_interevent_dist_data[C: ChromosomeLike](
    chrom_set: ChromosomeSet[C], proportional: bool = True
) -> tuple[str, tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]]:
    dist_data: list[float] = []

    for chrom in pool_iter(chrom_set):
        co_locs = get_crossover_locations(chrom)

        for loc_idx in range(1, len(co_locs)):
            interevent_dist = co_locs[loc_idx] - co_locs[loc_idx - 1] - 1
            dist_data.append(interevent_dist)

    totals, groups = np.histogram(dist_data, bins='auto')

    if proportional:
        totals = totals / chrom_set.num_chroms

    return chrom_set.label, (totals, groups)


def transform_cM_rf_curve_data[C: ChromosomeLike](
    chrom_set: ChromosomeSet[C],
) -> tuple[str, tuple[list[float], list[float]]]:
    dataset_cM: list[float] = []
    dataset_rf: list[float] = []

    total_cos_so_far = 0
    for x in range(chrom_set.chrom_length):
        visible_COs = 0
        for chrom in pool_iter(chrom_set):
            if has_crossover_at(chrom, x):
                total_cos_so_far += 1
            if has_visible_crossover(chrom, 0, x):
                visible_COs += 1

        dataset_cM.append(total_cos_so_far / chrom_set.num_chroms)
        dataset_rf.append(visible_COs / chrom_set.num_chroms)

    return chrom_set.label, (dataset_cM, dataset_rf)


def transform_mean_cos_per_chrom_data[C: ChromosomeLike](
    chrom_set: ChromosomeSet[C],
) -> tuple[str, np.float64]:
    return chrom_set.label, np.float64(
        np.mean([get_num_crossovers(x) for x in chrom_set])
    )
