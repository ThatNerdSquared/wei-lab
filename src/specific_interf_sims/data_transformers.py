from typing import Callable, Literal

import numpy as np
import numpy.typing as npt

from src.utils.chromatid import (
    Chromatid,
    get_crossover_locations,
    get_num_crossovers,
    has_crossover_at,
    has_visible_crossover,
)
from src.utils.chromosome_set import ChromosomeLike, ChromosomeSet, pool_iter, pool_size

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
        recomb_curve_data = recomb_curve_data / pool_size(chrom_set)

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
        dist_data = dist_data / pool_size(chrom_set)

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
        totals = totals / pool_size(chrom_set)

    return chrom_set.label, (totals, groups)


def transform_cM_rf_curve_data[C: ChromosomeLike](
    chrom_set: ChromosomeSet[C],
) -> tuple[str, tuple[list[float], list[float]]]:
    dataset_cM: list[float] = []
    dataset_rf: list[float] = []

    NUM_CHROMS_IN_POOL = pool_size(chrom_set)

    total_cos_so_far = 0
    for x in range(chrom_set.chrom_length):
        visible_COs = 0
        for chrom in pool_iter(chrom_set):
            if has_crossover_at(chrom, x):
                total_cos_so_far += 1
            if has_visible_crossover(chrom, 0, x):
                visible_COs += 1

        dataset_cM.append(total_cos_so_far / NUM_CHROMS_IN_POOL)
        dataset_rf.append(visible_COs / NUM_CHROMS_IN_POOL)

    return chrom_set.label, (dataset_cM, dataset_rf)


def transform_mean_cos_per_chrom_data[C: ChromosomeLike](
    chrom_set: ChromosomeSet[C],
) -> tuple[str, np.float64]:
    return chrom_set.label, np.float64(
        np.mean([get_num_crossovers(x) for x in chrom_set])
    )


def transform_co_haplotype_switch_data[C: ChromosomeLike](
    chrom_set: ChromosomeSet[C],
) -> tuple[
    str,
    tuple[Literal['non-haplotype-switching'], int],
    tuple[Literal['haplotype-switching'], int],
]:
    non_haplotype_switching_counter = 0
    haplotype_switching_counter = 0

    for chrom in pool_iter(chrom_set):
        assert isinstance(chrom, Chromatid)
        prev_haplotype = chrom.base_haplotype
        for co in chrom.crossovers:
            if co.haplotype == prev_haplotype:
                non_haplotype_switching_counter += 1
            else:
                haplotype_switching_counter += 1
            prev_haplotype = co.haplotype

    return (
        chrom_set.label,
        ('non-haplotype-switching', non_haplotype_switching_counter),
        ('haplotype-switching', haplotype_switching_counter),
    )
