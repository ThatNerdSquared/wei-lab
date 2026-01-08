from typing import Callable, Literal, NamedTuple

import numpy as np
import numpy.typing as npt

from src.utils.chromatid import (
    Chromatid,
    convert_chromatid_to_haplotype_tbl,
    get_crossover_locations,
    get_num_crossovers,
    has_crossover_at,
    has_visible_crossover,
)
from src.utils.chromosome_set import ChromosomeLike, ChromosomeSet, pool_iter, pool_size
from src.utils.cli_utils import elapsed_time_printer

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


def transform_cM_rf_curve_data_haplotype_switching_only[C: ChromosomeLike](
    chrom_set: ChromosomeSet[C],
) -> tuple[str, tuple[list[float], list[float]]]:
    dataset_cM: list[float] = []
    dataset_rf: list[float] = []

    NUM_CHROMS_IN_POOL = pool_size(chrom_set)

    total_cos_so_far = 0
    for x in range(chrom_set.chrom_length):
        visible_COs = 0
        for chrom in pool_iter(chrom_set):
            if has_crossover_at(chrom, x) and has_visible_crossover(chrom, x - 1, x):
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


def transform_mean_cos_per_chrom_in_pool_data[C: ChromosomeLike](
    chrom_set: ChromosomeSet[C],
) -> tuple[str, np.float64]:
    return chrom_set.label, np.float64(
        np.mean([get_num_crossovers(x) for x in pool_iter(chrom_set)])
    )


def transform_variance_cos_per_chrom_data[C: ChromosomeLike](
    chrom_set: ChromosomeSet[C],
) -> tuple[str, np.float64]:
    return chrom_set.label, np.float64(
        np.var([get_num_crossovers(x) for x in chrom_set])
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


class Region(NamedTuple):
    start: int
    end: int


DEFAULT_REF_REGION = Region(0, 9)


def transform_single_refregion_coc_curve_data[C: ChromosomeLike](
    chrom_set: ChromosomeSet[C],
    ref_region: Region = DEFAULT_REF_REGION,
    min_testregion_size: int = 1,
    calculate_inside_refregion: bool = False,
) -> tuple[str, np.ndarray[tuple[int], np.dtype[np.float64]]]:
    assert ref_region.start <= ref_region.end
    NUM_CHROMS_IN_POOL = pool_size(chrom_set)
    haplotype_tbl = np.zeros(
        (NUM_CHROMS_IN_POOL, chrom_set.chrom_length), dtype=np.int64
    )
    with elapsed_time_printer(op_name='haplotype tbl build'):
        for idx, chrom in enumerate(pool_iter(chrom_set)):
            assert isinstance(chrom, Chromatid)
            convert_chromatid_to_haplotype_tbl(chrom, haplotype_tbl[idx])

    proportion_refregion_visible_co = (
        np.sum(haplotype_tbl[:, ref_region.start] ^ haplotype_tbl[:, ref_region.end])
        / NUM_CHROMS_IN_POOL
    )

    haplotype_tbl[:, ref_region.end + 1 :] = haplotype_tbl[
        :, ref_region.end + 1 :
    ] ^ np.expand_dims(haplotype_tbl[:, ref_region.end], axis=1)
    haplotype_tbl[:, : ref_region.start] = haplotype_tbl[
        :, : ref_region.start
    ] ^ np.expand_dims(haplotype_tbl[:, ref_region.start], axis=1)
    numerator_unproportionalised = (
        np.expand_dims(
            haplotype_tbl[:, ref_region.start] ^ haplotype_tbl[:, ref_region.end],
            axis=1,
        )
        & haplotype_tbl
    )
    numerator_proportionalised = (
        np.sum(numerator_unproportionalised, axis=0) / NUM_CHROMS_IN_POOL
    )
    coc_data = numerator_proportionalised / (
        proportion_refregion_visible_co
        * (np.sum(haplotype_tbl, axis=0) / NUM_CHROMS_IN_POOL)
        + 1e-10
    )

    if calculate_inside_refregion:
        has_visible_co_with_start = haplotype_tbl[
            :, ref_region.start : ref_region.end + 1
        ] ^ np.expand_dims(haplotype_tbl[:, ref_region.start], axis=1)
        has_visible_co_with_end = haplotype_tbl[
            :, ref_region.start : ref_region.end + 1
        ] ^ np.expand_dims(haplotype_tbl[:, ref_region.end], axis=1)
        numerator_for_inside_refregion = (
            np.sum(has_visible_co_with_start & has_visible_co_with_end, axis=0)
            / NUM_CHROMS_IN_POOL
        )
        denominator_for_inside_refregion = (
            np.sum(has_visible_co_with_start, axis=0) / NUM_CHROMS_IN_POOL
        ) * (np.sum(has_visible_co_with_end, axis=0) / NUM_CHROMS_IN_POOL)
        coc_data[ref_region.start : ref_region.end + 1] = (
            numerator_for_inside_refregion / (denominator_for_inside_refregion + 1e-10)
        )
        coc_data[
            max(0, ref_region.start - (min_testregion_size - 1)) : ref_region.start
            + min_testregion_size
        ] = -1
        coc_data[
            ref_region.end - (min_testregion_size - 1) : min(
                chrom_set.chrom_length, ref_region.end + min_testregion_size
            )
        ] = -1
    else:
        coc_data[
            max(0, ref_region.start - (min_testregion_size - 1)) : ref_region.end
            + 1
            + (min_testregion_size - 1)
        ] = -1
    return chrom_set.label, coc_data


def transform_sliding_compare_region_coc_curve_data[C: ChromosomeLike](
    chrom_set: ChromosomeSet[C],
    ref_region: Region = DEFAULT_REF_REGION,
    sliding_region_size: int = 10,
) -> tuple[str, np.ndarray[tuple[int], np.dtype[np.float64]]]:
    assert ref_region.start <= ref_region.end
    NUM_CHROMS_IN_POOL = pool_size(chrom_set)
    haplotype_tbl = np.zeros(
        (NUM_CHROMS_IN_POOL, chrom_set.chrom_length), dtype=np.int64
    )
    with elapsed_time_printer(op_name='haplotype tbl build'):
        for idx, chrom in enumerate(pool_iter(chrom_set)):
            assert isinstance(chrom, Chromatid)
            convert_chromatid_to_haplotype_tbl(chrom, haplotype_tbl[idx])

    proportion_refregion_visible_co = (
        np.sum(haplotype_tbl[:, ref_region.start] ^ haplotype_tbl[:, ref_region.end])
        / NUM_CHROMS_IN_POOL
    )

    if sliding_region_size <= ref_region.start + 1:
        haplotype_tbl[:, : ref_region.start - sliding_region_size + 2] = (
            haplotype_tbl[:, : ref_region.start - sliding_region_size + 2]
            ^ haplotype_tbl[:, sliding_region_size - 1 : ref_region.start + 1]
        )
        haplotype_tbl[
            :, ref_region.start - sliding_region_size + 2 : ref_region.start
        ] = -2
    else:
        print('skipping area left of refregion start due to no space.')
        haplotype_tbl[:, : ref_region.start] = -2

    if sliding_region_size < chrom_set.chrom_length - ref_region.end:
        haplotype_tbl[
            :, ref_region.end + 1 : chrom_set.chrom_length - sliding_region_size + 1
        ] = (
            haplotype_tbl[
                :, ref_region.end + 1 : chrom_set.chrom_length - sliding_region_size + 1
            ]
            ^ haplotype_tbl[:, ref_region.end + sliding_region_size :]
        )
        haplotype_tbl[:, chrom_set.chrom_length - sliding_region_size + 1 :] = -2
    else:
        print('skipping area right of refregion start due to no space.')
        haplotype_tbl[:, ref_region.end + 1 :] = -2

    numerator_unproportionalised = (
        np.expand_dims(
            haplotype_tbl[:, ref_region.start] ^ haplotype_tbl[:, ref_region.end],
            axis=1,
        )
        & haplotype_tbl
    )
    numerator_proportionalised = (
        np.sum(numerator_unproportionalised, axis=0) / NUM_CHROMS_IN_POOL
    )
    coc_data = numerator_proportionalised / (
        proportion_refregion_visible_co
        * (np.sum(haplotype_tbl, axis=0) / NUM_CHROMS_IN_POOL)
        + 1e-10
    )
    coc_data[ref_region.start : ref_region.end + 1] = -2
    return chrom_set.label, coc_data


def transform_sliding_refregion_coc_curve_data[C: ChromosomeLike](
    chrom_set: ChromosomeSet[C], region_size: int = 10, step_size: int = 1
) -> tuple[str, np.ndarray[tuple[int, int], np.dtype[np.float64]]]:
    result = np.zeros((chrom_set.chrom_length - region_size, chrom_set.chrom_length))
    result -= 2
    print(result)
    for region_start in range(0, chrom_set.chrom_length - region_size, step_size):
        result[region_start] = transform_single_refregion_coc_curve_data(
            chrom_set, ref_region=Region(region_start, region_start + region_size)
        )[1]

    assert -2 not in result
    return chrom_set.label, result
