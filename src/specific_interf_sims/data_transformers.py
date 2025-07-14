from typing import Callable

import numpy as np
import numpy.typing as npt

from src.utils.chromatid import has_crossover_at, has_visible_crossover
from src.utils.chromosome_set import ChromosomeLike, ChromosomeSet

type DataTransformer[T, C: ChromosomeLike] = Callable[[ChromosomeSet[C]], tuple[str, T]]


def transform_recomb_curve_data[C: ChromosomeLike](
    chrom_set: ChromosomeSet[C], proportional: bool = True
) -> tuple[str, npt.NDArray[np.float64]]:
    recomb_curve_data = np.zeros(chrom_set.chrom_length)

    for loc in range(chrom_set.chrom_length):
        recomb_curve_data[loc] = sum(
            [1 if has_crossover_at(x, loc) else 0 for x in chrom_set]
        )

    if proportional:
        recomb_curve_data = recomb_curve_data / chrom_set.num_chroms

    return (chrom_set.label, recomb_curve_data)


def transform_cM_rf_curve_data[C: ChromosomeLike](
    chrom_set: ChromosomeSet[C],
) -> tuple[str, tuple[list[float], list[float]]]:
    dataset_cM: list[float] = []
    dataset_rf: list[float] = []

    total_cos_so_far = 0
    for x in range(chrom_set.chrom_length):
        visible_COs = 0
        for chrom in chrom_set:
            if has_crossover_at(chrom, x):
                total_cos_so_far += 1
            if has_visible_crossover(chrom, 0, x):
                visible_COs += 1

        dataset_cM.append(total_cos_so_far / chrom_set.num_chroms)
        dataset_rf.append(visible_COs / chrom_set.num_chroms)

    return chrom_set.label, (dataset_cM, dataset_rf)
