from typing import Callable

import numpy as np
import numpy.typing as npt

from src.utils.chromosome_set import ChromosomeSet

type DataTransformer[T] = Callable[[ChromosomeSet], tuple[str, T]]


def transform_recomb_curve_data(
    chrom_set: ChromosomeSet, proportional: bool = True
) -> tuple[str, npt.NDArray[np.float64]]:
    recomb_curve_data = np.zeros(chrom_set.chrom_length)

    for loc in range(chrom_set.chrom_length):
        recomb_curve_data[loc] = sum(
            [1 if x.has_crossover_at(loc) else 0 for x in chrom_set]
        )

    if proportional:
        recomb_curve_data = recomb_curve_data / chrom_set.num_chroms

    return (chrom_set.label, recomb_curve_data)
