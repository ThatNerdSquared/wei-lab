from typing import Callable

import numpy as np
import numpy.typing as npt

from src.utils.chromatid import has_crossover_at
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
