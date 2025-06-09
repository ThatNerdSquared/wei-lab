import numpy as np
import numpy.typing as npt

from typing import Callable

from src.utils.chromosome_set import ChromosomeSet

type BinArray = npt.NDArray[np.float64]
type BinIdxArray = npt.NDArray[np.int64]


def induce_COs_via_random_forward_draw(
    chrom_set: ChromosomeSet,
    recomb_rate_init: Callable[[BinIdxArray], BinArray],
    interf: Callable[[int, BinArray], BinArray]
) -> None:
    bin_locs = np.arange(chrom_set.chrom_length)

    counter = 0
    for chrom in chrom_set:
        bins: BinArray = recomb_rate_init(bin_locs)
        chrom_set.rand.shuffle(bin_locs)
        for loc in bin_locs:
            if chrom_set.rand.binomial(1, bins[loc]):
                bins = interf(loc, bins)
                chrom.insert_crossover(loc)

        print(counter)
        counter += 1
