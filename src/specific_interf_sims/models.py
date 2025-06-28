import numpy as np
import numpy.typing as npt

from src.specific_interf_sims.interf_schemes import BinArray, InterferenceScheme
from src.utils.chromosome_set import ChromosomeSet

type BinIdxArray = npt.NDArray[np.int64]


def induce_COs_via_random_forward_draw(
    chrom_set: ChromosomeSet,
    starting_bins: BinArray,
    interf: InterferenceScheme,
    rand: np.random.Generator,
) -> None:
    bin_locs = np.arange(chrom_set.chrom_length)

    for chrom in chrom_set:
        bins: BinArray = starting_bins.copy()
        rand.shuffle(bin_locs)
        for loc in bin_locs:
            if rand.binomial(1, bins[loc]):
                bins = interf(loc, bins)
                chrom.insert_crossover(loc)
