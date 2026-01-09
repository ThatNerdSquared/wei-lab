import numpy as np
import numpy.typing as npt

from src.specific_interf_sims.assurance_schemes import AssuranceScheme
from src.specific_interf_sims.interf_schemes import BinArray, InterferenceScheme
from src.utils.chromatid import insert_crossover
from src.utils.chromosome_set import ChromosomeLike, ChromosomeSet

type BinIdxArray = npt.NDArray[np.int64]


def induce_COs_via_random_forward_draw[C: ChromosomeLike](
    chrom_set: ChromosomeSet[C],
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
                insert_crossover(chrom, loc)


def induce_COs_via_random_forward_draw_w_assurance[C: ChromosomeLike](
    chrom_set: ChromosomeSet[C],
    starting_bins: BinArray,
    interf: InterferenceScheme,
    rand: np.random.Generator,
    *,
    assurance: AssuranceScheme,
) -> None:
    bin_locs = np.arange(chrom_set.chrom_length)

    for chrom in chrom_set:
        assurance.reset()
        bins: BinArray = starting_bins.copy()
        rand.shuffle(bin_locs)
        for loc in bin_locs:
            if rand.binomial(1, assurance.modify_co_probability(bins[loc])):
                bins = interf(loc, bins)
                insert_crossover(chrom, loc)
                assurance.handle_co_succeeded()
            else:
                assurance.handle_co_failed()
