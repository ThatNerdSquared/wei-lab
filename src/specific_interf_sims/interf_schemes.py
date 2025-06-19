import numpy as np
import numpy.typing as npt

from typing import Protocol

type BinArray = npt.NDArray[np.float64]

class InterferenceScheme(Protocol):
    def __call__(self, co_loc: int, bins: BinArray) -> BinArray: ...

def no_interf(co_loc: int, bins: BinArray) -> BinArray:
    return bins

def local_interf(co_loc: int, bins: BinArray) -> BinArray:
    bins_per_chrom = len(bins)
    bin_locs = np.arange(bins_per_chrom)
    return bins * (1 - np.exp(-1 * np.abs(
        (bin_locs / bins_per_chrom) - (co_loc / bins_per_chrom)
    )))

def global_interf(co_loc: int, bins: BinArray) -> BinArray:
    return bins * 0.7

def telomere_based_regional_interf(co_loc: int, bins: BinArray) -> BinArray:
    bin_locs = np.arange(len(bins))
    return bins * (
        np.where(bin_locs <= co_loc, 0, 1)
        if co_loc < len(bins) / 2 else
        np.where(bin_locs >= co_loc, 0, 1)
    )

def equalised_global_interf(co_loc: int, bins: BinArray) -> BinArray:
    new_total_p_co = np.sum(local_interf(co_loc, bins))
    current_total_p_co = np.sum(bins)
    return bins * new_total_p_co / current_total_p_co
