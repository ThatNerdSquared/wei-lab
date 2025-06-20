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

def global_interf(co_loc: int, bins: BinArray, strength: float = 0.7) -> BinArray:
    return bins * strength

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

def fixed_regional_interf(co_loc: int, bins: BinArray, strength_high: float = 0.8) -> BinArray:
    bins_per_chrom = len(bins)
    left_upper_bound, right_lower_bound = balanced_three_partition(bins_per_chrom)
    bin_locs = np.arange(len(bins))
    return bins * np.where(
        (bin_locs <= left_upper_bound) | (bin_locs >= right_lower_bound),
        strength_high,
        0.5 * strength_high
    )




def balanced_three_partition(num_to_divide: int) -> tuple[int, int]:
    '''
    returns a 2-tuple of form (largest index of first partition, smallest index
    of third partition).
    '''
    closest_third = num_to_divide // 3
    match num_to_divide % 3:
        case 0:
            return (closest_third - 1, closest_third * 2)
        case 1:
            return (closest_third - 1, closest_third * 2 + 1)
        case 2:
            return (closest_third, closest_third * 2 + 1)
        case _:
            raise ValueError(f'Could not divide by 3: {num_to_divide}')
