import cProfile
# import timeit
import numpy as np
from pstats import SortKey

from src.randomised_forward_draw.rfd_chromosome import BINS_PER_CHROMOSOME, InterfModifier, simulate_bins, simulate_bins_lazy

def interf_scaling(bin_loc: int, origin: int, p: float) -> float:
    return p * (1 - np.exp(-1 * np.abs(
        (bin_loc / BINS_PER_CHROMOSOME) - (origin / BINS_PER_CHROMOSOME)
    )))


def lazy_interf(loc: int):
    return InterfModifier(
        interval_start=0,
        interval_end=BINS_PER_CHROMOSOME,
        originator_loc=loc,
        modifier=interf_scaling
    )

def interf(co_loc, bins, bin_idxs):
    for idx in bin_idxs:
        bins[idx] = bins[idx] * (1 - np.exp(-1 * np.abs(
            (idx / BINS_PER_CHROMOSOME) - (co_loc / BINS_PER_CHROMOSOME)
        )))

bin_locs = np.arange(BINS_PER_CHROMOSOME)
def vectorised_interf(co_loc, bins, bin_idxs):
    bins = bins * (1 - np.exp(-1 * np.abs(
        (bin_locs / BINS_PER_CHROMOSOME) - (co_loc / BINS_PER_CHROMOSOME)
    )))

cProfile.run(
    'simulate_bins(interf)',
    # 'simulate_bins(vectorised_interf)',
    # 'simulate_bins_lazy(lazy_interf)',
    sort=SortKey.TIME
)

# curr_impl_avg_time = timeit.timeit(lambda: simulate_bins(lambda x, y, z: None), number=50)
# lazy_impl_avg_time = timeit.timeit(lambda: simulate_bins_lazy(
#     lambda loc: InterfModifier(interval_start=0, interval_end=0, modifier=lambda x, y: y)
# ), number=50)

# print(f'list impl avg time (s): {curr_impl_avg_time}')
# print(f'arange impl avg time (s): {lazy_impl_avg_time}')
