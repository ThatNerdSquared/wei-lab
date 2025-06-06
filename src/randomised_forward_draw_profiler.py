import cProfile
import timeit
from pstats import SortKey

from src.oneoffs.convert_otto_sims_to_randomised_forward_draw import simulate_bins, simulate_bins_arange

# cProfile.run(
#     "simulate_bins(lambda x, y, z: None)",
#     sort=SortKey.TIME
# )

list_impl_avg_time = timeit.timeit(lambda: simulate_bins(lambda x, y, z: None), number=50)
arange_impl_avg_time = timeit.timeit(lambda: simulate_bins_arange(lambda x, y, z: None), number=50)

print(f'list impl avg time (s): {list_impl_avg_time}')
print(f'arange impl avg time (s): {arange_impl_avg_time}')
