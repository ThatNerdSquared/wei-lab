import matplotlib.pyplot as plt
import numpy as np
from numpy._core.numerictypes import int64

NUM_SIMS = 1000
NUM_OF_CHROMOSOMES = 100
BINS_PER_CHROMOSOME = 550000


rng = np.random.default_rng(77777)


def sim(avg_cos_per_chrom: int, result: dict[int, int]) -> dict[int, int]:
    chroms = rng.poisson(avg_cos_per_chrom, NUM_OF_CHROMOSOMES)
    event_space: set[int] = {
        0
    }  # need to start counting distance to the left of 1st crossover

    allocation: int64
    for allocation in chroms:
        locations = rng.integers(0, BINS_PER_CHROMOSOME, allocation)
        event_space.update(locations)

    sorted_event_space = sorted(list(event_space))
    for space_idx in range(1, len(sorted_event_space)):
        interevent_dist = (
            sorted_event_space[space_idx] - sorted_event_space[space_idx - 1] - 1
        )
        try:
            result[interevent_dist] += 1
        except KeyError:
            result[interevent_dist] = 1

    return dict(sorted(result.items()))


print('simulating...')
result = {}
for count in range(NUM_SIMS):
    result = sim(2, result)

print('processing data...')
total_interevents = sum(result.values())
x_range = range(int(BINS_PER_CHROMOSOME / 2))
y_range = [result[x] / total_interevents if x in result else 0 for x in x_range]

probability_counter = 0
fifty_threshold = 0
ninety_five_threshold = 0
for i in x_range:
    probability_counter += y_range[i]
    if fifty_threshold == 0 and probability_counter > 0.5:
        fifty_threshold = i
        print(f'50% of interevent distances are: ≤{fifty_threshold / 10}kb')
    if probability_counter > 0.95:
        ninety_five_threshold = i
        print(f'95% of interevent distances are: ≤{ninety_five_threshold / 10}kb')
        break

print('plotting...')
fig, ax = plt.subplots(sharex=True, sharey=True, figsize=(9.0, 7.0))
ax.bar(x_range, y_range)
plt.axvline(
    x=fifty_threshold, color='tab:orange', label=f'50%: ≤{fifty_threshold / 10}kb'
)
plt.axvline(
    x=ninety_five_threshold,
    color='tab:red',
    label=f'95%: ≤{ninety_five_threshold / 10}kb',
)
plt.legend()
plt.show()
