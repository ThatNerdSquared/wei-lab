import matplotlib.pyplot as plt
import numpy as np

from numpy._core.numerictypes import int64

NUM_SIMS = 1000
NUM_OF_CHROMOSOMES = 100
BINS_PER_CHROMOSOME = 550000


rng = np.random.default_rng(77777)

def sim(avg_cos_per_chrom: int, result: dict[int, int]) -> dict[int, int]:
    chroms = rng.poisson(avg_cos_per_chrom, NUM_OF_CHROMOSOMES)
    event_space = np.zeros(BINS_PER_CHROMOSOME)

    allocation: int64
    for allocation in chroms:
        locations = rng.integers(0, BINS_PER_CHROMOSOME, allocation)
        for loc in locations:
            loc: int
            event_space[loc] += 1

    counter = 0
    for space in event_space:
        if space == 0:
            counter += 1
        else:
            if counter == 0: continue

            try:
                result[counter] += 1
            except KeyError:
                result[counter] = 1
            counter = 0

    return dict(sorted(result.items()))

result = {}
for count in range(NUM_SIMS):
    print(count)
    result = sim(2, result)


print("help!!")
fig, ax = plt.subplots(sharex=True, sharey=True, figsize=(9.0, 7.0))
x_range = range(int(BINS_PER_CHROMOSOME / 2))
total_interevents = sum(result.values())
ax.bar(x_range, [result[x] / total_interevents if x in result else 0 for x in x_range])
plt.show()
