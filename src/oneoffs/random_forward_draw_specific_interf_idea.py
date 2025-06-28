import matplotlib.pyplot as plt
import numpy as np

from src.overall_interf_markov_chain.recomb_markov_chain import RecombMarkovChain
from src.utils.chromosome import Chromosome

NUM_CHROMOSOMES = 3000
AVG_COS_PER_CHROMOSOME = 0.5
BINS_PER_CHROMOSOME = 1000

rand = np.random.default_rng()
chroms: list[Chromosome] = []

for i in range(NUM_CHROMOSOMES):
    chrom = Chromosome(BINS_PER_CHROMOSOME)
    p_hit = AVG_COS_PER_CHROMOSOME / BINS_PER_CHROMOSOME
    bins = list(range(BINS_PER_CHROMOSOME))
    rand.shuffle(bins)
    for loc in bins:
        if rand.binomial(1, p_hit):
            p_hit *= 0.5
            chrom.insert_crossover(loc)
    chroms.append(chrom)

dataset: dict[int, int] = {}

for chrom in chroms:
    num = chrom.get_num_crossovers()
    try:
        dataset[num] += 1
    except KeyError:
        dataset[num] = 1

sorted_dataset = dict(sorted(dataset.items()))
print(sorted_dataset)

fig, ax = plt.subplots(sharex=True, sharey=True, figsize=(9.0, 7.0))
ax.bar(sorted_dataset.keys(), sorted_dataset.values(), width=0.3, label='Simulated')

x = range(9)
rmc = RecombMarkovChain(9)
rmc.reset_to_poisson(np.exp(-1 * AVG_COS_PER_CHROMOSOME))
print(rmc)

ax.bar(
    [n + 0.3 for n in x],
    [NUM_CHROMOSOMES * rmc.get_p_cos(n) for n in x],
    width=0.3,
    label='MC Default',
)

rmc.on_all_onward_transitions(rmc.scale_transition_probability, 0.5, 1)

print(rmc)
ax.bar(
    [n + 0.6 for n in x],
    [NUM_CHROMOSOMES * rmc.get_p_cos(n) for n in x],
    width=0.3,
    label='MC Interf',
)

ax.set_xlabel('# of COs')
ax.set_ylabel('num of chromosomes')
ax.legend()
plt.savefig('out--random_forward_draw_specific_interf_idea.png')
plt.show()
