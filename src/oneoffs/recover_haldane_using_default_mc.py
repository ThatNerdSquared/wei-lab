import matplotlib.pyplot as plt
import numpy as np
from croissant.chromosomes import LegacyChromosome

from src.overall_interf_markov_chain.recomb_markov_chain import RecombMarkovChain
from src.utils.standard_funcs import haldane_input_cM

NUM_STATES = 25
NUM_CHROMS = 3000
CHROM_LENGTH = 500

chromosomes: list[LegacyChromosome] = []

rmc = RecombMarkovChain(NUM_STATES)
rmc.reset_to_poisson(0.1)

for state in range(NUM_STATES):
    allocation = round(rmc.get_p_cos(state) * NUM_CHROMS)
    for _ in range(allocation):
        chrom = LegacyChromosome(CHROM_LENGTH)
        chrom.randomly_add_crossovers(state)
        chromosomes.append(chrom)

dataset_cM: list[float] = []
dataset_rf: list[float] = []
total_cos_so_far = 0

for endpoint in range(CHROM_LENGTH):
    visible_COs = 0
    for chrom in chromosomes:
        if chrom.has_crossover_at(endpoint):
            total_cos_so_far += 1
        if chrom.has_visible_crossover(0, endpoint):
            visible_COs += 1

    dataset_cM.append(total_cos_so_far / NUM_CHROMS)
    dataset_rf.append(visible_COs / NUM_CHROMS)

# for a, b in zip(dataset_cM, dataset_rf):
#     print(a, b)

fig, ax = plt.subplots(sharex=True, sharey=True, figsize=(9.0, 7.0))
_ = ax.plot(dataset_cM, dataset_rf, label='MC Default')
x = np.linspace(0, 2, 1000)
_ = ax.plot(x, [haldane_input_cM(n) for n in x], label='Haldane')
# ax.plot(x, x, label='Morgan')
_ = ax.set_xlabel('morgans (M)')
_ = ax.set_ylabel('recombination frequency (rf)')
_ = ax.legend()
plt.savefig('out--recover_haldane_using_default_mc.png')
plt.show()
