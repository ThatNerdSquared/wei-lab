import numpy as np

from src.specific_interf_sims.data_transformers import transform_mean_cos_per_chrom_data
from src.specific_interf_sims.interf_schemes import local_interf, no_interf
from src.specific_interf_sims.models import induce_COs_via_random_forward_draw
from src.specific_interf_sims.simulator import SimulationQueue, queue_sim, run_sims
from src.utils.chromosome_set import ChromosomeSet, make_default_chromset
from src.utils.chromosome_with_sister_chromatids import ChromosomeSisterChromatids

NUM_CHROMS = 100000
CHROM_LEN = 1000

queue = SimulationQueue(
    induce_COs_via_random_forward_draw,
    rand_seed=93719,
    chroms_per_sim=NUM_CHROMS,
    default_chromset_maker=make_default_chromset,
)

type _CSet = ChromosomeSet[ChromosomeSisterChromatids]
chrom_sets: list[tuple[_CSet, _CSet]] = []

# for avg_cos_per_chrom in range(2, 11):
for avg_cos_per_chrom in [x / 10 for x in range(70, 81)]:
    chrom_sets.append(
        (
            queue_sim(
                queue,
                f'no interf ({avg_cos_per_chrom})',
                no_interf,
                starting_bins=np.zeros(CHROM_LEN) + avg_cos_per_chrom / CHROM_LEN,
            ),
            queue_sim(
                queue,
                f'local ({avg_cos_per_chrom})',
                local_interf,
                starting_bins=np.zeros(CHROM_LEN) + avg_cos_per_chrom / CHROM_LEN,
            ),
        )
    )

run_sims(queue)

for chromsets_for_avg in chrom_sets:
    no_interf_data, local_interf_data = map(
        transform_mean_cos_per_chrom_data, chromsets_for_avg
    )
    print(no_interf_data, local_interf_data)
