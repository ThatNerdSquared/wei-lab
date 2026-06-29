import numpy as np
from croissant.chromosomes import (
    haplotypeSwitchingCOsOnlySelector,
    make_chromset_custom_sister_selectors,
    selectSingleSisterForPool,
)
from croissant.chromosomes.core import Segments

from src.specific_interf_sims.interf_schemes import no_interf
from src.specific_interf_sims.models import induce_COs_via_random_forward_draw
from src.specific_interf_sims.simulator import SimulationQueue, queue_sim, run_sims

NUM_CHROMS = 100000
CHROM_LEN = 1000
AVG_COS_PER_CHROMOSOME = 2
STARTING_P_HIT = AVG_COS_PER_CHROMOSOME / CHROM_LEN
RAND_SEED = 54321
RAND_FOR_SELECTORS = np.random.default_rng(RAND_SEED)


no_drive_chroms = make_chromset_custom_sister_selectors(
    label='no drive',
    num_chroms=NUM_CHROMS,
    chrom_length=CHROM_LEN,
    co_selector=haplotypeSwitchingCOsOnlySelector,
    pool_selector=selectSingleSisterForPool,
)

queue = SimulationQueue(
    induce_COs_via_random_forward_draw,
    np.zeros(CHROM_LEN) + STARTING_P_HIT,
    rand_seed=RAND_SEED,
    chroms_per_sim=NUM_CHROMS,
)
_ = queue_sim(
    queue,
    no_drive_chroms.label,
    no_interf,
    provided_chroms=no_drive_chroms,
)
run_sims(queue)

print(no_drive_chroms.chroms[0].a1.get_crossover_locations())
chrom1_strand = Segments.from_strand(no_drive_chroms.chroms[0].a1)
print(chrom1_strand)
sered = chrom1_strand.to_str()
print(sered)
desered = Segments.from_str(sered)
print(desered)
