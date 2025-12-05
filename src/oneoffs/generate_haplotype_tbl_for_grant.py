import numpy as np

from src.specific_interf_sims.interf_schemes import local_interf
from src.specific_interf_sims.models import induce_COs_via_random_forward_draw
from src.specific_interf_sims.simulator import SimulationQueue, queue_sim, run_sims
from src.utils.chromatid import Chromatid, convert_chromatid_to_haplotype_tbl
from src.utils.chromosome_set import (
    make_chromset_haplotype_switching_only,
    pool_iter,
    pool_size,
)

NUM_CHROMS = 100000
# CHROM_LEN = 100
CHROM_LEN = 10
# AVG_COS_PER_CHROMOSOME = 1
AVG_COS_PER_CHROMOSOME = 7.5
STARTING_P_HIT = AVG_COS_PER_CHROMOSOME / CHROM_LEN

queue = SimulationQueue(
    induce_COs_via_random_forward_draw,
    np.zeros(CHROM_LEN) + STARTING_P_HIT,
    rand_seed=54321,
    chroms_per_sim=NUM_CHROMS,
    default_chromset_maker=make_chromset_haplotype_switching_only,
)
chroms = queue_sim(queue, 'local', local_interf)

run_sims(queue)


NUM_CHROMS_IN_POOL = pool_size(chroms)
haplotype_tbl = np.zeros((NUM_CHROMS_IN_POOL, chroms.chrom_length), dtype=np.int64)
for idx, chrom in enumerate(pool_iter(chroms)):
    assert isinstance(chrom, Chromatid)
    convert_chromatid_to_haplotype_tbl(chrom, haplotype_tbl[idx])

np.set_printoptions(linewidth=1000)
np.savetxt('output.txt', haplotype_tbl, fmt='%0d')
print(haplotype_tbl)
