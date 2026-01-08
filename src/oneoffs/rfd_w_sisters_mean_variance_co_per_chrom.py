import argparse
from datetime import datetime
from pathlib import Path
from typing import cast

import numpy as np

from src.specific_interf_sims.data_transformers import (
    transform_mean_cos_per_chrom_data,
    transform_variance_cos_per_chrom_data,
)
from src.specific_interf_sims.interf_schemes import (
    equalised_fixed_regional_interf,
    equalised_global_interf,
    local_interf,
    no_interf,
)
from src.specific_interf_sims.models import induce_COs_via_random_forward_draw
from src.specific_interf_sims.simulator import SimulationQueue, queue_sim, run_sims
from src.utils.chromosome_set import make_default_chromset

parser = argparse.ArgumentParser()
_ = parser.add_argument(
    '--alt-seed',
    help='use 93719 instead of 54321 as random seed',
    action='store_true',
)
args = parser.parse_args()

NUM_CHROMS = 100000
CHROM_LEN = 1000
AVG_COS_PER_CHROMOSOME = 7.5
STARTING_P_HIT = AVG_COS_PER_CHROMOSOME / CHROM_LEN

queue = SimulationQueue(
    induce_COs_via_random_forward_draw,
    np.zeros(CHROM_LEN) + STARTING_P_HIT,
    rand_seed=54321 if not cast(bool, args.alt_seed) else 93719,
    chroms_per_sim=NUM_CHROMS,
    default_chromset_maker=make_default_chromset,
)
no_interf_chroms_avg_2 = queue_sim(
    queue,
    'no interf (avg=2)',
    no_interf,
    starting_bins=np.zeros(CHROM_LEN) + (2 / CHROM_LEN),
)
no_interf_chroms = queue_sim(queue, 'no interf', no_interf)
local_chroms = queue_sim(queue, 'local', local_interf)
equalised_global_chroms = queue_sim(queue, 'equalised global', equalised_global_interf)
equalised_fixed_regional_chroms = queue_sim(
    queue, 'equalised fixed regional', equalised_fixed_regional_interf
)

run_sims(queue)

chrom_sets = [
    no_interf_chroms,
    local_chroms,
    equalised_global_chroms,
    equalised_fixed_regional_chroms,
    no_interf_chroms_avg_2,
]

chrom_set_means = map(transform_mean_cos_per_chrom_data, chrom_sets)
chrom_set_vars = map(transform_variance_cos_per_chrom_data, chrom_sets)
output = ''
output += 'INTERF SCHEME\tMEAN(COs per chrom)\tVAR(COS per chrom)\n\n'
for mean_output, var_output in zip(chrom_set_means, chrom_set_vars):
    output += f'{mean_output[0]}\t{str(mean_output[1])}\t{str(var_output[1])}\n'

curr_time = datetime.now().strftime('%Y-%m-%d__%H-%M-%S')
num_chrom_in_ks = str(NUM_CHROMS / 1000).replace('.', '_')
num_bins_in_ks = str(CHROM_LEN / 1000).replace('.', '_')
outputfile = Path(
    f'out--{curr_time}--mean_variance_co_per_chrom--{num_chrom_in_ks}k--{num_bins_in_ks}k.txt'
)
_ = outputfile.write_text(output)
print(f'output written to {str(outputfile.absolute())}')
