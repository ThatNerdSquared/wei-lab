import numpy as np

from src.specific_interf_sims.data_collectors import (
    collect_sliding_compareregion_coc_curve_data,
)
from src.specific_interf_sims.data_transformers import (
    Region,
    transform_mean_cos_per_chrom_data,
    transform_sliding_compare_region_coc_curve_data,
)
from src.specific_interf_sims.interf_schemes import (
    equalised_fixed_regional_interf,
    equalised_global_interf,
    local_interf,
    no_interf,
)
from src.specific_interf_sims.models import induce_COs_via_random_forward_draw
from src.specific_interf_sims.plotter import save_plot, standard_plotter
from src.specific_interf_sims.simulator import SimulationQueue, queue_sim, run_sims
from src.utils.chromosome_set import make_default_chromset
from src.utils.cli_utils import mapThenPrint

NUM_CHROMS = 100000
CHROM_LEN = 1000
AVG_COS_PER_CHROMOSOME = 2
STARTING_P_HIT = AVG_COS_PER_CHROMOSOME / CHROM_LEN

queue = SimulationQueue(
    induce_COs_via_random_forward_draw,
    np.zeros(CHROM_LEN) + STARTING_P_HIT,
    # rand_seed=54321,
    rand_seed=93719,
    chroms_per_sim=NUM_CHROMS,
    default_chromset_maker=make_default_chromset,
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
]

mapThenPrint(transform_mean_cos_per_chrom_data, chrom_sets)

coc_curve_data = collect_sliding_compareregion_coc_curve_data(
    lambda x: transform_sliding_compare_region_coc_curve_data(
        x, ref_region=Region(0, 75), sliding_region_size=75
    ),
    chrom_sets,
)
coc_plot = standard_plotter(coc_curve_data)
save_plot(coc_plot)
