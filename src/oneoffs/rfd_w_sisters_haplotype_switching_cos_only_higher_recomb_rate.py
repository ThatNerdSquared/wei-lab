import numpy as np
from croissant.chromosomes import make_chromset_haplotype_switching_only

from src.specific_interf_sims.data_collectors import (
    collect_cM_rf_curve_data,
)
from src.specific_interf_sims.data_transformers import (
    transform_cM_rf_curve_data,
    transform_co_haplotype_switch_data,
    transform_mean_cos_per_chrom_data,
)
from src.specific_interf_sims.interf_schemes import (
    equalised_fixed_regional_interf,
    equalised_global_interf,
    local_interf,
    no_interf,
)
from src.specific_interf_sims.models import induce_COs_via_random_forward_draw
from src.specific_interf_sims.plotter import (
    plot_canonical_map_funcs,
    save_plot,
    standard_plotter,
)
from src.specific_interf_sims.simulator import SimulationQueue, queue_sim, run_sims
from src.utils.cli_utils import mapThenPrint

NUM_CHROMS = 100000
CHROM_LEN = 1000
AVG_COS_PER_CHROMOSOME = 8
STARTING_P_HIT = AVG_COS_PER_CHROMOSOME / CHROM_LEN

queue = SimulationQueue(
    induce_COs_via_random_forward_draw,
    np.zeros(CHROM_LEN) + STARTING_P_HIT,
    rand_seed=54321,
    chroms_per_sim=NUM_CHROMS,
    default_chromset_maker=make_chromset_haplotype_switching_only,
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

cM_rf_curve_data = collect_cM_rf_curve_data(transform_cM_rf_curve_data, chrom_sets)
cM_rf_curve_plot = standard_plotter(cM_rf_curve_data)
cM_rf_ax = cM_rf_curve_plot.get_axes()[0]
plot_canonical_map_funcs(cM_rf_ax)
save_plot(cM_rf_curve_plot)

mapThenPrint(transform_mean_cos_per_chrom_data, chrom_sets)
mapThenPrint(transform_co_haplotype_switch_data, chrom_sets)
