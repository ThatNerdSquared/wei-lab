import numpy as np

from src.specific_interf_sims.data_collectors import collect_preinterf_cM_rf_curve_data
from src.specific_interf_sims.data_transformers import (
    transform_preinterf_cM_rf_curve_data,
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
from src.utils.chromosome_set import make_chromset_with_legacy_chromosome

CHROM_LEN = 1000
AVG_COS_PER_CHROMOSOME = 1
STARTING_P_HIT = AVG_COS_PER_CHROMOSOME / CHROM_LEN

starting_bins = np.zeros(CHROM_LEN) + STARTING_P_HIT
queue = SimulationQueue(
    induce_COs_via_random_forward_draw,
    default_starting_bins=starting_bins,
    rand_seed=54321,
    chroms_per_sim=100000,
    default_chromset_maker=make_chromset_with_legacy_chromosome,
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
preinterf_cM_rf_curve_data = collect_preinterf_cM_rf_curve_data(
    lambda x: transform_preinterf_cM_rf_curve_data(x, starting_bins), chrom_sets
)
preinterf_cM_rf_curve_plot = standard_plotter(preinterf_cM_rf_curve_data)
preinterf_cM_rf_ax = preinterf_cM_rf_curve_plot.get_axes()[0]
plot_canonical_map_funcs(preinterf_cM_rf_ax)
_ = preinterf_cM_rf_ax.set_xlim(right=1.5)
save_plot(preinterf_cM_rf_curve_plot)
