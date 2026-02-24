import numpy as np
from croissant.chromosomes import make_chromset_with_legacy_chromosome

from src.specific_interf_sims.data_collectors import (
    collect_cM_rf_curve_data,
    collect_co_per_chrom_dist_data,
    collect_interevent_dist_data,
    collect_recomb_curve_data,
)
from src.specific_interf_sims.data_transformers import (
    transform_cM_rf_curve_data_haplotype_switching_only,
    transform_co_per_chrom_dist_data,
    transform_interevent_dist_data,
    transform_recomb_curve_data,
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

NUM_CHROMS = 100000
CHROM_LEN = 1000
AVG_COS_PER_CHROMOSOME = 1
STARTING_P_HIT = AVG_COS_PER_CHROMOSOME / CHROM_LEN

queue = SimulationQueue(
    induce_COs_via_random_forward_draw,
    np.zeros(CHROM_LEN) + STARTING_P_HIT,
    rand_seed=54321,
    chroms_per_sim=NUM_CHROMS,
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

recomb_curve_data = collect_recomb_curve_data(transform_recomb_curve_data, chrom_sets)
recomb_curve_plot = standard_plotter(recomb_curve_data)
save_plot(recomb_curve_plot)

co_per_chrom_dist_data = collect_co_per_chrom_dist_data(
    transform_co_per_chrom_dist_data, chrom_sets
)
co_per_chrom_dist_plot = standard_plotter(co_per_chrom_dist_data)
save_plot(co_per_chrom_dist_plot)

interevent_dist_data = collect_interevent_dist_data(
    transform_interevent_dist_data, chrom_sets
)
interevent_dist_plot = standard_plotter(interevent_dist_data)
save_plot(interevent_dist_plot)

cM_rf_curve_data = collect_cM_rf_curve_data(
    transform_cM_rf_curve_data_haplotype_switching_only, chrom_sets
)
cM_rf_curve_plot = standard_plotter(cM_rf_curve_data)
cM_rf_ax = cM_rf_curve_plot.get_axes()[0]
plot_canonical_map_funcs(cM_rf_ax)
_ = cM_rf_ax.set_xlim(right=1.5)
save_plot(cM_rf_curve_plot)
