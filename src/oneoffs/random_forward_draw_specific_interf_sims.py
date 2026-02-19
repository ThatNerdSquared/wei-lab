import numpy as np

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
    global_interf,
    local_interf,
    no_interf,
    telomere_based_regional_interf,
)
from src.specific_interf_sims.models import induce_COs_via_random_forward_draw
from src.specific_interf_sims.plotter import (
    plot_canonical_map_funcs,
    save_plot,
    standard_plotter,
)
from src.specific_interf_sims.simulator import SimulationQueue, queue_sim, run_sims
from src.utils.chromosome_set import make_chromset_with_legacy_chromosome

NUM_CHROMOSOMES = 100000
BINS_PER_CHROMOSOME = 1000
AVG_COS_PER_CHROMOSOME = 1
STARTING_P_HIT = np.float64(AVG_COS_PER_CHROMOSOME / BINS_PER_CHROMOSOME)
RAND_SEED = 54321

default_bins = np.zeros(BINS_PER_CHROMOSOME, dtype=np.float64) + STARTING_P_HIT
queue = SimulationQueue(
    induce_COs_via_random_forward_draw,
    default_bins,
    rand_seed=RAND_SEED,
    chroms_per_sim=NUM_CHROMOSOMES,
    default_chromset_maker=make_chromset_with_legacy_chromosome,
)
print('setting up chromosomes...')
no_interf_chroms = queue_sim(queue, 'no interf', no_interf)
local_interf_chroms = queue_sim(queue, 'local interf', local_interf)
global_interf_chroms = queue_sim(queue, 'global interf', global_interf)
regional_interf_chroms = queue_sim(
    queue, 'regional interf', telomere_based_regional_interf
)

run_sims(queue)

chrom_sets = [
    no_interf_chroms,
    local_interf_chroms,
    global_interf_chroms,
    regional_interf_chroms,
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
