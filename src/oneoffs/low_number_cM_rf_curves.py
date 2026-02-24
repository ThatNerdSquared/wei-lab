import numpy as np
from croissant.chromosomes import make_chromset_with_legacy_chromosome

from src.specific_interf_sims.data_collectors import collect_cM_rf_curve_data
from src.specific_interf_sims.data_transformers import (
    transform_cM_rf_curve_data_haplotype_switching_only,
)
from src.specific_interf_sims.interf_schemes import (
    fixed_regional_interf,
    global_interf,
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

CHROM_LEN = 1000
AVG_COS_PER_CHROMOSOME = 1
STARTING_P_HIT = AVG_COS_PER_CHROMOSOME / CHROM_LEN

queue = SimulationQueue(
    induce_COs_via_random_forward_draw,
    np.zeros(CHROM_LEN) + STARTING_P_HIT,
    rand_seed=54321,
    default_chromset_maker=make_chromset_with_legacy_chromosome,
)

hundred_chrom_sets = [
    queue_sim(queue, 'no interf (100)', no_interf, chrom_count=100),
    queue_sim(queue, 'local interf (100)', local_interf, chrom_count=100),
    queue_sim(queue, 'global interf (100)', global_interf, chrom_count=100),
    queue_sim(
        queue, 'fixed regional interf (100)', fixed_regional_interf, chrom_count=100
    ),
]

thousand_chrom_sets = [
    queue_sim(queue, 'no interf (1000)', no_interf, chrom_count=1000),
    queue_sim(queue, 'local interf (1000)', local_interf, chrom_count=1000),
    queue_sim(queue, 'global interf (1000)', global_interf, chrom_count=1000),
    queue_sim(
        queue, 'fixed regional interf (1000)', fixed_regional_interf, chrom_count=1000
    ),
]

ten_thousand_chrom_sets = [
    queue_sim(queue, 'no interf (10000)', no_interf, chrom_count=10000),
    queue_sim(queue, 'local interf (10000)', local_interf, chrom_count=10000),
    queue_sim(queue, 'global interf (10000)', global_interf, chrom_count=10000),
    queue_sim(
        queue, 'fixed regional interf (10000)', fixed_regional_interf, chrom_count=10000
    ),
]

hundred_thousand_chrom_sets = [
    queue_sim(queue, 'no interf (100000)', no_interf, chrom_count=100000),
    queue_sim(queue, 'local interf (100000)', local_interf, chrom_count=100000),
    queue_sim(queue, 'global interf (100000)', global_interf, chrom_count=100000),
    queue_sim(
        queue,
        'fixed regional interf (100000)',
        fixed_regional_interf,
        chrom_count=100000,
    ),
]

run_sims(queue)

for chrom_sets in [
    hundred_chrom_sets,
    thousand_chrom_sets,
    ten_thousand_chrom_sets,
    hundred_thousand_chrom_sets,
]:
    cM_rf_curve_data = collect_cM_rf_curve_data(
        transform_cM_rf_curve_data_haplotype_switching_only, chrom_sets
    )
    cM_rf_curve_plot = standard_plotter(cM_rf_curve_data)
    cM_rf_ax = cM_rf_curve_plot.get_axes()[0]
    plot_canonical_map_funcs(cM_rf_ax)
    _ = cM_rf_ax.set_xlim(right=1.5)
    save_plot(cM_rf_curve_plot)
