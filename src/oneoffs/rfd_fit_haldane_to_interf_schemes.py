import numpy as np

from src.randomised_forward_draw.rfd_chromosome import induce_COs_via_random_forward_draw
from src.specific_interf_sims.interf_schemes import equalised_fixed_regional_interf, equalised_global_interf, fixed_regional_interf, global_interf, local_interf, no_interf
from src.specific_interf_sims.plotter import fit_haldane_p_to_data, plot_canonical_map_funcs
from src.specific_interf_sims.simulator import SimulationQueue, queue_sim, run_sims
from src.utils.chromosome_set import ChromosomeSet, gather_cM_rf_curve_data
from src.utils.cli_utils import plot_name_formatter, quickplot

NUM_CHROMS = 100000
CHROM_LENGTH = 1000
AVG_COs_PER_CHROMOSOME = 1
STARTING_P_HIT = AVG_COs_PER_CHROMOSOME / CHROM_LENGTH

starting_bins = np.zeros(CHROM_LENGTH) + STARTING_P_HIT
queue = SimulationQueue(
    induce_COs_via_random_forward_draw,
    default_starting_bins=starting_bins,
    rand_seed=54321,
    chroms_per_sim=NUM_CHROMS
)

no_interf_chroms = queue_sim(queue, 'no interf', no_interf)
local_chroms = queue_sim(queue, 'local', local_interf)
global_chroms = queue_sim(queue, 'global', global_interf)
fixed_regional_chroms = queue_sim(queue, 'fixed regional', fixed_regional_interf)
equalised_global_chroms = queue_sim(queue, 'equalised global', equalised_global_interf)
equalised_fixed_regional_chroms = queue_sim(queue, 'equalised fixed regional', equalised_fixed_regional_interf)

run_sims(queue)

for chrom_set in [
    no_interf_chroms,
    local_chroms,
    global_chroms,
    fixed_regional_chroms,
    equalised_global_chroms,
    equalised_fixed_regional_chroms,
]:
    actual_data_cM, actual_data_rf = gather_cM_rf_curve_data(chrom_set)
    p, fitted_rf = fit_haldane_p_to_data(actual_data_cM, actual_data_rf)

    fmt_name = chrom_set.label.replace(' ', '-')
    with quickplot(
        fig_name=plot_name_formatter(
            f'cM_rf_curve_fitted_{fmt_name}',
            chrom_set.num_chroms,
            chrom_set.chrom_length
        ),
        x_label='morgans (M)',
        y_label='recombinant frequency (rf)',
        y_lim=0.75,
    ) as ax:
        _ = ax.scatter(actual_data_cM, actual_data_rf, label=chrom_set.label)
        _ = ax.plot(actual_data_cM, fitted_rf, label=f'haldane, p = {p}', color='tab:pink')
        plot_canonical_map_funcs(ax)
        _ = ax.set_xlim(left=0.0, right=1.0)
