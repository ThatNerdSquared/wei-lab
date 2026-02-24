import matplotlib.pyplot as plt
import numpy as np
from croissant.chromosomes import make_chromset_with_legacy_chromosome

from src.specific_interf_sims.data_transformers import transform_cM_rf_curve_data
from src.specific_interf_sims.interf_schemes import (
    equalised_fixed_regional_interf,
    equalised_global_interf,
    fixed_regional_interf,
    global_interf,
    local_interf,
    no_interf,
)
from src.specific_interf_sims.models import induce_COs_via_random_forward_draw
from src.specific_interf_sims.plotter import (
    fit_haldane_p_to_data,
    plot_canonical_map_funcs,
    save_plot,
)
from src.specific_interf_sims.simulator import SimulationQueue, queue_sim, run_sims

NUM_CHROMS = 100000
CHROM_LENGTH = 1000
AVG_COs_PER_CHROMOSOME = 1
STARTING_P_HIT = AVG_COs_PER_CHROMOSOME / CHROM_LENGTH

starting_bins = np.zeros(CHROM_LENGTH) + STARTING_P_HIT
queue = SimulationQueue(
    induce_COs_via_random_forward_draw,
    default_starting_bins=starting_bins,
    rand_seed=54321,
    chroms_per_sim=NUM_CHROMS,
    default_chromset_maker=make_chromset_with_legacy_chromosome,
)

no_interf_chroms = queue_sim(queue, 'no interf', no_interf)
local_chroms = queue_sim(queue, 'local', local_interf)
global_chroms = queue_sim(queue, 'global', global_interf)
fixed_regional_chroms = queue_sim(queue, 'fixed regional', fixed_regional_interf)
equalised_global_chroms = queue_sim(queue, 'equalised global', equalised_global_interf)
equalised_fixed_regional_chroms = queue_sim(
    queue, 'equalised fixed regional', equalised_fixed_regional_interf
)

run_sims(queue)

for chrom_set in [
    no_interf_chroms,
    local_chroms,
    global_chroms,
    fixed_regional_chroms,
    equalised_global_chroms,
    equalised_fixed_regional_chroms,
]:
    label, data = transform_cM_rf_curve_data(chrom_set)
    actual_data_cM, actual_data_rf = data
    p, fitted_rf = fit_haldane_p_to_data(actual_data_cM, actual_data_rf)

    fig = plt.figure(figsize=(9.0, 7.0))
    ax = fig.add_subplot()
    _ = ax.set_xlabel('morgans (M)')
    _ = ax.set_ylabel('recombinant frequency (rf)')

    _ = ax.scatter(actual_data_cM, actual_data_rf, label=chrom_set.label)
    _ = ax.plot(actual_data_cM, fitted_rf, label=f'haldane, p = {p}', color='tab:pink')
    plot_canonical_map_funcs(ax)

    _ = ax.legend()
    _ = ax.set_xlim(left=0.0, right=1.0)
    _ = ax.set_ylim(bottom=0, top=0.75)

    fmt_name = chrom_set.label.replace(' ', '-')
    num_chrom_in_ks = str(NUM_CHROMS / 1000)
    num_bins_in_ks = str(CHROM_LENGTH / 1000)
    _ = fig.suptitle(
        f'cM_rf_curve_fitted_{fmt_name} -- {num_chrom_in_ks}k chroms @ {num_bins_in_ks}k bins / chrom',  # noqa: E501
    )
    save_plot(fig)
