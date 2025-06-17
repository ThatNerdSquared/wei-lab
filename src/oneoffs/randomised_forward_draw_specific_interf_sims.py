from typing import cast
import numpy as np
import matplotlib.pyplot as plt

from src.randomised_forward_draw.rfd_chromosome import induce_COs_via_random_forward_draw
from src.specific_interf_sims.simulator import BinArray, InterferenceSim, SimulationQueue, queue_sim, run_sims
from src.utils.chromosome_set import ChromosomeSet, gather_co_per_chrom_dist_data, gather_interevent_dist_data, gather_recomb_curve_data, gather_rf_cM_curve_data
from src.utils.cli_utils import calculate_max_y, plot_name_formatter, quickplot
from src.utils.standard_funcs import haldane_input_cM, kosambi

NUM_CHROMOSOMES = 100000
BINS_PER_CHROMOSOME = 1000
AVG_COS_PER_CHROMOSOME = 1
STARTING_P_HIT = np.float64(AVG_COS_PER_CHROMOSOME / BINS_PER_CHROMOSOME)
RAND_SEED = 54321

def no_interf(_: int, bins: BinArray) -> BinArray:
    return bins

def local_interf(co_loc: int, bins: BinArray) -> BinArray:
    bin_locs = np.arange(len(bins))
    return bins * (1 - np.exp(-1 * np.abs(
        (bin_locs / BINS_PER_CHROMOSOME) - (co_loc / BINS_PER_CHROMOSOME)
    )))

def global_interf(co_loc: int, bins: BinArray) -> BinArray:
    return bins * 0.7

def equalised_global_interf(co_loc: int, bins: BinArray) -> BinArray:
    new_total_p_co = np.sum(local_interf(co_loc, bins))
    current_total_p_co = np.sum(bins)
    return bins * new_total_p_co / current_total_p_co

def regional_interf(co_loc: int, bins: BinArray) -> BinArray:
    bin_locs = np.arange(len(bins))
    return bins * (
        np.where(bin_locs <= co_loc, 0, 1)
        if co_loc < len(bins) / 2 else
        np.where(bin_locs >= co_loc, 0, 1)
    )

print('setting up chromosomes...')
no_interf_chroms = ChromosomeSet(
    num_chroms=NUM_CHROMOSOMES,
    chrom_length=BINS_PER_CHROMOSOME,
    rand_seed=RAND_SEED
)

local_interf_chroms = ChromosomeSet(
    num_chroms=NUM_CHROMOSOMES,
    chrom_length=BINS_PER_CHROMOSOME,
    rand_seed=RAND_SEED
)

global_interf_chroms = ChromosomeSet(
    num_chroms=NUM_CHROMOSOMES,
    chrom_length=BINS_PER_CHROMOSOME,
    rand_seed=RAND_SEED
)

regional_interf_chroms = ChromosomeSet(
    num_chroms=NUM_CHROMOSOMES,
    chrom_length=BINS_PER_CHROMOSOME,
    rand_seed=RAND_SEED
)

default_bins = cast(
    BinArray,
    np.zeros(BINS_PER_CHROMOSOME, dtype=np.float64) + STARTING_P_HIT
)
queue = SimulationQueue(induce_COs_via_random_forward_draw, default_bins)
no_interf_sim = InterferenceSim(
    label='no interf',
    interf_strategy=no_interf,
    chrom_set=no_interf_chroms
)
queue_sim(queue, no_interf_sim)
local_interf_sim = InterferenceSim(
    label='local interf',
    interf_strategy=local_interf,
    chrom_set=local_interf_chroms
)
queue_sim(queue, local_interf_sim)
global_interf_sim = InterferenceSim(
    label='global interf',
    interf_strategy=global_interf,
    chrom_set=global_interf_chroms
)
queue_sim(queue, global_interf_sim)
regional_interf_sim = InterferenceSim(
    label='regional interf',
    interf_strategy=regional_interf,
    chrom_set=regional_interf_chroms
)
queue_sim(queue, regional_interf_sim)

run_sims(queue)

no_interf_recomb_curve = gather_recomb_curve_data(no_interf_chroms)
local_interf_recomb_curve = gather_recomb_curve_data(local_interf_chroms)
global_interf_recomb_curve = gather_recomb_curve_data(global_interf_chroms)
regional_interf_recomb_curve = gather_recomb_curve_data(regional_interf_chroms)
with quickplot(
    x_label='bin',
    y_label='% of chroms with CO',
    y_lim=(
        no_interf_recomb_curve,
        local_interf_recomb_curve,
        global_interf_recomb_curve,
        regional_interf_recomb_curve
    ),
    fig_name=plot_name_formatter('recomb_curve', NUM_CHROMOSOMES, BINS_PER_CHROMOSOME)
) as recomb_curve:
    recomb_curve.plot(no_interf_recomb_curve, label='no interf')
    recomb_curve.plot(local_interf_recomb_curve, label='local interf')
    recomb_curve.plot(global_interf_recomb_curve, label='global interf')
    recomb_curve.plot(regional_interf_recomb_curve, label='regional interf')


no_interf_co_per_chrom_dist = gather_co_per_chrom_dist_data(
    no_interf_chroms
)
local_interf_co_per_chrom_dist = gather_co_per_chrom_dist_data(
    local_interf_chroms
)
global_interf_co_per_chrom_dist = gather_co_per_chrom_dist_data(
    global_interf_chroms
)
regional_interf_co_per_chrom_dist = gather_co_per_chrom_dist_data(
    regional_interf_chroms
)
with quickplot(
    x_label='# of COs',
    y_label='% of chroms',
    y_lim=(
        no_interf_co_per_chrom_dist,
        local_interf_co_per_chrom_dist,
        global_interf_co_per_chrom_dist,
        regional_interf_co_per_chrom_dist
    ),
    fig_name=plot_name_formatter('co_per_chrom', NUM_CHROMOSOMES, BINS_PER_CHROMOSOME)
) as co_per_chrom_dist:
    co_per_chrom_dist.stairs(no_interf_co_per_chrom_dist, label='no interf')
    co_per_chrom_dist.stairs(local_interf_co_per_chrom_dist, label='local interf')
    co_per_chrom_dist.stairs(global_interf_co_per_chrom_dist, label='global interf')
    co_per_chrom_dist.stairs(regional_interf_co_per_chrom_dist, label='regional interf')


no_interf_interevent_totals, no_interf_interevent_groups = gather_interevent_dist_data(
    no_interf_chroms
)
local_interf_interevent_totals, local_interf_interevent_groups = gather_interevent_dist_data(
    local_interf_chroms
)
global_interf_interevent_totals, global_interf_interevent_groups = gather_interevent_dist_data(
    global_interf_chroms
)
regional_interf_interevent_totals, regional_interf_interevent_groups = gather_interevent_dist_data(
    regional_interf_chroms
)
with quickplot(
    x_label='inter-CO distance',
    y_label='% of chroms',
    y_lim=(
        no_interf_interevent_totals,
        local_interf_interevent_totals,
        global_interf_interevent_totals,
        regional_interf_interevent_totals
    ),
    fig_name=plot_name_formatter('interevent_dist', NUM_CHROMOSOMES, BINS_PER_CHROMOSOME)
) as interevent_dist:
    interevent_dist.stairs(
        no_interf_interevent_totals,
        no_interf_interevent_groups,
        label='no interf'
    )
    interevent_dist.stairs(
        local_interf_interevent_totals,
        local_interf_interevent_groups,
        label='local interf'
    )
    interevent_dist.stairs(
        global_interf_interevent_totals,
        global_interf_interevent_groups,
        label='global interf'
    )
    interevent_dist.stairs(
        regional_interf_interevent_totals,
        regional_interf_interevent_groups,
        label='regional interf'
    )


no_interf_cM, no_interf_rf = gather_rf_cM_curve_data(no_interf_chroms)
local_interf_cM, local_interf_rf = gather_rf_cM_curve_data(local_interf_chroms)
global_interf_cM, global_interf_rf = gather_rf_cM_curve_data(global_interf_chroms)
regional_interf_cM, regional_interf_rf = gather_rf_cM_curve_data(regional_interf_chroms)
with quickplot(
    x_label='centimorgans (cM)',
    y_label='recombinant frequency (rf)',
    y_lim=1.0,
    fig_name=plot_name_formatter('rf_cM', NUM_CHROMOSOMES, BINS_PER_CHROMOSOME)
) as rf_cM:
    rf_cM.plot(no_interf_cM, no_interf_rf, label='no interf')
    rf_cM.plot(local_interf_cM, local_interf_rf, label='local interf')
    rf_cM.plot(global_interf_cM, global_interf_rf, label='global interf')
    rf_cM.plot(regional_interf_cM, regional_interf_rf, label='regional interf')
    plt.axhline(
        y=0.5,
        color='tab:red',
        label=f'50% RF'
    ).set_dashes((5, 2))
    x = np.linspace(0, 2, 1000)
    rf_cM.plot(x, [haldane_input_cM(n) for n in x], label='haldane')
    rf_cM.plot(x, x, label='morgan')
    rf_cM.plot(x, [kosambi(n) for n in x], label='kosambi')
