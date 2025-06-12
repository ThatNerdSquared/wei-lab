import numpy as np
import matplotlib.pyplot as plt

from src.randomised_forward_draw.rfd_chromosome import BinArray, induce_COs_via_random_forward_draw
from src.utils.chromosome_set import ChromosomeSet, gather_co_per_chrom_dist_data, gather_interevent_dist_data, gather_recomb_curve_data, gather_rf_cM_curve_data
from src.utils.cli_utils import calculate_max_y, elapsed_time_printer, plot_name_formatter, quickplot
from src.utils.standard_funcs import haldane_input_cM

NUM_CHROMOSOMES = 100000
BINS_PER_CHROMOSOME = 1000
AVG_COS_PER_CHROMOSOME = 1
STARTING_P_HIT = AVG_COS_PER_CHROMOSOME / BINS_PER_CHROMOSOME
RAND_SEED = 54321

def no_interf(_: int, bins: BinArray):
    return bins

def local_interf(co_loc: int, bins: BinArray):
    bin_locs = np.arange(len(bins))
    return bins * (1 - np.exp(-1 * np.abs(
        (bin_locs / BINS_PER_CHROMOSOME) - (co_loc / BINS_PER_CHROMOSOME)
    )))

def global_interf(co_loc: int, bins: BinArray):
    return bins * 0.7

# def regional_interf(co_loc: int, bins: BinArray):
#     bin_locs = np.arange(len(bins))
#     bin_
#     return bins * 0.3

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

print('simulating crossovers...')

with elapsed_time_printer(
    op_name='no interf',
    op_number=(1, 3),
    chrom_set=no_interf_chroms
):
    induce_COs_via_random_forward_draw(
        no_interf_chroms,
        lambda x: np.zeros(len(x)) + STARTING_P_HIT,
        no_interf
    )

with elapsed_time_printer(
    op_name='local interf',
    op_number=(2, 3),
    chrom_set=local_interf_chroms
):
    induce_COs_via_random_forward_draw(
        local_interf_chroms,
        lambda x: np.zeros(len(x), dtype=np.float64) + STARTING_P_HIT,
        local_interf
    )

with elapsed_time_printer(
    op_name='global interf',
    op_number=(3, 3),
    chrom_set=global_interf_chroms
):
    induce_COs_via_random_forward_draw(
        global_interf_chroms,
        lambda x: np.zeros(len(x), dtype=np.float64) + STARTING_P_HIT,
        global_interf
    )


no_interf_recomb_curve = gather_recomb_curve_data(no_interf_chroms)
local_interf_recomb_curve = gather_recomb_curve_data(local_interf_chroms)
global_interf_recomb_curve = gather_recomb_curve_data(global_interf_chroms)
with quickplot(
    x_label='bin',
    y_label='% of chroms with CO',
    y_lim=(
        no_interf_recomb_curve,
        local_interf_recomb_curve,
        global_interf_recomb_curve
    ),
    fig_name=plot_name_formatter('recomb_curve', NUM_CHROMOSOMES, BINS_PER_CHROMOSOME)
) as recomb_curve:
    recomb_curve.plot(no_interf_recomb_curve, label='no interf')
    recomb_curve.plot(local_interf_recomb_curve, label='local interf')
    recomb_curve.plot(global_interf_recomb_curve, label='global interf')


no_interf_co_per_chrom_dist = gather_co_per_chrom_dist_data(
    no_interf_chroms
)
local_interf_co_per_chrom_dist = gather_co_per_chrom_dist_data(
    local_interf_chroms
)
global_interf_co_per_chrom_dist = gather_co_per_chrom_dist_data(
    global_interf_chroms
)
with quickplot(
    x_label='# of COs',
    y_label='% of chroms',
    y_lim=(
        no_interf_co_per_chrom_dist,
        local_interf_co_per_chrom_dist,
        global_interf_co_per_chrom_dist
    ),
    fig_name=plot_name_formatter('co_per_chrom', NUM_CHROMOSOMES, BINS_PER_CHROMOSOME)
) as co_per_chrom_dist:
    co_per_chrom_dist.stairs(no_interf_co_per_chrom_dist, label='no interf')
    co_per_chrom_dist.stairs(local_interf_co_per_chrom_dist, label='local interf')
    co_per_chrom_dist.stairs(global_interf_co_per_chrom_dist, label='global interf')


no_interf_interevent_totals, no_interf_interevent_groups = gather_interevent_dist_data(
    no_interf_chroms
)
local_interf_interevent_totals, local_interf_interevent_groups = gather_interevent_dist_data(
    local_interf_chroms
)
global_interf_interevent_totals, global_interf_interevent_groups = gather_interevent_dist_data(
    global_interf_chroms
)
with quickplot(
    x_label='inter-CO distance',
    y_label='% of chroms',
    y_lim=(
        no_interf_interevent_totals,
        local_interf_interevent_totals,
        global_interf_interevent_totals
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


no_interf_cM, no_interf_rf = gather_rf_cM_curve_data(no_interf_chroms)
local_interf_cM, local_interf_rf = gather_rf_cM_curve_data(local_interf_chroms)
global_interf_cM, global_interf_rf = gather_rf_cM_curve_data(global_interf_chroms)
with quickplot(
    x_label='centimorgans (cM)',
    y_label='recombinant frequency (rf)',
    y_lim=1.0,
    fig_name=plot_name_formatter('rf_cM', NUM_CHROMOSOMES, BINS_PER_CHROMOSOME)
) as rf_cM:
    rf_cM.plot(no_interf_cM, no_interf_rf, label='no interf')
    rf_cM.plot(local_interf_cM, local_interf_rf, label='local interf')
    rf_cM.plot(global_interf_cM, global_interf_rf, label='global interf')
    x = np.linspace(0, 2, 1000)
    rf_cM.plot(x, [haldane_input_cM(n) for n in x], label='haldane')
    rf_cM.plot(x, x, label='morgan')
