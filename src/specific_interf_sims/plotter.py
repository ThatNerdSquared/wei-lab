import numpy as np
import matplotlib.pyplot as plt

from src.utils.chromosome_set import ChromosomeSet, gather_cM_rf_curve_data, gather_co_per_chrom_dist_data, gather_interevent_dist_data, gather_recomb_curve_data
from src.utils.cli_utils import plot_name_formatter, quickplot
from src.utils.standard_funcs import haldane_input_cM, kosambi


def plot_recomb_curve(*args: ChromosomeSet):
    recomb_curves = [gather_recomb_curve_data(x) for x in args]
    with quickplot(
        x_label='bin',
        y_label='% of chroms with CO',
        y_lim=recomb_curves,
        fig_name=plot_name_formatter('recomb_curve', args[0].num_chroms, args[0].chrom_length)
    ) as recomb_curve:
        for curve_idx in range(len(recomb_curves)):
            _ = recomb_curve.plot(recomb_curves[curve_idx], label=args[curve_idx].label)

def plot_co_per_chrom_dist(*args: ChromosomeSet):
    co_per_chrom_dists = [gather_co_per_chrom_dist_data(x) for x in args]
    with quickplot(
        x_label='# of COs',
        y_label='% of chroms',
        y_lim=co_per_chrom_dists,
        fig_name=plot_name_formatter('co_per_chrom', args[0].num_chroms, args[0].chrom_length)
    ) as co_per_chrom_dist:
        for i in range(len(co_per_chrom_dists)):
            _ = co_per_chrom_dist.stairs(co_per_chrom_dists[i], label=args[i].label)

def plot_interevent_dist(*args: ChromosomeSet):
    interevent_dists = [gather_interevent_dist_data(x) for x in args]
    with quickplot(
        x_label='inter-CO distance',
        y_label='% of chroms',
        y_lim=[n[0] for n in interevent_dists],
        fig_name=plot_name_formatter('interevent_dist', args[0].num_chroms, args[0].chrom_length)
    ) as interevent_dist:
        for i in range(len(interevent_dists)):
            _ = interevent_dist.stairs(interevent_dists[i][0], interevent_dists[i][1], label=args[i].label)

def plot_cM_rf_curve(*args: ChromosomeSet):
    cM_rf_curves = [gather_cM_rf_curve_data(x) for x in args]
    with quickplot(
        x_label='morgans (M)',
        y_label='recombinant frequency (rf)',
        y_lim=1.0,
        fig_name=plot_name_formatter('cM_rf_curve', args[0].num_chroms, args[0].chrom_length)
    ) as morgan_rf_curves:
        for i in range(len(cM_rf_curves)):
            _ = morgan_rf_curves.plot(cM_rf_curves[i][0], cM_rf_curves[i][1], label=args[i].label)
        plt.axhline(
            y=0.5,
            color='tab:red',
            label=f'50% RF'
        ).set_dashes((5, 2))
        x = np.linspace(0, 2, 1000)
        _ = morgan_rf_curves.plot(x, [haldane_input_cM(n) for n in x], label='haldane')
        _ = morgan_rf_curves.plot(x, x, label='morgan')
        _ = morgan_rf_curves.plot(x, [kosambi(n) for n in x], label='kosambi')
