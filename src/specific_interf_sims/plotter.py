from datetime import datetime
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy.optimize import curve_fit

from src.specific_interf_sims.data_packet import DataPacket
from src.specific_interf_sims.interf_schemes import BinArray
from src.utils.chromosome_set import (
    ChromosomeLike,
    ChromosomeSet,
    gather_cM_rf_curve_data,
    gather_co_per_chrom_dist_data,
    gather_interevent_dist_data,
    gather_preinterf_cM_rf_curve_data,
    gather_recomb_curve_data,
)
from src.utils.cli_utils import plot_name_formatter, quickplot
from src.utils.standard_funcs import haldane_input_cM, kosambi

type Gatherer[T, C: ChromosomeLike] = Callable[[ChromosomeSet[C]], T]


def plotter[T, C: ChromosomeLike](
    data_gatherer: Gatherer[T, C],
    plot_callback: Callable[[Axes, T, ChromosomeSet[C]], Any],
    chrom_sets: list[ChromosomeSet[C],] | tuple[ChromosomeSet[C], ...],
    y_lim: float | None = None,
    y_lim_func: Callable[[list[T]], list[Any]] | None = None,
    fig_name: str | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    post_plot: Callable[[Axes], Any] | None = None,
):
    print(f'processing data: {fig_name}')
    gathered_data = [data_gatherer(x) for x in chrom_sets]
    with quickplot(
        x_label=x_label or '',
        y_label=y_label or '',
        y_lim=(y_lim or gathered_data) if not y_lim_func else y_lim_func(gathered_data),
        fig_name=plot_name_formatter(
            fig_name or '', chrom_sets[0].num_chroms, chrom_sets[0].chrom_length
        ),
    ) as plot:
        for data, chrom_set in zip(gathered_data, chrom_sets):
            plot_callback(plot, data, chrom_set)
        if post_plot:
            post_plot(plot)


def _fmt_plot_name[T](data_packet: DataPacket[T]):
    num_chrom_in_ks = str(data_packet.num_chroms / 1000)
    num_bins_in_ks = str(data_packet.num_bins / 1000)
    return f'{data_packet.fig_name} -- {num_chrom_in_ks}k chroms @ {num_bins_in_ks}k bins / chrom'  # noqa: E501


def _fmt_plot_filename(plot_title: str):
    curr_time = datetime.now().strftime('%Y-%m-%d__%H-%M-%S')
    plot_title_fileready = (
        plot_title.replace(' bins / chrom', '')
        .replace('chroms @', '--')
        .replace(' -- ', '--')
        .replace('.', '_')
        .replace(' ', '_')
        .strip()
        .lower()
    )
    return f'out--{curr_time}--{plot_title_fileready}.png'


def standard_plotter[T](data_packet: DataPacket[T]) -> Figure:
    fig = plt.figure(figsize=(9.0, 7.0))
    ax = fig.add_subplot()
    data_packet.curve_adder(data_packet, ax)

    _ = ax.set_xlabel(data_packet.xlabel or '')
    _ = ax.set_ylabel(data_packet.ylabel or '')
    _ = ax.legend()
    _ = ax.set_ylim(bottom=0, top=data_packet.ylim)

    _ = fig.suptitle(_fmt_plot_name(data_packet))
    return fig


def save_plot(plot: Figure):
    plot_filename = _fmt_plot_filename(plot.get_suptitle())
    plot.savefig(plot_filename)
    print(f'plot "{plot.get_suptitle()}" saved.')


def line_curve_adder[T](data: DataPacket[npt.ArrayLike], ax: Axes):
    for k, v in data.data.items():
        _ = ax.plot(v, label=k)


def line_unpacked_data_curve_adder(
    data: DataPacket[tuple[list[float], list[float]]], ax: Axes
):
    for k, v in data.data.items():
        _ = ax.plot(v[0], v[1], label=k)


def histogram_curve_adder(data: DataPacket[npt.ArrayLike], ax: Axes):
    for k, v in data.data.items():
        _ = ax.stairs(v, label=k)


def histogram_unpacked_data_curve_adder(
    data: DataPacket[tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]],
    ax: Axes,
):
    for k, v in data.data.items():
        _ = (ax.stairs(v[0], v[1], label=k),)


def plot_recomb_curve[C: ChromosomeLike](*args: ChromosomeSet[C]):
    plotter(
        gather_recomb_curve_data,
        lambda ax, data, chroms: ax.plot(data, label=chroms.label),
        args,
        fig_name='recomb_curve',
        x_label='bin',
        y_label='% of chroms with CO',
    )


def plot_co_per_chrom_dist[C: ChromosomeLike](*args: ChromosomeSet[C]):
    plotter(
        gather_co_per_chrom_dist_data,
        lambda ax, data, chroms: ax.stairs(data, label=chroms.label),
        args,
        fig_name='co_per_chrom',
        x_label='# of COs',
        y_label='% of chroms',
    )


def plot_interevent_dist[C: ChromosomeLike](*args: ChromosomeSet[C]):
    plotter(
        gather_interevent_dist_data,
        lambda ax, data, chroms: ax.stairs(data[0], data[1], label=chroms.label),
        args,
        fig_name='interevent_dist',
        x_label='inter-CO distance',
        y_label='% of chroms',
        y_lim_func=lambda data: [x[0] for x in data],
    )


def plot_canonical_map_funcs(ax: Axes):
    ax.axhline(y=0.5, color='tab:red', label='50% RF').set_dashes((5, 2))
    x = np.linspace(0, 1, 1000)
    _ = ax.plot(x, haldane_input_cM(x), label='haldane')
    _ = ax.plot(x, x, label='morgan')
    _ = ax.plot(x, kosambi(x), label='kosambi')


def plot_cM_rf_curve[C: ChromosomeLike](*args: ChromosomeSet[C]):
    plotter(
        gather_cM_rf_curve_data,
        lambda ax, data, chroms: ax.plot(data[0], data[1], label=chroms.label),
        args,
        fig_name='cM_rf_curve',
        x_label='morgans (M)',
        y_label='recombinant frequency (rf)',
        y_lim=1.0,
        post_plot=plot_canonical_map_funcs,
    )


def plot_preinterf_cM_rf_curve[C: ChromosomeLike](
    starting_bins: BinArray, *args: ChromosomeSet[C]
):
    plotter(
        lambda x: gather_preinterf_cM_rf_curve_data(x, starting_bins),
        lambda ax, data, chroms: ax.plot(data[0], data[1], label=chroms.label),
        args,
        fig_name='preinterf_cM_rf_curve',
        x_label='pre-interf morgans (M)',
        y_label='recombinant frequency (rf)',
        y_lim=1.0,
        post_plot=plot_canonical_map_funcs,
    )


def fit_haldane_p_to_data(
    dataset_cM: list[float], dataset_rf: list[float]
) -> tuple[float, list[float]]:
    """
    return type: (p, y_data_fitted)
    """
    param, _ = curve_fit(haldane_input_cM, dataset_cM, dataset_rf, p0=[2])
    return (param, haldane_input_cM(dataset_cM, param))
