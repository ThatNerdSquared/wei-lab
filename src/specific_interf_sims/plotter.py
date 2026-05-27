from datetime import datetime
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib import colormaps
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.axes3d import Axes3D
from scipy.optimize import curve_fit

from src.specific_interf_sims.data_packet import DataPacket, ThreeDimDataPacket
from src.utils.standard_funcs import haldane_input_cM, kosambi


def _fmt_plot_name[T](data_packet: DataPacket[T] | ThreeDimDataPacket[T]):
    num_chrom_in_ks = str(data_packet.num_chroms / 1000)
    num_bins_in_ks = str(data_packet.num_bins / 1000)
    return f'{data_packet.fig_name} -- {num_chrom_in_ks}k chroms @ {num_bins_in_ks}k bins / chrom'  # noqa: E501


def _fmt_plot_filename(plot_title: str, fmt: str, additional_annotation: str = ''):
    curr_time = datetime.now().strftime('%Y-%m-%d__%H-%M-%S')
    plot_title_fileready = (
        plot_title.replace(' bins / chrom', '')
        .replace('chroms @', '--')
        .replace(' -- ', '--')
        .replace('.', '_')
        .replace(' ', '_')
        .replace('/', '_')
        .strip()
        .lower()
    )
    return f'out--{curr_time}--{plot_title_fileready}{additional_annotation}.{fmt}'


class PlotGridDimensions(NamedTuple):
    width: int
    height: int


class SingleFigPlotter:
    fig: Figure
    fig_grid_size: PlotGridDimensions
    _subplot_idx: int = 1
    _individual_legends: bool

    def __init__(
        self,
        fig_grid_size: PlotGridDimensions,
        figtitle: str,
        subplot_size: tuple[float, float],
        individual_legends: bool = False,
    ):
        self.fig = plt.figure(
            figsize=(
                float(fig_grid_size.width * subplot_size[0]),
                float(fig_grid_size.height * subplot_size[1]),
            ),
            layout='constrained',
        )
        _ = self.fig.suptitle(figtitle, wrap=True)
        self.fig_grid_size = fig_grid_size
        self._individual_legends = individual_legends

    def add_plot[T](self, data_packet: DataPacket[T], title: str):
        ax = self.fig.add_subplot(
            self.fig_grid_size.height, self.fig_grid_size.width, self._subplot_idx
        )
        data_packet.curve_adder(data_packet, ax)

        _ = ax.set_xlabel(data_packet.xlabel or '')
        _ = ax.set_ylabel(data_packet.ylabel or '')
        if self._individual_legends:
            _ = ax.legend()
        else:
            handles, labels = ax.get_legend_handles_labels()
            _ = self.fig.legend(handles, labels)
        _ = ax.set_ylim(bottom=0, top=data_packet.ylim)
        _ = ax.set_title(title)

        self._subplot_idx += 1


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


def standard_plotter_threedim[T](data_packet: ThreeDimDataPacket[T]) -> Figure:
    fig = plt.figure(figsize=(9.0, 7.0))
    ax = fig.add_subplot(projection='3d')
    data_packet.surface_adder(data_packet, ax)

    _ = ax.set_xlabel(data_packet.xlabel or '')
    _ = ax.set_ylabel(data_packet.ylabel or '')
    _ = ax.legend()
    ax.set_zlim(bottom=0, top=data_packet.zlim)

    _ = fig.suptitle(_fmt_plot_name(data_packet))
    return fig


def save_plot(plot: Figure, fmt: str = 'svg', additional_annotation: str = ''):
    plot_filename = _fmt_plot_filename(plot.get_suptitle(), fmt, additional_annotation)
    plot.savefig(plot_filename, format=fmt)
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


def histogram_curve_adder_minor_tick_size_1(data: DataPacket[npt.ArrayLike], ax: Axes):
    _ = ax.set_xticks(np.arange(0, 20, 1), minor=True)
    for k, v in data.data.items():
        _ = ax.stairs(
            v,
            label=k,
        )


def histogram_unpacked_data_curve_adder(
    data: DataPacket[tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]],
    ax: Axes,
):
    for k, v in data.data.items():
        _ = (ax.stairs(v[0], v[1], label=k),)


def three_dim_surface_adder(
    data: ThreeDimDataPacket[
        tuple[
            np.ndarray[tuple[int, int], np.dtype[np.int64]],
            np.ndarray[tuple[int, int], np.dtype[np.int64]],
            np.ndarray[tuple[int, int], np.dtype[np.float64]],
        ]
    ],
    ax: Axes3D,
):
    label, raw_data = data.data
    _ = ax.plot_surface(*raw_data, label=label, cmap=colormaps['magma'])


def plot_canonical_map_funcs(ax: Axes, up_to: float = 1.0):
    ax.axhline(y=0.5, color='tab:red', label='50% RF').set_dashes((5, 2))
    x = np.linspace(0, up_to, int(1000 * up_to))
    _ = ax.plot(x, haldane_input_cM(x), label='haldane')
    _ = ax.plot(x, x, label='morgan')
    _ = ax.plot(x, kosambi(x), label='kosambi')
    _ = ax.legend()


def fit_haldane_p_to_data(
    dataset_cM: list[float], dataset_rf: list[float]
) -> tuple[float, list[float]]:
    """
    return type: (p, y_data_fitted)
    """
    param, _ = curve_fit(haldane_input_cM, dataset_cM, dataset_rf, p0=[2])
    return (param, haldane_input_cM(dataset_cM, param))
