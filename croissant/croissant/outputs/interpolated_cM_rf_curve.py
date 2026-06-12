import itertools
import operator
from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import markers
from matplotlib.figure import Figure
from src.specific_interf_sims.plotter import save_plot

from croissant.chromosomes import ChromosomeSet
from croissant.chromosomes.core import ChromosomeLike
from croissant.outputs.cM_rf_curve import calculate_cM_rf_data
from croissant.outputs.output import (
    ListOfPoints,
    MultipleCurves,
    axis_lists_to_points,
    label_figure,
    points_to_axis_lists,
)


@dataclass(slots=True, init=False)
class InterpolatedcMRFCurve:
    fig_name: ClassVar[str] = 'interpolated cM-RF curves'
    xlabel: ClassVar[str] = 'morgans (M)'
    ylabel: ClassVar[str] = 'recombinant frequency (RF)'
    data: MultipleCurves
    output_file: Figure
    with_interf: bool

    def __init__(self, *, with_interf: bool):
        self.with_interf = with_interf

    def build(self, chrom_sets: dict[str, Sequence[ChromosomeSet[ChromosomeLike]]]):
        chroms = chrom_sets['to_interpolate']
        cM_rf_data = [calculate_cM_rf_data(x) for x in chroms]
        cM_rf_data_labelled = {k.label: v for k, v in zip(chroms, cM_rf_data)}
        interpolants = interpolate_cM_rf_data(chroms, provided_cM_rf_data=cM_rf_data)
        self.data = cM_rf_data_labelled | interpolants
        self.output_file = plot_interpolated_cM_rf_curve(self.data)
        label_figure(self)
        save_plot(
            self.output_file,
            additional_annotation='_w_interf' if self.with_interf else '_no_interf',
        )


def interpolate_cM_rf_data(
    sims: Sequence[ChromosomeSet[ChromosomeLike]],
    provided_cM_rf_data: list[ListOfPoints] | None = None,
) -> MultipleCurves:
    """
    Interpolates the cM-rf curve across [0, d], where d is the min of
    max-cM-value across all sims.
    """

    cM_rf_data = (
        provided_cM_rf_data
        if provided_cM_rf_data
        else [calculate_cM_rf_data(x) for x in sims]
    )
    max_x = min(map(lambda x: max(x, key=operator.itemgetter(0))[0], cM_rf_data))
    xs = np.linspace(0, max_x)

    data: dict[str, list[tuple[float, float]]] = {}
    for chroms, dataset in zip(sims, cM_rf_data):
        dataset.sort(key=operator.itemgetter(0))
        raw_cM, raw_rf = points_to_axis_lists(dataset)
        rf_pts = np.interp(xs, raw_cM, raw_rf)
        data[f'{chroms.label} (interp.)'] = axis_lists_to_points((xs, rf_pts))
    return data


def plot_interpolated_cM_rf_curve(data: MultipleCurves):
    fig = plt.figure(figsize=(9.0, 7.0))
    ax = fig.add_subplot()
    marker = markers.MarkerStyle('D', fillstyle='none')
    for curve_label, points in data.items():
        if '(interp.)' in curve_label:
            _ = ax.plot(*zip(*points), label=curve_label)
        else:
            # thin out the points for clearer viewing on plot
            thinned_pts = itertools.islice(points, 0, None, 20)
            _ = ax.scatter(*zip(*thinned_pts), marker=marker, label=curve_label)
    _ = ax.legend()
    return fig
