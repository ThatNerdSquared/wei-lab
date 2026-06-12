import itertools
import operator
from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from src.specific_interf_sims.plotter import save_plot

from croissant.chromosomes import ChromosomeSet
from croissant.chromosomes.core import ChromosomeLike
from croissant.map_funcs import (
    MapFuncData,
    generate_comparative_map_func_data,
    plot_canonical_map_funcs,
)
from croissant.outputs.interpolated_cM_rf_curve import compare_interpolated_rf
from croissant.outputs.output import MultipleCurves, label_figure


@dataclass(slots=True, init=False)
class DeltaRFCurve:
    fig_name: ClassVar[str] = '∆RF per morgan'
    xlabel: ClassVar[str] = 'morgans (M)'
    ylabel: ClassVar[str] = '∆ recombinant frequency (∆RF)'
    data: MultipleCurves
    output_file: Figure
    with_interf: bool

    def __init__(self, *, with_interf: bool):
        self.with_interf = with_interf

    def build(self, chrom_sets: dict[str, Sequence[ChromosomeSet[ChromosomeLike]]]):
        self.data = process_delta_rf_data(
            chrom_sets['reference'][0], chrom_sets['experimental']
        )
        map_data = generate_comparative_map_func_data(
            self.data['no drive'], operator.sub
        )
        self.output_file = plot_comparative_drive_rf_curves(self.data, map_data)
        label_figure(self)
        save_plot(
            self.output_file,
            additional_annotation='_w_interf' if self.with_interf else '_no_interf',
        )


def process_delta_rf_data(
    ref_chromset: ChromosomeSet[ChromosomeLike],
    exp_chromsets: Sequence[ChromosomeSet[ChromosomeLike]],
) -> MultipleCurves:
    return compare_interpolated_rf(ref_chromset, exp_chromsets, operator.sub)


def plot_comparative_drive_rf_curves(data: MultipleCurves, map_data: MapFuncData):
    fig = plt.figure(figsize=(9.0, 7.0))
    ax = fig.add_subplot()
    no_drive_curve = None
    for curve_label, points in data.items():
        cM, rf = zip(*points[1:])
        if curve_label == 'no drive':
            no_drive_curve = ax.plot(cM, rf, label=curve_label)
        else:
            _ = ax.plot(cM, rf, label=curve_label)
    if no_drive_curve:
        no_drive_curve[0].remove()

    plot_canonical_map_funcs(ax, map_data)

    del data['no drive']
    datapts_without_first = list(map(lambda x: x[1:], data.values()))
    max_y = max(
        itertools.chain.from_iterable(datapts_without_first), key=operator.itemgetter(1)
    )[1]
    min_y_of_data = min(
        itertools.chain.from_iterable(datapts_without_first), key=operator.itemgetter(1)
    )[1]
    min_y_of_map_func_data = min(
        itertools.chain(map_data.haldane, map_data.morgan, map_data.kosambi)
    )
    min_y = min(min_y_of_data, min_y_of_map_func_data)
    _ = ax.set_ylim(
        bottom=(min_y * 0.5) if min_y > 0 else (min_y * 1.5), top=max_y * 1.5
    )

    return fig
