import itertools
import operator
from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from src.specific_interf_sims.plotter import save_plot
from src.utils.standard_funcs import haldane_input_cM, kosambi

from croissant.chromosomes import ChromosomeSet
from croissant.chromosomes.core import ChromosomeLike
from croissant.outputs.interpolated_cM_rf_curve import compare_interpolated_rf
from croissant.outputs.output import MultipleCurves, label_figure


@dataclass(slots=True, init=False)
class RFRatioCurve:
    fig_name: ClassVar[str] = 'RF*/RF per morgan'
    xlabel: ClassVar[str] = 'morgans (M)'
    ylabel: ClassVar[str] = 'recombinant frequency w drive/w no drive (RF*)'
    data: MultipleCurves
    output_file: Figure
    with_interf: bool

    def __init__(self, *, with_interf: bool):
        self.with_interf = with_interf

    def build(self, chrom_sets: dict[str, Sequence[ChromosomeSet[ChromosomeLike]]]):
        self.data = process_rf_ratio_data(
            chrom_sets['reference'][0], chrom_sets['experimental']
        )
        self.output_file = plot_rf_ratio_curve(self.data)
        label_figure(self)
        save_plot(
            self.output_file,
            additional_annotation='_w_interf' if self.with_interf else '_no_interf',
        )


def process_rf_ratio_data(
    ref_chromset: ChromosomeSet[ChromosomeLike],
    exp_chromsets: Sequence[ChromosomeSet[ChromosomeLike]],
) -> MultipleCurves:
    return compare_interpolated_rf(ref_chromset, exp_chromsets, operator.truediv)


def plot_rf_ratio_curve(data: MultipleCurves):
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

    raw_ref_cM, raw_ref_rf = zip(*data['no drive'][1:])
    ref_cM = np.array(raw_ref_cM)
    ref_rf = np.array(raw_ref_rf)
    _ = ax.plot(ref_cM, haldane_input_cM(ref_cM) / ref_rf, label='haldane')
    _ = ax.plot(ref_cM, ref_cM / ref_rf, label='morgan')
    _ = ax.plot(ref_cM, kosambi(ref_cM) / ref_rf, label='kosambi')
    _ = ax.legend()

    del data['no drive']
    datapts_without_first = list(map(lambda x: x[1:], data.values()))
    max_y = max(
        itertools.chain.from_iterable(datapts_without_first), key=operator.itemgetter(1)
    )[1]
    min_y = min(
        itertools.chain.from_iterable(datapts_without_first), key=operator.itemgetter(1)
    )[1]
    _ = ax.set_ylim(bottom=min_y * 0.5, top=max_y * 1.5)

    return fig
