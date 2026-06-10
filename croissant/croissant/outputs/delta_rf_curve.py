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
from croissant.outputs.cM_rf_curve import calculate_cM_rf_data
from croissant.outputs.interpolated_cM_rf_curve import interpolate_cM_rf_data
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
        self.output_file = plot_delta_rf_curve(self.data)
        label_figure(self)
        save_plot(
            self.output_file,
            additional_annotation='_w_interf' if self.with_interf else '_no_interf',
        )


def process_delta_rf_data(
    ref_chromset: ChromosomeSet[ChromosomeLike],
    exp_chromsets: Sequence[ChromosomeSet[ChromosomeLike]],
) -> MultipleCurves:
    """
    Assumes that the first chromset is the reference, against which
    the "delta" for all other chromsets is calculated.
    """

    data: dict[str, list[tuple[float, float]]] = {}

    interpolants = interpolate_cM_rf_data((ref_chromset, *exp_chromsets))
    ref_rf_curve = interpolants[f'{ref_chromset.label} (interp.)']

    for k, v in interpolants.items():
        data[k.replace(' (interp.)', '')] = [
            (ref[0], exp[1] - ref[1]) for ref, exp in zip(ref_rf_curve, v)
        ]

    data[ref_chromset.label] = calculate_cM_rf_data(ref_chromset)
    return data


def plot_delta_rf_curve(data: MultipleCurves):
    fig = plt.figure(figsize=(9.0, 7.0))
    ax = fig.add_subplot()
    no_drive_curve = None
    for curve_label, points in data.items():
        cM, rf = zip(*points)
        if curve_label == 'no drive':
            no_drive_curve = ax.plot(cM[1:], rf[1:], label=curve_label)
        else:
            _ = ax.plot(cM[1:], rf[1:], label=curve_label)
    if no_drive_curve:
        no_drive_curve[0].remove()

    raw_ref_cM, raw_ref_rf = zip(*data['no drive'])
    ref_cM = np.array(raw_ref_cM)
    ref_rf = np.array(raw_ref_rf)
    _ = ax.plot(ref_cM[1:], (haldane_input_cM(ref_cM) - ref_rf)[1:], label='haldane')
    _ = ax.plot(ref_cM[1:], (ref_cM - ref_rf)[1:], label='morgan')
    _ = ax.plot(ref_cM[1:], (kosambi(ref_cM) - ref_rf)[1:], label='kosambi')
    _ = ax.legend()

    return fig
