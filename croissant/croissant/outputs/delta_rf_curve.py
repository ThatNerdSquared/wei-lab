from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.polynomial import polynomial
from src.specific_interf_sims.plotter import save_plot
from src.utils.standard_funcs import haldane_input_cM, kosambi

from croissant.chromosomes import ChromosomeSet
from croissant.chromosomes.core import ChromosomeLike
from croissant.outputs.cM_rf_curve import calculate_cM_rf_data
from croissant.outputs.output import (
    MultipleCurves,
    axis_lists_to_points,
    label_figure,
    points_to_axis_lists,
)


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

    raw_ref_chrom_data = calculate_cM_rf_data(ref_chromset)
    ref_cM, ref_rf = points_to_axis_lists(raw_ref_chrom_data)
    domain = [0, max(ref_cM) * 5]
    ref_curve = polynomial.Polynomial.fit(ref_cM, ref_rf, 2, domain=domain)

    data: dict[str, list[tuple[float, float]]] = {}
    data['no drive'] = axis_lists_to_points(
        ref_curve.linspace(n=len(raw_ref_chrom_data))
    )
    exp_chrom_data = [calculate_cM_rf_data(x) for x in exp_chromsets]
    for chrom_set, dataset in zip(exp_chromsets, exp_chrom_data):
        exp_cM, exp_rf = points_to_axis_lists(dataset)
        exp_curve = polynomial.Polynomial.fit(exp_cM, exp_rf, 2, domain=domain)
        delta_curve = exp_curve - ref_curve
        data[chrom_set.label] = axis_lists_to_points(
            delta_curve.linspace(n=len(raw_ref_chrom_data))
        )
    return data


def plot_delta_rf_curve(data: MultipleCurves):
    fig = plt.figure(figsize=(9.0, 7.0))
    ax = fig.add_subplot()
    for curve_label, points in data.items():
        _ = ax.plot(*zip(*points), label=curve_label)

    raw_ref_cM, ref_rf = zip(*data['no drive'])
    ref_cM = np.array(raw_ref_cM)
    ref_cM.sort()
    _ = ax.plot(
        ref_cM,
        list(map(float.__sub__, haldane_input_cM(ref_cM), ref_rf)),
        label='haldane',
    )
    _ = ax.plot(ref_cM, list(map(float.__sub__, ref_cM, ref_rf)), label='morgan')
    _ = ax.plot(
        ref_cM, list(map(float.__sub__, kosambi(ref_cM), ref_rf)), label='kosambi'
    )
    _ = ax.legend()

    return fig
