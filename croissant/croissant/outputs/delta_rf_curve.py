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
from croissant.outputs.cM_rf_curve import calculate_cM_rf_data
from croissant.outputs.output import MultipleCurves, label_figure, smart_dictify


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
    ref_data = smart_dictify(
        raw_ref_chrom_data, resolver=lambda x: np.mean(x, dtype=np.float64)
    )
    exp_chrom_data = [calculate_cM_rf_data(x) for x in exp_chromsets]

    data: dict[str, list[tuple[float, float]]] = {}
    data[ref_chromset.label] = raw_ref_chrom_data
    for chrom_set, dataset in zip(exp_chromsets, exp_chrom_data):
        num_pts_discarded = 0
        res: list[tuple[float, float]] = []
        for pt in dataset:
            try:
                res.append((pt[0], pt[1] - ref_data[pt[0]]))
            except KeyError:
                num_pts_discarded += 1
                continue
        res.sort(key=operator.itemgetter(0))
        data[chrom_set.label] = res
        print(
            f'∆RF calculated for {chrom_set.label} with {num_pts_discarded}/{len(dataset)} points discarded.'  # noqa: E501
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
