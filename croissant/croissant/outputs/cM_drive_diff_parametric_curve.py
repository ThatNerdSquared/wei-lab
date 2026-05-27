import operator
from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from src.specific_interf_sims.plotter import save_plot

from croissant.chromosomes import ChromosomeSet
from croissant.chromosomes.core import ChromosomeLike
from croissant.outputs.output import label_figure

type PairedCMData = list[tuple[float, float]]


@dataclass(slots=True, init=False)
class cMDriveDiffParametricCurve:
    fig_name: ClassVar[str] = 'cM with/without drive, parametric wrt recomb rate'
    xlabel: ClassVar[str] = 'morgans (M) w/o drive'
    ylabel: ClassVar[str] = 'morgans (M) w/ drive'
    data: PairedCMData
    output_file: Figure
    with_interf: bool

    def __init__(self, *, with_interf: bool):
        self.with_interf = with_interf

    def build(self, chrom_sets: dict[str, Sequence[ChromosomeSet[ChromosomeLike]]]):
        self.data = process_paired_cM_data(
            chrom_sets['reference'], chrom_sets['experimental']
        )
        self.output_file = plot_cM_parametric_curve(self.data)
        label_figure(self)
        save_plot(
            self.output_file,
            additional_annotation='_w_interf' if self.with_interf else '_no_interf',
        )


# starting w totals, let's expand to full-data later...
def process_paired_cM_data(
    ref_chromsets: Sequence[ChromosomeSet[ChromosomeLike]],
    exp_chromsets: Sequence[ChromosomeSet[ChromosomeLike]],
) -> PairedCMData:
    data: PairedCMData = []
    for ref_chroms, exp_chroms in zip(ref_chromsets, exp_chromsets):
        data.append(
            (
                sum((x.get_num_crossovers() for x in ref_chroms))
                / ref_chroms.num_chroms,
                sum((x.get_num_crossovers() for x in exp_chroms))
                / exp_chroms.num_chroms,
            )
        )
    data.sort(key=operator.itemgetter(0))
    return data


def plot_cM_parametric_curve(data: PairedCMData):
    fig = plt.figure(figsize=(9.0, 7.0))
    ax = fig.add_subplot()
    _ = ax.plot(*zip(*data))
    return fig
