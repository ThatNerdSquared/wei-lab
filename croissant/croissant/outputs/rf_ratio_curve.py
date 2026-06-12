import operator
from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar

from matplotlib.figure import Figure
from src.specific_interf_sims.plotter import save_plot

from croissant.chromosomes import ChromosomeSet
from croissant.chromosomes.core import ChromosomeLike
from croissant.map_funcs import generate_comparative_map_func_data
from croissant.outputs.delta_rf_curve import plot_comparative_drive_rf_curves
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
        map_data = generate_comparative_map_func_data(
            self.data['no drive'], operator.truediv
        )
        self.output_file = plot_comparative_drive_rf_curves(self.data, map_data)
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
