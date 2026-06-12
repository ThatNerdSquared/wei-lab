import operator
from collections.abc import Sequence
from typing import ClassVar, override

from src.specific_interf_sims.interf_schemes import BinArray
from src.specific_interf_sims.plotter import save_plot

from croissant.chromosomes import ChromosomeSet
from croissant.chromosomes.core import ChromosomeLike
from croissant.map_funcs import generate_comparative_map_func_data
from croissant.outputs.delta_rf_curve import (
    DeltaRFCurve,
    plot_comparative_drive_rf_curves,
)
from croissant.outputs.output import MultipleCurves, label_figure
from croissant.outputs.preinterf_cM_rf_curve import calculate_preinterf_cM_rf_data


class PreinterfDeltaRFCurve(DeltaRFCurve):
    fig_name: ClassVar[str] = '∆RF per pre-interf morgan'
    xlabel: ClassVar[str] = 'pre-interf morgans (M)'
    pre_interf_dist: BinArray

    def __init__(self, *, with_interf: bool, pre_interf_dist: BinArray):
        super().__init__(with_interf=with_interf)
        self.pre_interf_dist = pre_interf_dist

    @override
    def build(self, chrom_sets: dict[str, Sequence[ChromosomeSet[ChromosomeLike]]]):
        self.data = process_preinterf_delta_rf_data(
            chrom_sets['reference'][0], chrom_sets['experimental'], self.pre_interf_dist
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


def process_preinterf_delta_rf_data(
    ref_chromset: ChromosomeSet[ChromosomeLike],
    exp_chromsets: Sequence[ChromosomeSet[ChromosomeLike]],
    pre_interf_dist: BinArray,
) -> MultipleCurves:
    raw_ref_chrom_data = calculate_preinterf_cM_rf_data(ref_chromset, pre_interf_dist)
    raw_ref_chrom_data.sort(key=operator.itemgetter(0))
    exp_chrom_data = [
        calculate_preinterf_cM_rf_data(x, pre_interf_dist) for x in exp_chromsets
    ]

    data: dict[str, list[tuple[float, float]]] = {}
    data[ref_chromset.label] = raw_ref_chrom_data
    for chrom_set, dataset in zip(exp_chromsets, exp_chrom_data):
        dataset.sort(key=operator.itemgetter(0))
        res = [
            (ref[0], exp[1] - ref[1]) for ref, exp in zip(raw_ref_chrom_data, dataset)
        ]
        res.sort(key=operator.itemgetter(0))
        data[chrom_set.label] = res
    return data
