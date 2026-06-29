from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from src.specific_interf_sims.plotter import save_plot

from croissant.chromosomes import ChromosomeSet
from croissant.chromosomes.core import ChromosomeLike, num_strands, strandify
from croissant.haplotype_table import (
    HaplotypeTable,
    indicators_to_totals,
    make_indicator_tbl_chrom_sized,
)
from croissant.map_funcs import (
    MapFuncData,
    generate_canonical_map_func_data,
    plot_canonical_map_funcs,
)
from croissant.outputs.output import ListOfPoints, MultipleCurves, label_figure


@dataclass(slots=True, init=False)
class cMRFCurve:
    fig_name: ClassVar[str] = 'cM-RF curve'
    xlabel: ClassVar[str] = 'morgans (M)'
    ylabel: ClassVar[str] = 'recombinant frequency (rf)'
    data: MultipleCurves
    output_file: Figure
    annotation: str
    plot_map_funcs_up_to: float

    def __init__(self, *, annotation: str = '', plot_map_funcs_up_to: float = 1.0):
        self.annotation = annotation
        self.plot_map_funcs_up_to = plot_map_funcs_up_to

    def build(self, chrom_sets: dict[str, Sequence[ChromosomeSet[ChromosomeLike]]]):
        chroms = chrom_sets['data']
        self.data = dict(map(lambda x: (x.label, calculate_cM_rf_data(x)), chroms))
        map_data = generate_canonical_map_func_data(up_to=self.plot_map_funcs_up_to)
        self.output_file = plot_cM_rf_curve(self.data, map_data)
        label_figure(self)
        save_plot(self.output_file, additional_annotation=self.annotation)


# TODO: write test!!
def calculate_cM_rf_data(
    chrom_set: ChromosomeSet[ChromosomeLike],
) -> ListOfPoints:

    visible_cos_tbl = HaplotypeTable.from_chromosomes(chrom_set.chroms)
    visible_cos_tbl.convert_to_visible_co_indicators()
    visible_cos_tbl_totals = visible_cos_tbl.to_totals()
    rf_across_chrom = visible_cos_tbl_totals / num_strands(chrom_set.chroms)

    all_cos_tbl = make_indicator_tbl_chrom_sized(chrom_set.chroms)
    all_cos_tbl[:] = 0
    for strand, all_cos_row in zip(strandify(chrom_set.chroms), all_cos_tbl):
        for co_loc in strand.get_crossover_locations():
            all_cos_row[co_loc] = 1
    all_cos_tbl_totals = indicators_to_totals(all_cos_tbl)
    all_cos_cumulative = np.cumulative_sum(all_cos_tbl_totals)
    morgans_across_chrom = all_cos_cumulative / num_strands(chrom_set.chroms)

    return [(morgan, rf) for morgan, rf in zip(morgans_across_chrom, rf_across_chrom)]


def plot_cM_rf_curve(data: MultipleCurves, map_data: MapFuncData) -> Figure:
    fig = plt.figure(figsize=(9.0, 7.0))
    ax = fig.add_subplot()
    for curve_label, points in data.items():
        cM, rf = zip(*points)
        _ = ax.plot(cM, rf, label=curve_label)

    plot_canonical_map_funcs(ax, map_data)
    ax.axhline(y=0.5, color='tab:red', label='50% RF').set_dashes((5, 2))
    _ = ax.set_ylim(bottom=0, top=1)

    return fig
