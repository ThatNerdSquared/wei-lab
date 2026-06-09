import numpy as np

from croissant.chromosomes import ChromosomeSet
from croissant.chromosomes.core import ChromosomeLike, num_strands, strandify
from croissant.haplotype_table import (
    HaplotypeTable,
    indicators_to_totals,
    make_indicator_tbl_chrom_sized,
)
from croissant.outputs.output import ListOfPoints


def calculate_cM_rf_data(
    chrom_set: ChromosomeSet[ChromosomeLike],
) -> ListOfPoints:
    # one approach: create two haplotypetbls, one for all COs, one
    # for all visible COs, and then can add columnwise to get the
    # totals at each point of the chromosome length? then can
    # iterate CO-wise per chromosome, hopefully saving a lot of
    # time (since expected num of COs per strand is not *that*
    # high, or at least << 1000, which is the chrom length we
    # usually use)

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
