import numpy as np
from src.specific_interf_sims.interf_schemes import BinArray

from croissant.chromosomes import ChromosomeSet
from croissant.chromosomes.core import ChromosomeLike, num_strands
from croissant.haplotype_table import HaplotypeTable
from croissant.outputs.output import ListOfPoints


# TODO: write test!!
def calculate_preinterf_cM_rf_data(
    chrom_set: ChromosomeSet[ChromosomeLike], pre_interf_dist: BinArray
) -> ListOfPoints:
    visible_cos_tbl = HaplotypeTable.from_chromosomes(chrom_set.chroms)
    visible_cos_tbl.convert_to_visible_co_indicators()
    visible_cos_tbl_totals = visible_cos_tbl.to_totals()
    rf_across_chrom = visible_cos_tbl_totals / num_strands(chrom_set.chroms)
    morgans_across_chrom = np.cumulative_sum(pre_interf_dist)
    return [(morgan, rf) for morgan, rf in zip(morgans_across_chrom, rf_across_chrom)]
