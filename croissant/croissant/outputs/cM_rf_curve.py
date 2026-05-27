from croissant.chromosomes import ChromosomeSet
from croissant.chromosomes.core import ChromosomeLike, num_strands, strandify
from croissant.haplotype_table import make_indicator_tbl_chrom_sized
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

    all_cos_tbl = make_indicator_tbl_chrom_sized(chrom_set.chroms)
    all_cos_tbl[:] = 0
    visible_cos_tbl = make_indicator_tbl_chrom_sized(chrom_set.chroms)
    visible_cos_tbl[:] = 0

    for strand, all_cos_row, visible_cos_row in zip(
        strandify(chrom_set.chroms), all_cos_tbl, visible_cos_tbl
    ):
        for co_loc in strand.get_crossover_locations():
            all_cos_row[co_loc] = 1
            if strand.haplotype_at(co_loc) != strand.base_haplotype:
                visible_cos_row[co_loc] = 1

    res: ListOfPoints = []
    for all_cos_col, visible_cos_col in zip(all_cos_tbl.T, visible_cos_tbl.T):
        morgans = all_cos_col.sum() / num_strands(chrom_set.chroms)
        rf = visible_cos_col.sum() / num_strands(chrom_set.chroms)
        res.append((morgans, rf))

    return res
