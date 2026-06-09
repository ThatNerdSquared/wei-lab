import itertools
from collections.abc import Sequence
from typing import Self

import numpy as np

from croissant.chromosomes.core import ChromosomeLike, num_strands, strandify

type IndicatorTbl = np.ndarray[tuple[int, int], np.dtype[np.int64]]
type TotalTbl = np.ndarray[tuple[int], np.dtype[np.int64]]


def make_indicator_tbl(*, height: int, width: int) -> IndicatorTbl:
    tbl = np.zeros((height, width), dtype=np.int64)
    tbl[:] = -1
    return tbl


def make_indicator_tbl_chrom_sized(
    chroms: Sequence[ChromosomeLike],
) -> IndicatorTbl:
    return make_indicator_tbl(
        height=num_strands(chroms),
        width=chroms[0].length,
    )


class HaplotypeTable:
    tbl: IndicatorTbl

    def __init__(self, *, tbl: IndicatorTbl):
        self.tbl = tbl
        # if not no_build:

    @classmethod
    def of_size(cls, *, num_strands: int, chrom_length: int) -> Self:
        return cls(tbl=make_indicator_tbl(height=num_strands, width=chrom_length))

    @classmethod
    def from_chromosomes(cls, chroms: Sequence[ChromosomeLike]) -> Self:
        res = cls(tbl=make_indicator_tbl_chrom_sized(chroms))
        for strand, curr_row in zip(strandify(chroms), res.tbl):
            if strand.get_num_crossovers() == 0:
                curr_row[:] = strand.base_haplotype.value
                continue

            co_locs = strand.get_crossover_locations()
            curr_row[0 : co_locs[0]] = strand.base_haplotype.value

            for curr_loc, next_loc in itertools.pairwise(co_locs):
                curr_row[curr_loc:next_loc] = strand.haplotype_at(curr_loc).value
            curr_row[co_locs[-1] :] = strand.haplotype_at(co_locs[-1]).value

        return res

    def convert_to_visible_co_indicators(self):
        base_haplotypes = self.tbl[:, 0]
        self.tbl[:, 1:] = self.tbl[:, 1:] ^ np.expand_dims(base_haplotypes, axis=1)

    def to_totals(self) -> TotalTbl:
        return indicators_to_totals(self.tbl)


def indicators_to_totals(tbl: IndicatorTbl) -> TotalTbl:
    return np.sum(tbl, axis=0)
