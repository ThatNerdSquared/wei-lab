from collections.abc import Sequence
from typing import Self

import numpy as np

from croissant.chromosomes.core import ChromosomeLike, num_strands, segments, strandify

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

    @classmethod
    def of_size(cls, *, num_strands: int, chrom_length: int) -> Self:
        return cls(tbl=make_indicator_tbl(height=num_strands, width=chrom_length))

    @classmethod
    def from_chromosomes(cls, chroms: Sequence[ChromosomeLike]) -> Self:
        res = cls(tbl=make_indicator_tbl_chrom_sized(chroms))
        for strand, curr_row in zip(strandify(chroms), res.tbl):
            for segment in segments(strand):
                curr_row[segment.start : segment.end + 1] = segment.data.value
        return res

    def convert_to_visible_co_indicators(self):
        base_haplotypes = self.tbl[:, 0]
        self.tbl[:, 1:] = self.tbl[:, 1:] ^ np.expand_dims(base_haplotypes, axis=1)
        # can't have visible CO in the first bin, since can't have a "more
        # leftmost" marker to indicate a visible CO w
        self.tbl[:, 0] = 0

    def to_totals(self) -> TotalTbl:
        return indicators_to_totals(self.tbl)


def indicators_to_totals(tbl: IndicatorTbl) -> TotalTbl:
    return np.sum(tbl, axis=0)
