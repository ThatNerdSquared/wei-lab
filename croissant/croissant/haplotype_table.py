import itertools
from collections.abc import Sequence
from typing import Self

import numpy as np

from croissant.chromosomes.core import ChromosomeLike, strandify


class HaplotypeTable:
    tbl: np.ndarray[tuple[int, int], np.dtype[np.int64]]

    def __init__(self, *, num_strands: int, chrom_length: int):
        self.tbl = np.zeros((num_strands, chrom_length), dtype=np.int64)
        self.tbl[:] = -1

    @classmethod
    def from_chromosomes(cls, chroms: Sequence[ChromosomeLike]) -> Self:
        res = cls(
            num_strands=len(chroms) * chroms[0].num_strands,
            chrom_length=chroms[0].length,
        )
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
