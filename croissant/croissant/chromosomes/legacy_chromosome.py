from collections.abc import Iterable, Iterator, Sequence
from itertools import pairwise
from random import randint
from typing import ClassVar, Self, override
from uuid import UUID, uuid4

import numpy as np

from croissant.chromosomes.core import Haplotype, Segment, Strand


class LegacyChromosome:
    """
    Legacy Chromosome class with no sister chromatids or haplotype tracking.

    Previously used in simulations that did not account for sister chromatids.
    Because of this, every crossover has a different haplotype on either side,
    allowing us to determine whether a visible CO was present by determining
    whether there were an even or odd number of COs. No longer recommended for
    use as lack of sister chromatids leads to unrealistic RF (goes above 50,
    because you don't have limiter of two sisters crossing over, or drawing one
    of the non-recombined chromatids in the pool).

    No @deprecated annotation attached, to prevent type errors in old scripts
    that did use this.
    """

    num_strands: ClassVar[int] = 1
    base_haplotype: Haplotype = Haplotype.BLUE

    def __init__(self, length: int = 100):
        self.id: UUID = uuid4()
        self.length: int = length
        self._crossovers: dict[int, bool] = {}

    @override
    def __str__(self) -> str:
        res = ['-'] * self.length
        for co_loc in self._crossovers.keys():
            res[co_loc] = 'x'
        return ''.join(res)

    def insert_crossover(self, location: int):
        if location > self.length:
            raise ValueError(
                f'Location is out of bounds for {self.length}-bin chromosome!'
            )
        self._crossovers[location] = True

    def randomly_add_crossovers(self, num: int):
        for _ in range(num):
            self.insert_crossover(randint(0, self.length - 1))

    def get_num_crossovers(self) -> int:
        return len(self._crossovers)

    def get_crossover_locations(self) -> list[int]:
        return sorted(self._crossovers.keys())

    def haplotype_at(self, location: int) -> Haplotype:
        if self.get_num_crossovers() == 0:
            return self.base_haplotype

        num_before = 0
        for co_loc in self._crossovers:
            if co_loc <= location:
                num_before += 1
                continue
            break

        return Haplotype(num_before % 2)

    def has_crossover_at(self, location: int) -> bool:
        try:
            return self._crossovers[location]
        except KeyError:
            return False

    def has_visible_crossover(self, start: int = 0, end: int | None = None) -> bool:
        endpoint: int = end if end is not None else self.length
        crossover_locations = sorted(self._crossovers.keys())
        num_crossovers_in_region = len(
            [x for x in crossover_locations if x >= start and x <= endpoint]
        )
        return num_crossovers_in_region % 2 == 1

    def into_list[C: LegacyChromosome](self: C, num_times: int) -> Sequence[C]:
        return [self.__class__(length=self.length) for _ in range(num_times)]

    def as_strands(self) -> Iterable[Strand]:
        yield self

    def into_haplotype_tbl(self, input_arr: np.ndarray[tuple[int], np.dtype[np.int64]]):
        assert (1, self.length) == input_arr.shape
        input_arr = input_arr[0]

        input_arr[:] = 0
        locs = self.get_crossover_locations()
        if not locs:
            return

        curr_hap = True
        for curr_co, next_co in pairwise(locs):
            input_arr[curr_co:next_co] = 1 if curr_hap else 0
            curr_hap = not curr_hap
        input_arr[locs[-1] :] = 1 if curr_hap else 0

    @classmethod
    def from_haplotype_tbl(
        cls, input_arr: np.ndarray[tuple[int], np.dtype[np.int64]]
    ) -> Self:
        print(input_arr)
        raise NotImplementedError()

    def fill_from_segments(self, segments: Iterator[Segment]) -> None:
        self._crossovers = {}
        for segment in segments:
            self.insert_crossover(int(segment.start))
