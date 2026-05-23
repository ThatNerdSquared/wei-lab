from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from itertools import pairwise
from typing import ClassVar, Self
from uuid import UUID, uuid4

import numpy as np

from croissant.chromosomes.core import Crossover, CrossoverStack, Haplotype, Strand


def invert_haplotypes_in_co_stack(co_stack: CrossoverStack) -> CrossoverStack:
    return CrossoverStack(
        [Crossover(x.loc, Haplotype(not x.haplotype)) for x in co_stack]
    )


@dataclass(slots=True, kw_only=True)
class Chromatid:
    id: UUID = field(init=False)
    base_haplotype: Haplotype
    length: int
    crossovers: list[Crossover] = field(default_factory=list)
    num_strands: ClassVar[int] = 1

    def __post_init__(self):
        self.id = uuid4()

    def into_list[C: Chromatid](self: C, num_times: int) -> Sequence[C]:
        return [
            self.__class__(base_haplotype=self.base_haplotype, length=self.length)
            for _ in range(num_times)
        ]

    def insert_crossover(self, location: int) -> None:
        assert location <= self.length
        old_co_stack = self.remove_co_stack_to_exchange(location)
        co_stack_flipped_haplotypes = invert_haplotypes_in_co_stack(old_co_stack)
        self.add_co_stack_from_exchange(co_stack_flipped_haplotypes)

    def get_num_crossovers(self) -> int:
        return len(self.crossovers)

    def get_crossover_locations(self) -> list[int]:
        return [x.loc for x in self.crossovers]

    def haplotype_at(self, location: int) -> Haplotype:
        res = None

        for co in reversed(self.crossovers):
            if co.loc <= location:
                res = co.haplotype
                break

        return self.base_haplotype if res is None else res

    def has_crossover_at(self, location: int) -> bool:
        return any(x.loc == location for x in self.crossovers)

    def has_visible_crossover(self, start: int = 0, end: int | None = None) -> bool:
        if not self.crossovers:
            return False

        endpoint: int = end if end is not None else self.length
        start_haplotype = None
        end_haplotype = None

        for co in reversed(self.crossovers):
            if end_haplotype is None and co.loc <= endpoint:
                end_haplotype = co.haplotype
            if start_haplotype is None and co.loc <= start:
                start_haplotype = co.haplotype

        return (
            start_haplotype if start_haplotype is not None else self.base_haplotype
        ) != (end_haplotype if end_haplotype is not None else self.base_haplotype)

    def as_strands(self) -> Iterable[Strand]:
        yield self

    def into_haplotype_tbl(self, input_arr: np.ndarray[tuple[int], np.dtype[np.int64]]):
        # can use arr.reshape(1, -1, copy=False) to turn 1d ndarray into 2d, if
        # necessary
        assert (1, self.length) == input_arr.shape

        return convert_chromatid_to_haplotype_tbl(self, input_arr[0])

    @classmethod
    def from_haplotype_tbl(
        cls, input_arr: np.ndarray[tuple[int], np.dtype[np.int64]]
    ) -> Self:
        print(input_arr)
        raise NotImplementedError()

    def remove_co_stack_to_exchange(self, location: int) -> CrossoverStack:
        assert location <= self.length
        result: CrossoverStack = CrossoverStack([])

        while self.crossovers and location < self.crossovers[-1].loc:
            result.append(self.crossovers.pop())

        result.append(
            Crossover(
                location,
                self.crossovers[-1].haplotype
                if self.crossovers
                else self.base_haplotype,
            )
        )
        return result

    def add_co_stack_from_exchange(self, co_stack: CrossoverStack) -> None:
        co_stack.reverse()
        self.crossovers += co_stack


def convert_chromatid_to_haplotype_tbl(
    chrom: Chromatid, input_array: np.ndarray[tuple[int], np.dtype[np.int64]]
) -> None:
    assert chrom.length == input_array.size

    if not chrom.crossovers:
        input_array[0 : chrom.length] = chrom.base_haplotype.value
        return

    input_array[0 : chrom.crossovers[0].loc] = chrom.base_haplotype.value
    for curr_co, next_co in pairwise(chrom.crossovers):
        input_array[curr_co.loc : next_co.loc] = curr_co.haplotype.value
    input_array[chrom.crossovers[-1].loc :] = chrom.crossovers[-1].haplotype.value
