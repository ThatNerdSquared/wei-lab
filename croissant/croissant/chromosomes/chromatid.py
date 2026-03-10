from dataclasses import dataclass, field
from enum import Enum
from functools import singledispatch
from itertools import pairwise
from typing import ClassVar, NamedTuple, NewType
from uuid import UUID, uuid4

import numpy as np


class Haplotype(Enum):
    BLUE = 0
    PINK = 1

    def __bool__(self) -> bool:
        return self is Haplotype.PINK


class Crossover(NamedTuple):
    loc: int
    haplotype: Haplotype
    """
    specifies the haplotype to the right of this crossover.
    """


CrossoverStack = NewType('CrossoverStack', list[Crossover])


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

    def into_list[C: Chromatid](self: C, num_times: int) -> list[C]:
        return [
            self.__class__(base_haplotype=self.base_haplotype, length=self.length)
            for _ in range(num_times)
        ]

    def into_haplotype_tbl(self, input_arr: np.ndarray[tuple[int], np.dtype[np.int64]]):
        # can use arr.reshape(1, -1, copy=False) to turn 1d ndarray into 2d, if
        # necessary
        assert (1, self.length) == input_arr.shape

        return convert_chromatid_to_haplotype_tbl(self, input_arr[0])


def make_default_chromatid(length: int):
    return Chromatid(base_haplotype=Haplotype.BLUE, length=length)


def remove_co_stack_to_exchange(chrom: Chromatid, location: int) -> CrossoverStack:
    assert location <= chrom.length
    result: CrossoverStack = CrossoverStack([])

    while chrom.crossovers and location < chrom.crossovers[-1].loc:
        result.append(chrom.crossovers.pop())

    result.append(
        Crossover(
            location,
            chrom.crossovers[-1].haplotype
            if chrom.crossovers
            else chrom.base_haplotype,
        )
    )
    return result


def add_co_stack_from_exchange(chrom: Chromatid, co_stack: CrossoverStack) -> None:
    co_stack.reverse()
    chrom.crossovers += co_stack


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


@singledispatch
def insert_crossover(chrom: Chromatid, location: int) -> None:
    assert location <= chrom.length
    old_co_stack = remove_co_stack_to_exchange(chrom, location)
    co_stack_flipped_haplotypes = invert_haplotypes_in_co_stack(old_co_stack)
    add_co_stack_from_exchange(chrom, co_stack_flipped_haplotypes)


@singledispatch
def get_num_crossovers(chrom: Chromatid) -> int:
    return len(chrom.crossovers)


@singledispatch
def get_crossover_locations(chrom: Chromatid) -> list[int]:
    return [x.loc for x in chrom.crossovers]


@singledispatch
def has_crossover_at(chrom: Chromatid, location: int) -> bool:
    return any(x.loc == location for x in chrom.crossovers)


@singledispatch
def has_visible_crossover(
    chrom: Chromatid, start: int = 0, end: int | None = None
) -> bool:
    if not chrom.crossovers:
        return False

    endpoint: int = end if end is not None else chrom.length
    start_haplotype = None
    end_haplotype = None

    for co in reversed(chrom.crossovers):
        if end_haplotype is None and co.loc <= endpoint:
            end_haplotype = co.haplotype
        if start_haplotype is None and co.loc <= start:
            start_haplotype = co.haplotype

    return (
        start_haplotype if start_haplotype is not None else chrom.base_haplotype
    ) != (end_haplotype if end_haplotype is not None else chrom.base_haplotype)
