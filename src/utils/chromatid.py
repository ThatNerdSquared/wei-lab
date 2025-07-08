from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from functools import singledispatch
from typing import NamedTuple, NewType


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
    base_haplotype: Haplotype
    length: int
    crossovers: list[Crossover] = []


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
    # co_stack.reverse()
    chrom.crossovers += co_stack


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
def get_crossover_locations(chrom: Chromatid) -> Iterable[int]:
    return (x.loc for x in chrom.crossovers)


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
        if not end_haplotype and co.loc < endpoint:
            end_haplotype = co.haplotype
        if not start_haplotype and co.loc < start:
            start_haplotype = co.haplotype

    return (start_haplotype or chrom.base_haplotype) == (
        end_haplotype or chrom.base_haplotype
    )
