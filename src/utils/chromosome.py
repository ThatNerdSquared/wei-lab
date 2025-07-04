from collections.abc import Generator
from dataclasses import dataclass
from enum import Enum
from functools import singledispatch
from random import randint
from typing import NamedTuple, override
from uuid import UUID, uuid4


class Chromosome:
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


class Haplotype(Enum):
    RED = 0
    BLACK = 1

    def __bool__(self) -> bool:
        return self is Haplotype.BLACK


class Crossover(NamedTuple):
    loc: int
    haplotype: Haplotype
    """
    specifies the haplotype to the right of this crossover.
    """


@dataclass(slots=True, kw_only=True)
class Chromatid:
    base_haplotype: Haplotype
    crossovers: list[Crossover]
    length: int


@singledispatch
def insert_crossover(
    chrom: Chromatid, location: int, haplotype: Haplotype
) -> list[Crossover]:
    new_co = Crossover(location, haplotype)
    result: list[Crossover] = []

    while chrom.crossovers and location < chrom.crossovers[-1].loc:
        result.append(chrom.crossovers.pop())

    chrom.crossovers.append(new_co)
    return result


@singledispatch
def get_num_crossovers(chrom: Chromatid) -> int:
    return len(chrom.crossovers)


@singledispatch
def get_crossover_locations(chrom: Chromatid) -> Generator[int]:
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
