import itertools
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
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


@dataclass(slots=True, kw_only=True)
class Chromatid:
    base_haplotype: Haplotype
    length: int
    crossovers: list[Crossover] = []


@dataclass(slots=True, kw_only=True)
class ChromosomeSisterChromatids:
    a1: Chromatid = field(init=False)
    a2: Chromatid = field(init=False)
    b1: Chromatid = field(init=False)
    b2: Chromatid = field(init=False)
    all_chromatids: tuple[Chromatid, Chromatid, Chromatid, Chromatid] = field(
        init=False
    )
    length: int
    selectSistersForCO: Callable[[], tuple[Chromatid, Chromatid]]
    selectSistersForPool: Callable[[], tuple[Chromatid, ...]]

    def __post_init__(self):
        self.a1 = Chromatid(base_haplotype=Haplotype.BLUE, length=self.length)
        self.a2 = Chromatid(base_haplotype=Haplotype.BLUE, length=self.length)
        self.b1 = Chromatid(base_haplotype=Haplotype.PINK, length=self.length)
        self.b1 = Chromatid(base_haplotype=Haplotype.PINK, length=self.length)
        self.all_chromatids = (self.a1, self.a2, self.b1, self.b2)


def remove_haplotypes_for_exchange(chrom: Chromatid, location: int) -> list[Crossover]:
    result: list[Crossover] = []

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


def insert_crossover(chrom: ChromosomeSisterChromatids, location: int) -> None:
    involved_one, involved_two = chrom.selectSistersForCO()

    haplotype_moved_to_two = remove_haplotypes_for_exchange(involved_one, location)
    # haplotype_moved_to_two.reverse()
    haplotype_moved_to_one = remove_haplotypes_for_exchange(involved_two, location)
    # haplotype_moved_to_one.reverse()

    involved_one.crossovers += haplotype_moved_to_one
    involved_two.crossovers += haplotype_moved_to_two


@singledispatch
def get_num_crossovers(chrom: Chromatid) -> int:
    return len(chrom.crossovers)


@get_num_crossovers.register
def _(chrom: ChromosomeSisterChromatids) -> int:
    return sum(map(get_num_crossovers, chrom.all_chromatids))


@singledispatch
def get_crossover_locations(chrom: Chromatid) -> Iterable[int]:
    return (x.loc for x in chrom.crossovers)


@get_crossover_locations.register
def _(chrom: ChromosomeSisterChromatids) -> Iterable[int]:
    return itertools.chain.from_iterable(
        map(get_crossover_locations, chrom.all_chromatids)
    )


@singledispatch
def has_crossover_at(chrom: Chromatid, location: int) -> bool:
    return any(x.loc == location for x in chrom.crossovers)


@has_crossover_at.register
def _(chrom: ChromosomeSisterChromatids, location: int) -> bool:
    return any([has_crossover_at(x, location) for x in chrom.all_chromatids])


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


@has_visible_crossover.register
def _(
    chrom: ChromosomeSisterChromatids, start: int = 0, end: int | None = None
) -> bool:
    return any([has_visible_crossover(x, start, end) for x in chrom.all_chromatids])
