from __future__ import (
    annotations,
)  # needed to use ChromosomeSisterChromatids in its own definition

import itertools
import random
from dataclasses import dataclass, field
from typing import Callable

from src.utils.chromatid import (
    Chromatid,
    Haplotype,
    add_co_stack_from_exchange,
    get_crossover_locations,
    get_num_crossovers,
    has_crossover_at,
    has_visible_crossover,
    insert_crossover,
    remove_co_stack_to_exchange,
)


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
    selectSistersForCO: Callable[
        [ChromosomeSisterChromatids, int], tuple[Chromatid, Chromatid]
    ]
    selectSistersForPool: Callable[[ChromosomeSisterChromatids], tuple[Chromatid, ...]]

    def __post_init__(self):
        self.a1 = Chromatid(base_haplotype=Haplotype.BLUE, length=self.length)
        self.a2 = Chromatid(base_haplotype=Haplotype.BLUE, length=self.length)
        self.b1 = Chromatid(base_haplotype=Haplotype.PINK, length=self.length)
        self.b2 = Chromatid(base_haplotype=Haplotype.PINK, length=self.length)
        self.all_chromatids = (self.a1, self.a2, self.b1, self.b2)

    def __iter__(self):
        return self.selectSistersForPool(self).__iter__()


def uniformlyRandomSisterSelector(
    chrom: ChromosomeSisterChromatids, _: int
) -> tuple[Chromatid, Chromatid]:
    selected = random.sample(chrom.all_chromatids, k=2)
    return (selected[0], selected[1])


class SequentialSisterSelector:
    _counter: int

    def __init__(self) -> None:
        self._counter = -1

    def __call__(
        self, chrom: ChromosomeSisterChromatids, _: int
    ) -> tuple[Chromatid, Chromatid]:
        self._counter += 1
        try:
            return chrom.all_chromatids[self._counter], chrom.all_chromatids[
                self._counter + 1
            ]
        except IndexError:
            return chrom.all_chromatids[self._counter], chrom.all_chromatids[0]


def haplotypeSwitchingCOsOnlySelector(
    chrom: ChromosomeSisterChromatids, loc: int
) -> tuple[Chromatid, Chromatid]:
    blue_haplotype_at_loc: list[Chromatid] = []
    pink_haplotype_at_loc: list[Chromatid] = []

    for chromatid in chrom.all_chromatids:
        haplotype_at_loc = chromatid.base_haplotype

        for co in reversed(chromatid.crossovers):
            if co.loc > loc:
                continue
            haplotype_at_loc = co.haplotype
            break

        match haplotype_at_loc:
            case Haplotype.BLUE:
                blue_haplotype_at_loc.append(chromatid)
            case Haplotype.PINK:
                pink_haplotype_at_loc.append(chromatid)

    assert len(blue_haplotype_at_loc) != 0
    assert len(pink_haplotype_at_loc) != 0

    return random.choice(blue_haplotype_at_loc), random.choice(pink_haplotype_at_loc)


def selectSingleSisterForPool(
    chrom: ChromosomeSisterChromatids,
) -> tuple[Chromatid]:
    return (random.choice(chrom.all_chromatids),)


def selectAllSistersForPool(
    chrom: ChromosomeSisterChromatids,
) -> tuple[Chromatid, Chromatid, Chromatid, Chromatid]:
    return chrom.all_chromatids


def make_default_chrom_w_sisters(length: int) -> ChromosomeSisterChromatids:
    return ChromosomeSisterChromatids(
        length=length,
        selectSistersForCO=uniformlyRandomSisterSelector,
        selectSistersForPool=selectSingleSisterForPool,
    )


@insert_crossover.register
def _(chrom: ChromosomeSisterChromatids, location: int) -> None:
    involved_one, involved_two = chrom.selectSistersForCO(chrom, location)

    haplotype_moved_to_two = remove_co_stack_to_exchange(involved_one, location)
    haplotype_moved_to_one = remove_co_stack_to_exchange(involved_two, location)

    add_co_stack_from_exchange(involved_one, haplotype_moved_to_one)
    add_co_stack_from_exchange(involved_two, haplotype_moved_to_two)


@get_num_crossovers.register
def _(chrom: ChromosomeSisterChromatids) -> int:
    return int(sum(map(get_num_crossovers, chrom.all_chromatids)) / 2)


@get_crossover_locations.register
def _(chrom: ChromosomeSisterChromatids) -> list[int]:
    return sorted(
        set(
            itertools.chain.from_iterable(
                map(get_crossover_locations, chrom.all_chromatids)
            )
        )
    )


@has_crossover_at.register
def _(chrom: ChromosomeSisterChromatids, location: int) -> bool:
    return any([has_crossover_at(x, location) for x in chrom.all_chromatids])


@has_visible_crossover.register
def _(
    chrom: ChromosomeSisterChromatids, start: int = 0, end: int | None = None
) -> bool:
    return any([has_visible_crossover(x, start, end) for x in chrom.all_chromatids])
