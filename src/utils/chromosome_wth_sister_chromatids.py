from __future__ import (
    annotations,
)  # needed to use ChromosomeSisterChromatids in its own definition

import itertools
import random
from collections.abc import Iterable
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
        [ChromosomeSisterChromatids], tuple[Chromatid, Chromatid]
    ]
    selectSistersForPool: Callable[[], tuple[Chromatid, ...]]

    def __post_init__(self):
        self.a1 = Chromatid(base_haplotype=Haplotype.BLUE, length=self.length)
        self.a2 = Chromatid(base_haplotype=Haplotype.BLUE, length=self.length)
        self.b1 = Chromatid(base_haplotype=Haplotype.PINK, length=self.length)
        self.b1 = Chromatid(base_haplotype=Haplotype.PINK, length=self.length)
        self.all_chromatids = (self.a1, self.a2, self.b1, self.b2)


def uniformlyRandomSisterSelector(
    chrom: ChromosomeSisterChromatids,
) -> tuple[Chromatid, Chromatid]:
    selected = random.sample(chrom.all_chromatids, k=2)
    return (selected[0], selected[1])


def selectSingleSisterForPool(
    chrom: ChromosomeSisterChromatids,
) -> tuple[Chromatid]:
    return (random.choice(chrom.all_chromatids),)


@insert_crossover.register
def _(chrom: ChromosomeSisterChromatids, location: int) -> None:
    involved_one, involved_two = chrom.selectSistersForCO(chrom)

    haplotype_moved_to_two = remove_co_stack_to_exchange(involved_one, location)
    haplotype_moved_to_one = remove_co_stack_to_exchange(involved_two, location)

    add_co_stack_from_exchange(involved_one, haplotype_moved_to_one)
    add_co_stack_from_exchange(involved_two, haplotype_moved_to_two)


@get_num_crossovers.register
def _(chrom: ChromosomeSisterChromatids) -> int:
    return sum(map(get_num_crossovers, chrom.all_chromatids))


@get_crossover_locations.register
def _(chrom: ChromosomeSisterChromatids) -> Iterable[int]:
    return itertools.chain.from_iterable(
        map(get_crossover_locations, chrom.all_chromatids)
    )


@has_crossover_at.register
def _(chrom: ChromosomeSisterChromatids, location: int) -> bool:
    return any([has_crossover_at(x, location) for x in chrom.all_chromatids])


@has_visible_crossover.register
def _(
    chrom: ChromosomeSisterChromatids, start: int = 0, end: int | None = None
) -> bool:
    return any([has_visible_crossover(x, start, end) for x in chrom.all_chromatids])
