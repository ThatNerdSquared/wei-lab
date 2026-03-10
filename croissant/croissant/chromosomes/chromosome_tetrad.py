from __future__ import (
    annotations,
)  # needed to use ChromosomeSisterChromatids in its own definition

import itertools
import random
from dataclasses import dataclass, field
from typing import Callable, ClassVar

import numpy as np

from croissant.chromosomes.chromatid import (
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

type CrossoverSisterSelector = Callable[
    [ChromosomeTetrad, int], tuple[Chromatid, Chromatid]
]

type PoolSisterSelector = Callable[
    [ChromosomeTetrad], tuple[Chromatid, ...] | tuple[()]
]


@dataclass(slots=True, kw_only=True)
class ChromosomeTetrad:
    a1: Chromatid = field(init=False)
    a2: Chromatid = field(init=False)
    b1: Chromatid = field(init=False)
    b2: Chromatid = field(init=False)
    all_chromatids: tuple[Chromatid, Chromatid, Chromatid, Chromatid] = field(
        init=False
    )
    length: int
    selectSistersForCO: CrossoverSisterSelector
    selectSistersForPool: PoolSisterSelector
    num_strands: ClassVar[int] = 4

    def __post_init__(self):
        self.a1 = Chromatid(base_haplotype=Haplotype.BLUE, length=self.length)
        self.a2 = Chromatid(base_haplotype=Haplotype.BLUE, length=self.length)
        self.b1 = Chromatid(base_haplotype=Haplotype.PINK, length=self.length)
        self.b2 = Chromatid(base_haplotype=Haplotype.PINK, length=self.length)
        self.all_chromatids = (self.a1, self.a2, self.b1, self.b2)

    def __iter__(self):
        return self.selectSistersForPool(self).__iter__()

    def into_list[C: ChromosomeTetrad](self: C, num_times: int) -> list[C]:
        return [
            self.__class__(
                length=self.length,
                selectSistersForCO=self.selectSistersForCO,
                selectSistersForPool=self.selectSistersForPool,
            )
            for _ in range(num_times)
        ]

    def into_haplotype_tbl(self, input_arr: np.ndarray[tuple[int], np.dtype[np.int64]]):
        assert (4, self.length) == input_arr.shape

        for idx, chrom in enumerate(self.all_chromatids):
            chrom.into_haplotype_tbl(input_arr[idx : idx + 1])


def uniformlyRandomSisterSelector(
    chrom: ChromosomeTetrad, _: int
) -> tuple[Chromatid, Chromatid]:
    selected = random.sample(chrom.all_chromatids, k=2)
    return (selected[0], selected[1])


class SequentialSisterSelector:
    _counter: int

    def __init__(self, init_count: int | None = None) -> None:
        self._counter = init_count if init_count is not None else -1

    def __call__(self, chrom: ChromosomeTetrad, _: int) -> tuple[Chromatid, Chromatid]:
        self._counter += 1
        try:
            return chrom.all_chromatids[self._counter], chrom.all_chromatids[
                self._counter + 1
            ]
        except IndexError:
            return chrom.all_chromatids[self._counter], chrom.all_chromatids[0]


def haplotypeSwitchingCOsOnlySelector(
    chrom: ChromosomeTetrad, loc: int
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
    chrom: ChromosomeTetrad,
) -> tuple[Chromatid]:
    return (random.choice(chrom.all_chromatids),)


def selectAllSistersForPool(
    chrom: ChromosomeTetrad,
) -> tuple[Chromatid, Chromatid, Chromatid, Chromatid]:
    return chrom.all_chromatids


def selectMostRecombinantForPool(
    chrom: ChromosomeTetrad,
) -> tuple[Chromatid] | tuple[()]:
    if get_num_crossovers(chrom) == 0:
        return ()
    return (max(chrom.all_chromatids, key=get_num_crossovers),)


def selectOnlyRecombinantsForPool(
    chrom: ChromosomeTetrad,
) -> tuple[Chromatid] | tuple[()]:
    if get_num_crossovers(chrom) == 0:
        return ()
    selectable = list(filter(lambda x: get_num_crossovers(x) > 0, chrom.all_chromatids))
    return (random.choice(selectable),)


def probabilistic_selector(
    chrom: ChromosomeTetrad,
    rand: np.random.Generator,
    selector1: PoolSisterSelector,
    selector2: PoolSisterSelector,
    p1: float,
) -> tuple[Chromatid, ...] | tuple[()]:
    if rand.binomial(1, p1):
        return selector1(chrom)
    else:
        return selector2(chrom)


def make_default_chrom_w_sisters(length: int) -> ChromosomeTetrad:
    return ChromosomeTetrad(
        length=length,
        selectSistersForCO=uniformlyRandomSisterSelector,
        selectSistersForPool=selectSingleSisterForPool,
    )


@insert_crossover.register
def _(chrom: ChromosomeTetrad, location: int) -> None:
    involved_one, involved_two = chrom.selectSistersForCO(chrom, location)

    haplotype_moved_to_two = remove_co_stack_to_exchange(involved_one, location)
    haplotype_moved_to_one = remove_co_stack_to_exchange(involved_two, location)

    add_co_stack_from_exchange(involved_one, haplotype_moved_to_one)
    add_co_stack_from_exchange(involved_two, haplotype_moved_to_two)


@get_num_crossovers.register
def _(chrom: ChromosomeTetrad) -> int:
    return int(sum(map(get_num_crossovers, chrom.all_chromatids)) / 2)


@get_crossover_locations.register
def _(chrom: ChromosomeTetrad) -> list[int]:
    return sorted(
        set(
            itertools.chain.from_iterable(
                map(get_crossover_locations, chrom.all_chromatids)
            )
        )
    )


@has_crossover_at.register
def _(chrom: ChromosomeTetrad, location: int) -> bool:
    return any([has_crossover_at(x, location) for x in chrom.all_chromatids])


@has_visible_crossover.register
def _(chrom: ChromosomeTetrad, start: int = 0, end: int | None = None) -> bool:
    return any([has_visible_crossover(x, start, end) for x in chrom.all_chromatids])
