import functools
import itertools
from collections.abc import Iterable, Iterator
from dataclasses import InitVar, dataclass, field
from typing import Callable, Generic, Protocol, TypeVar, override
from uuid import UUID, uuid4

import more_itertools

from croissant.chromosomes.chromosome_tetrad import (
    ChromosomeTetrad,
    CrossoverSisterSelector,
    PoolSisterSelector,
    haplotypeSwitchingCOsOnlySelector,
    make_default_chrom_w_sisters,
    selectAllSistersForPool,
    selectSingleSisterForPool,
    uniformlyRandomSisterSelector,
)
from croissant.chromosomes.core import ChromosomeLike
from croissant.chromosomes.legacy_chromosome import LegacyChromosome

type ChromosomeMaker[C: ChromosomeLike] = Callable[[int], C]

C_co = TypeVar('C_co', bound=ChromosomeLike, covariant=True)


@dataclass(kw_only=True, slots=True)
class ChromosomeSet(Generic[C_co]):
    id: UUID = field(init=False, default_factory=uuid4)
    label: str
    num_chroms: int
    chrom_length: int
    chroms: list[C_co] = field(init=False)
    chrom_maker: InitVar[ChromosomeMaker[C_co]]

    def __post_init__(self, chrom_maker: ChromosomeMaker[C_co]):
        self.chroms = [chrom_maker(self.chrom_length) for _ in range(self.num_chroms)]

    def __iter__(self) -> Iterator[C_co]:
        return iter(self.chroms)

    @override
    def __hash__(self) -> int:
        return hash(self.id)


def pool_iter(
    chrom_set: ChromosomeSet[ChromosomeLike],
) -> Iterator[ChromosomeLike]:
    if len(chrom_set.chroms) == 0:
        return iter(())
    if isinstance(chrom_set.chroms[0], Iterable):
        return itertools.chain.from_iterable(chrom_set.chroms)  # pyright: ignore[reportArgumentType, reportUnknownVariableType]
    return iter(chrom_set.chroms)


@functools.cache
def pool_size(chrom_set: ChromosomeSet[ChromosomeLike]) -> int:
    if len(chrom_set.chroms) == 0:
        return 0
    if isinstance(chrom_set.chroms[0], Iterable):
        return more_itertools.ilen(pool_iter(chrom_set))
    return chrom_set.num_chroms


class ChromsetMaker(Protocol):
    def __call__(
        self, *, label: str, num_chroms: int, chrom_length: int
    ) -> ChromosomeSet[ChromosomeLike]: ...


make_chromset_with_legacy_chromosome = functools.partial(
    ChromosomeSet[LegacyChromosome], chrom_maker=LegacyChromosome
)


def make_chromset_custom_sister_selectors(
    *,
    label: str,
    num_chroms: int,
    chrom_length: int,
    co_selector: CrossoverSisterSelector,
    pool_selector: PoolSisterSelector,
) -> ChromosomeSet[ChromosomeTetrad]:
    return ChromosomeSet(
        label=label,
        num_chroms=num_chroms,
        chrom_length=chrom_length,
        chrom_maker=lambda length: ChromosomeTetrad(
            length=length,
            selectSistersForCO=co_selector,
            selectSistersForPool=pool_selector,
        ),
    )


make_default_chromset = functools.partial(
    ChromosomeSet[ChromosomeTetrad], chrom_maker=make_default_chrom_w_sisters
)

make_chromset_all_sisters_in_pool = functools.partial(
    ChromosomeSet[ChromosomeTetrad],
    chrom_maker=lambda x: ChromosomeTetrad(
        length=x,
        selectSistersForCO=uniformlyRandomSisterSelector,
        selectSistersForPool=selectAllSistersForPool,
    ),
)

make_chromset_haplotype_switching_only = functools.partial(
    ChromosomeSet[ChromosomeTetrad],
    chrom_maker=lambda x: ChromosomeTetrad(
        length=x,
        selectSistersForCO=haplotypeSwitchingCOsOnlySelector,
        selectSistersForPool=selectSingleSisterForPool,
    ),
)
