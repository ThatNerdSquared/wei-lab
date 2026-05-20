from collections.abc import Iterable
from functools import singledispatch

from croissant.chromosomes import (
    Chromatid,
    ChromosomeLike,
    ChromosomeTetrad,
    LegacyChromosome,
)


@singledispatch
def per_strand(chrom: Chromatid) -> Iterable[ChromosomeLike]:
    yield chrom


@per_strand.register
def _(chrom: LegacyChromosome) -> Iterable[ChromosomeLike]:
    yield chrom


@per_strand.register
def _(chrom: ChromosomeTetrad) -> Iterable[ChromosomeLike]:
    return iter(chrom.all_chromatids)
