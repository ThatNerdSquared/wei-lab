from __future__ import (
    annotations,
)  # needed to have Strand and ChromosomeLike in same file

import ast
import itertools
from collections.abc import Iterable, Iterator, Sequence
from enum import Enum
from typing import ClassVar, Literal, NamedTuple, NewType, Protocol, Self

import numpy as np

from croissant._numeric_utils import TaggedInterval


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


class ChromosomeLike(Protocol):
    length: int
    num_strands: ClassVar[int]

    def into_list(self, num_times: int) -> Sequence[Self]:
        """
        Returns a list of `num_times` items, where each element is a chromosome
        of the same type and settings. Runs in O(`num_times`) time.
        """
        ...

    def insert_crossover(self, location: int) -> None:
        """
        Inserts a CO at the given location, updating haplotypes if needed.

        Takes O(m) time, where m is the number of COs already inserted.

        Ideally, don't we want this to be O(1)?
        """
        ...

    def get_num_crossovers(self) -> int:
        """
        Returns the number of COs placed on the given chromosome.

        For multi-strand chromosomes, this does not deduplicate COs on
        different strands resulting from the same recombination event. (should
        it?) Runs in O(1) time.
        """
        ...

    def get_crossover_locations(self) -> list[int]:
        """
        Returns the locations of all COs placed on this chromosome.

        The returned locations are de-duplicated and sorted in ascending order.
        Runs in O(m log m) time, where m is the number of COs inserted on the
        chromosome.
        """
        ...

    def has_crossover_at(self, location: int) -> bool:
        """
        Returns True if this chromosome has a CO at the given location, and
        False otherwise, in O(1) time.
        """
        ...

    def has_visible_crossover(self, start: int = 0, end: int | None = None) -> bool:
        """
        Returns True if a CO is visible on this chromosome within the given
        region, and False otherwise. Runs in O(m) time, where m is the number
        of COs inserted on the chromosome.
        """
        ...

    def as_strands(self) -> Iterable[Strand]:
        """
        Returns an iterable of all strands within this chromosome. For
        chromosomes consisting of a single strand, this method returns an
        iterable containing self. Runs in O(1) time.
        """
        ...

    def into_haplotype_tbl(
        self, input_arr: np.ndarray[tuple[int], np.dtype[np.int64]]
    ): ...

    @classmethod
    def from_haplotype_tbl(
        cls, input_arr: np.ndarray[tuple[int], np.dtype[np.int64]]
    ) -> Self: ...


class Strand(ChromosomeLike, Protocol):
    base_haplotype: Haplotype

    def haplotype_at(self, location: int) -> Haplotype:
        """
        Returns the haplotype of the strand at the given location, in O(m)
        time, where m is the number of COs inserted on the strand.
        """
        ...

    def fill_from_segments(self, segments: Iterator[Segment]) -> None:
        """
        Given an iterator of segments, insert COs into the strand such that it
        has segments matching those in the iterator.

        Note that this MAY REPLACE ALL existing COs on this strand - the
        behaviour of calling `fill_from_segments` on a strand where COs have
        already been placed is undefined.
        """
        ...


def strandify(chroms: Sequence[ChromosomeLike]) -> Iterable[Strand]:
    """
    Returns an iterable over all strands in the provided chromosomes.

    To get the strands for a single chromosome, use its `as_strands()` method
    instead.
    """
    return itertools.chain.from_iterable((x.as_strands() for x in chroms))


def num_strands(chroms: Sequence[ChromosomeLike]) -> int:
    """
    Returns the number of Strands within the iterable produced by
    calling `strandify` on the given list of chromosomes, in
    constant time.
    """
    return len(chroms) * chroms[0].num_strands


type Segment = TaggedInterval[Haplotype]


def segments(strand: Strand) -> Iterator[Segment]:
    if strand.get_num_crossovers() == 0:
        yield TaggedInterval(start=0, end=strand.length - 1, data=strand.base_haplotype)
        return

    co_locs = strand.get_crossover_locations()
    yield TaggedInterval(start=0, end=co_locs[0] - 1, data=strand.base_haplotype)

    for curr_loc, next_loc in itertools.pairwise(co_locs):
        yield TaggedInterval(
            start=curr_loc, end=next_loc - 1, data=strand.haplotype_at(curr_loc)
        )

    yield TaggedInterval(
        start=co_locs[-1],
        end=strand.length - 1,
        data=strand.haplotype_at(co_locs[-1]),
    )


def segments_from_str(raw_segments: str) -> Iterator[Segment]:
    parsed_raw_segments: list[tuple[int, int, int]] = ast.literal_eval(raw_segments)  # pyright: ignore[reportAny]
    return (
        TaggedInterval(start=x[0], end=x[1], data=Haplotype(x[2]))
        for x in parsed_raw_segments
    )


def segments_serialisable(
    strand: Strand,
) -> Iterator[tuple[float, float, Literal[0, 1]]]:
    return ((x.start, x.end, x.data.value) for x in segments(strand))
