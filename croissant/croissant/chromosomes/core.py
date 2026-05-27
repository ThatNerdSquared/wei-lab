import itertools
from collections.abc import Iterable, Sequence
from enum import Enum
from typing import ClassVar, NamedTuple, NewType, Protocol, Self

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

    def haplotype_at(self, location: int) -> Haplotype: ...


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
