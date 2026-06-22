from __future__ import (
    annotations,
)  # needed to use Interval in its own definition

from dataclasses import dataclass
from math import inf


@dataclass(slots=True, frozen=True, kw_only=True)
class Interval:
    start: float = inf
    end: float = -inf
    include_start: bool = True
    include_end: bool = True

    @classmethod
    def from_abs_tolerance(
        cls,
        centre: float,
        *,
        tolerance: float,
        include_start: bool = True,
        include_end: bool = True,
    ) -> Interval:
        return Interval(
            start=centre - tolerance,
            end=centre + tolerance,
            include_start=include_start,
            include_end=include_end,
        )

    def __gt__(self, other: int | float) -> bool:
        return self.start > other if self.include_end else self.start >= other

    def __lt__(self, other: int | float) -> bool:
        return self.end < other if self.include_end else self.end <= other

    def __contains__(self, compare_to: int | float) -> bool:
        return not (self > compare_to or self < compare_to)

    def __ge__(self, other: int | float) -> bool:
        return self > other or other in self

    def __le__(self, other: int | float) -> bool:
        return self < other or other in self
