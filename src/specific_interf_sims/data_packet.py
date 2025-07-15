from __future__ import (
    annotations,
)  # needed to use DataPacket in its own definition

from dataclasses import dataclass
from typing import Callable, Generic, TypeVar

from matplotlib.axes import Axes

T_co = TypeVar('T_co', covariant=True)


@dataclass(slots=True)
class DataPacket(Generic[T_co]):
    data: dict[str, T_co]
    num_chroms: int
    num_bins: int
    curve_adder: Callable[[DataPacket[T_co], Axes], None]

    ylim: float
    xlabel: str
    ylabel: str
    fig_name: str
