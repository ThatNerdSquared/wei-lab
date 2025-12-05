from __future__ import (
    annotations,
)  # needed to use DataPacket in its own definition

from dataclasses import dataclass
from typing import Callable, Generic, TypeVar

from matplotlib.axes import Axes
from mpl_toolkits.mplot3d.axes3d import Axes3D

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


@dataclass(slots=True)
class ThreeDimDataPacket(Generic[T_co]):
    data: tuple[str, T_co]
    num_chroms: int
    num_bins: int
    surface_adder: Callable[[ThreeDimDataPacket[T_co], Axes3D], None]

    zlim: float
    xlabel: str
    ylabel: str
    zlabel: str
    fig_name: str
