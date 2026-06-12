from typing import Callable, NamedTuple

import numpy as np
from matplotlib.axes import Axes
from src.utils.standard_funcs import haldane_input_cM, kosambi

from croissant.outputs.output import ListOfPoints

type _DataList = np.ndarray[tuple[int], np.dtype[np.float64]]


class MapFuncData(NamedTuple):
    haldane: _DataList
    morgan: _DataList
    kosambi: _DataList
    xvals: _DataList


def generate_canonical_map_func_data(up_to: float = 1.0) -> MapFuncData:
    x = np.linspace(0, up_to, int(1000 * up_to))
    return MapFuncData(haldane_input_cM(x), x, kosambi(x), x)


def generate_comparative_map_func_data(
    ref_data: ListOfPoints, comparator: Callable[[float, float], float]
) -> MapFuncData:
    # trim the 1st point since it's usually not well suited?
    raw_xs, raw_ys = zip(*ref_data[1:])
    xs = np.array(raw_xs)
    ys = np.array(raw_ys)
    return MapFuncData(
        comparator(haldane_input_cM(xs), ys),
        comparator(xs, ys),
        comparator(kosambi(xs), ys),
        xs,
    )


def plot_canonical_map_funcs(ax: Axes, data: MapFuncData):
    _ = ax.plot(data.xvals, data.haldane, label='haldane')
    _ = ax.plot(data.xvals, data.morgan, label='morgan')
    _ = ax.plot(data.xvals, data.kosambi, label='kosambi')
    _ = ax.legend()
