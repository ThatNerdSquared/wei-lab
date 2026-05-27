from collections.abc import Sequence
from typing import Callable, ClassVar, Protocol, cast

from matplotlib.axes import Axes
from matplotlib.figure import Figure

from croissant.chromosomes import ChromosomeSet
from croissant.chromosomes.core import ChromosomeLike

type ListOfPoints = list[tuple[float, float]]
type MultipleCurves = dict[str, ListOfPoints]


class Output[T, R](Protocol):
    fig_name: ClassVar[str]
    xlabel: ClassVar[str]
    ylabel: ClassVar[str]
    data: T
    output_file: R

    def build(self, chrom_sets: dict[str, Sequence[ChromosomeSet[ChromosomeLike]]]): ...


def label_figure[T, R](
    output: Output[T, R],
    figure_to_label: Figure | None = None,
    axes_to_label: Axes | None = None,
):
    fig = (
        figure_to_label
        if figure_to_label is not None
        else cast(Figure, output.output_file)
    )
    ax = axes_to_label if axes_to_label is not None else fig.get_axes()[0]
    _ = ax.set_xlabel(output.xlabel)
    _ = ax.set_ylabel(output.ylabel)
    _ = ax.legend()
    _ = fig.suptitle(output.fig_name)


def smart_dictify[K, V](
    input_pairs: Sequence[tuple[K, V]],
    resolver: Callable[[list[V]], V] | None = None,
) -> dict[K, V]:
    if not resolver:
        return dict(input_pairs)
    aggregate: dict[K, list[V]] = {}
    for key, val in input_pairs:
        aggregate.setdefault(key, []).append(val)
    res: dict[K, V] = {}
    for key, val in aggregate.items():
        res[key] = resolver(val)
    return res
