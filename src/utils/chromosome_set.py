import functools
import itertools
from collections.abc import Iterable, Iterator
from dataclasses import InitVar, dataclass, field
from typing import Callable, Protocol, override
from uuid import UUID, uuid4

import more_itertools
import numpy as np
import numpy.typing as npt

from src.specific_interf_sims.interf_schemes import BinArray
from src.utils.chromatid import (
    Chromatid,
    get_crossover_locations,
    get_num_crossovers,
    has_crossover_at,
    has_visible_crossover,
)
from src.utils.chromosome import Chromosome
from src.utils.chromosome_with_sister_chromatids import (
    ChromosomeSisterChromatids,
    make_default_chrom_w_sisters,
    selectAllSistersForPool,
    uniformlyRandomSisterSelector,
)

type ChromosomeLike = Chromatid | ChromosomeSisterChromatids | Chromosome
type ChromosomeMaker[C: ChromosomeLike] = Callable[[int], C]


@dataclass(kw_only=True, slots=True)
class ChromosomeSet[C: ChromosomeLike]:
    id: UUID = field(init=False, default_factory=uuid4)
    label: str
    num_chroms: int
    chrom_length: int
    chroms: list[C] = field(init=False)
    chrom_maker: InitVar[ChromosomeMaker[C]]

    def __post_init__(self, chrom_maker: ChromosomeMaker[C]):
        self.chroms = [chrom_maker(self.chrom_length) for _ in range(self.num_chroms)]

    def __iter__(self) -> Iterator[C]:
        return iter(self.chroms)

    @override
    def __hash__(self) -> int:
        return hash(self.id)


def pool_iter[C: ChromosomeLike](
    chrom_set: ChromosomeSet[C],
) -> Iterator[ChromosomeLike]:
    if len(chrom_set.chroms) == 0:
        return iter(())
    if isinstance(chrom_set.chroms[0], Iterable):
        return itertools.chain.from_iterable(chrom_set.chroms)  # pyright: ignore[reportArgumentType, reportUnknownVariableType]
    return iter(chrom_set.chroms)


@functools.cache
def pool_size[C: ChromosomeLike](chrom_set: ChromosomeSet[C]) -> int:
    if len(chrom_set.chroms) == 0:
        return 0
    if isinstance(chrom_set.chroms[0], Iterable):
        return more_itertools.ilen(pool_iter(chrom_set))
    return chrom_set.num_chroms


class ChromsetMaker[C: ChromosomeLike](Protocol):
    def __call__(
        self, *, label: str, num_chroms: int, chrom_length: int
    ) -> ChromosomeSet[C]: ...


make_chromset_with_legacy_chromosome = functools.partial(
    ChromosomeSet[Chromosome], chrom_maker=Chromosome
)

make_default_chromset = functools.partial(
    ChromosomeSet[ChromosomeSisterChromatids], chrom_maker=make_default_chrom_w_sisters
)

make_chromset_all_sisters_in_pool = functools.partial(
    ChromosomeSet[ChromosomeSisterChromatids],
    chrom_maker=lambda x: ChromosomeSisterChromatids(
        length=x,
        selectSistersForCO=uniformlyRandomSisterSelector,
        selectSistersForPool=selectAllSistersForPool,
    ),
)


def gather_recomb_curve_data[C: ChromosomeLike](
    chrom_set: ChromosomeSet[C], proportional: bool = True
) -> npt.NDArray[np.float64]:
    recomb_curve_data = np.zeros(chrom_set.chrom_length)

    for loc in range(chrom_set.chrom_length):
        recomb_curve_data[loc] = sum(
            [1 if has_crossover_at(x, loc) else 0 for x in pool_iter(chrom_set)]
        )

    if proportional:
        recomb_curve_data = recomb_curve_data / chrom_set.num_chroms

    return recomb_curve_data


def gather_co_per_chrom_dist_data[C: ChromosomeLike](
    chrom_set: ChromosomeSet[C], proportional: bool = True
) -> npt.NDArray[np.float64]:
    data_dict: dict[int, float] = {}

    for chrom in pool_iter(chrom_set):
        try:
            data_dict[get_num_crossovers(chrom)] += 1.0
        except KeyError:
            data_dict[get_num_crossovers(chrom)] = 1.0

    dist_data = np.array([data_dict.get(x, 0) for x in range(max(data_dict))])

    if proportional:
        dist_data = dist_data / chrom_set.num_chroms

    return dist_data


def gather_interevent_dist_data[C: ChromosomeLike](
    chrom_set: ChromosomeSet[C], proportional: bool = True
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    dist_data: list[float] = []

    for chrom in pool_iter(chrom_set):
        co_locs = get_crossover_locations(chrom)

        for loc_idx in range(1, len(co_locs)):
            interevent_dist = co_locs[loc_idx] - co_locs[loc_idx - 1] - 1
            dist_data.append(interevent_dist)

    totals, groups = np.histogram(dist_data, bins='auto')

    if proportional:
        totals = totals / chrom_set.num_chroms

    return (totals, groups)


def gather_cM_rf_curve_data[C: ChromosomeLike](
    chrom_set: ChromosomeSet[C],
) -> tuple[list[float], list[float]]:
    dataset_cM: list[float] = []
    dataset_rf: list[float] = []

    total_cos_so_far = 0
    for x in range(chrom_set.chrom_length):
        visible_COs = 0
        for chrom in pool_iter(chrom_set):
            if has_crossover_at(chrom, x):
                total_cos_so_far += 1
            if has_visible_crossover(chrom, 0, x):
                visible_COs += 1

        dataset_cM.append(total_cos_so_far / chrom_set.num_chroms)
        dataset_rf.append(visible_COs / chrom_set.num_chroms)

    return (dataset_cM, dataset_rf)


def gather_preinterf_cM_rf_curve_data[C: ChromosomeLike](
    chrom_set: ChromosomeSet[C], pre_interf_dist: BinArray
) -> tuple[list[float], list[float]]:
    dataset_cM: list[float] = []
    dataset_rf: list[float] = []

    cM_so_far = 0
    for x in range(chrom_set.chrom_length):
        cM_so_far += pre_interf_dist[x]
        visible_COs = 0
        for chrom in pool_iter(chrom_set):
            if has_visible_crossover(chrom, end=x):
                visible_COs += 1

        dataset_cM.append(cM_so_far)
        dataset_rf.append(visible_COs / chrom_set.num_chroms)

    return (dataset_cM, dataset_rf)
