from dataclasses import dataclass
from typing import Protocol

import numpy as np
import numpy.typing as npt

from src.specific_interf_sims.data_transformers import DataTransformer
from src.utils.chromosome_set import ChromosomeLike, ChromosomeSet
from src.utils.cli_utils import calculate_max_y


@dataclass(slots=True)
class DataPacket[T](Protocol):
    data: dict[str, T]
    num_chroms: int
    num_bins: int
    ylim: float
    xlabel: str
    ylabel: str
    fig_name: str


@dataclass(slots=True)
class RecombCurveDataPacket:
    data: dict[str, npt.NDArray[np.float64]]
    num_chroms: int
    num_bins: int

    ylim: float
    xlabel: str = 'bin'
    ylabel: str = '% of chroms with CO'
    fig_name: str = 'recomb curve'


def collect_recomb_curve_data[T, C: ChromosomeLike](
    transformer: DataTransformer[npt.NDArray[np.float64], C],
    chrom_sets: list[ChromosomeSet[C]],
) -> RecombCurveDataPacket:
    transformed_data = list(map(transformer, chrom_sets))
    return RecombCurveDataPacket(
        data=dict(transformed_data),
        num_chroms=chrom_sets[0].num_chroms,
        num_bins=chrom_sets[0].chrom_length,
        ylim=calculate_max_y([x[1] for x in transformed_data]),
    )


@dataclass(slots=True)
class cMRFCurveDataPacket:
    data: dict[str, tuple[list[float], list[float]]]
    num_chroms: int
    num_bins: int

    ylim: float = 1.0
    xlabel: str = 'morgans (M)'
    ylabel: str = 'recombinant frequency (rf)'
    fig_name: str = 'cM_rf_curve'


def collect_cM_rf_curve_data[T, C: ChromosomeLike](
    transformer: DataTransformer[tuple[list[float], list[float]], C],
    chrom_sets: list[ChromosomeSet[C]],
) -> cMRFCurveDataPacket:
    transformed_data = list(map(transformer, chrom_sets))
    return cMRFCurveDataPacket(
        data=dict(transformed_data),
        num_chroms=chrom_sets[0].num_chroms,
        num_bins=chrom_sets[0].chrom_length,
        ylim=calculate_max_y([x[1] for x in transformed_data]),
    )
