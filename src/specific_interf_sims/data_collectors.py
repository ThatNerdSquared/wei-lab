import numpy as np
import numpy.typing as npt

from src.specific_interf_sims.data_packet import DataPacket
from src.specific_interf_sims.data_transformers import DataTransformer
from src.specific_interf_sims.plotter import (
    histogram_curve_adder,
    histogram_unpacked_data_curve_adder,
    line_curve_adder,
    line_unpacked_data_curve_adder,
)
from src.utils.chromosome_set import ChromosomeLike, ChromosomeSet
from src.utils.cli_utils import calculate_max_y


def collect_recomb_curve_data[C: ChromosomeLike](
    transformer: DataTransformer[npt.NDArray[np.float64], C],
    chrom_sets: list[ChromosomeSet[C]],
) -> DataPacket[npt.NDArray[np.float64]]:
    transformed_data = list(map(transformer, chrom_sets))
    return DataPacket(
        data=dict(transformed_data),
        num_chroms=chrom_sets[0].num_chroms,
        num_bins=chrom_sets[0].chrom_length,
        curve_adder=line_curve_adder,
        ylim=calculate_max_y([x[1] for x in transformed_data]),
        xlabel='bin',
        ylabel='% of chroms with CO',
        fig_name='recomb curve',
    )


def collect_co_per_chrom_dist_data[C: ChromosomeLike](
    transformer: DataTransformer[npt.NDArray[np.float64], C],
    chrom_sets: list[ChromosomeSet[C]],
) -> DataPacket[npt.NDArray[np.float64]]:
    transformed_data = list(map(transformer, chrom_sets))
    return DataPacket(
        data=dict(transformed_data),
        num_chroms=chrom_sets[0].num_chroms,
        num_bins=chrom_sets[0].chrom_length,
        curve_adder=histogram_curve_adder,
        ylim=calculate_max_y([x[1] for x in transformed_data]),
        xlabel='# of COs',
        ylabel='% of chroms',
        fig_name='co_per_chrom',
    )


def collect_interevent_dist_data[C: ChromosomeLike](
    transformer: DataTransformer[
        tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]], C
    ],
    chrom_sets: list[ChromosomeSet[C]],
) -> DataPacket[tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]]:
    transformed_data = list(map(transformer, chrom_sets))
    return DataPacket(
        data=dict(transformed_data),
        num_chroms=chrom_sets[0].num_chroms,
        num_bins=chrom_sets[0].chrom_length,
        curve_adder=histogram_unpacked_data_curve_adder,
        ylim=calculate_max_y([x[1][0] for x in transformed_data]),
        xlabel='inter-CO distance',
        ylabel='% of chroms',
        fig_name='interevent_dist',
    )


def collect_cM_rf_curve_data[C: ChromosomeLike](
    transformer: DataTransformer[tuple[list[float], list[float]], C],
    chrom_sets: list[ChromosomeSet[C]],
) -> DataPacket[tuple[list[float], list[float]]]:
    transformed_data = list(map(transformer, chrom_sets))
    return DataPacket(
        data=dict(transformed_data),
        num_chroms=chrom_sets[0].num_chroms,
        num_bins=chrom_sets[0].chrom_length,
        curve_adder=line_unpacked_data_curve_adder,
        ylim=1.0,
        xlabel='morgans (M)',
        ylabel='recombinant frequency (rf)',
        fig_name='cM_rf_curve',
    )


def collect_single_refregion_coc_curve_data[C: ChromosomeLike](
    transformer: DataTransformer[np.ndarray[tuple[int], np.dtype[np.float64]], C],
    chrom_sets: list[ChromosomeSet[C]],
) -> DataPacket[np.ndarray[tuple[int], np.dtype[np.float64]]]:
    transformed_data = list(map(transformer, chrom_sets))
    return DataPacket(
        data=dict(transformed_data),
        num_chroms=chrom_sets[0].num_chroms,
        num_bins=chrom_sets[0].chrom_length,
        curve_adder=line_curve_adder,
        ylim=calculate_max_y([x[1] for x in transformed_data]),
        xlabel='chromosome location',
        ylabel='CoC',
        fig_name='single_refregion_coc_curve',
    )
