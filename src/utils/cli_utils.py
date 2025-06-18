import itertools
from typing import Any
import numpy as np
import matplotlib.pyplot as plt

from contextlib import contextmanager
from datetime import datetime

from src.utils.chromosome_set import ChromosomeSet


@contextmanager
def elapsed_time_printer(*,
    op_name: str,
    op_number: tuple[int, int] | None = None,
    chrom_set: ChromosomeSet | None = None
):
    start_time = datetime.now()
    try:
        yield
    finally:
        elapsed = (datetime.now() - start_time).total_seconds()
        print(
            'op complete',
            f'[{op_number[0]}/{op_number[1]}]:' if op_number else ':'
        )
        print(
            f'{elapsed}s elapsed',
            f'for {op_name}' if op_name else '',
            f'on {chrom_set.num_chroms} chroms @ {chrom_set.chrom_length} bins/chrom.' if chrom_set else '.'
        )

def calculate_max_y(args: list) -> float:
    flat_args = list(itertools.chain.from_iterable(args))
    return np.max(flat_args) * 1.1


@contextmanager
def quickplot(
    x_label: str = '',
    y_label: str = '',
    y_lim: float | list[Any] = 1.0,
    fig_name: str = ''
):
    fig = plt.figure(figsize=(9.0, 7.0))
    ax = fig.add_subplot()
    print(f'plotting: {fig_name}')
    try:
        yield ax
    finally:
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.legend()
        ax.set_ylim(
            0,
            y_lim if isinstance(y_lim, float) else calculate_max_y(y_lim)
        )
        fig.savefig(f'out--{fig_name}.png')
        print(f'plot {fig_name} saved.')


def plot_name_formatter(
    plot_name: str,
    num_chroms: int,
    num_bins: int
):
    num_chrom_in_ks = str(num_chroms / 1000).replace('.', '_')
    num_bins_in_ks = str(num_bins / 1000).replace('.', '_')
    return f'{plot_name}--{num_chrom_in_ks}k__{num_bins_in_ks}k'
