import itertools
from collections.abc import Callable, Iterable
from contextlib import contextmanager
from datetime import datetime
from typing import Any

import numpy as np
from croissant.chromosomes import ChromosomeSet
from croissant.chromosomes.core import ChromosomeLike


@contextmanager
def elapsed_time_printer[C: ChromosomeLike](
    *,
    op_name: str,
    op_number: tuple[int, int] | None = None,
    chrom_set: ChromosomeSet[C] | None = None,
):
    start_time = datetime.now()
    try:
        yield
    finally:
        elapsed = (datetime.now() - start_time).total_seconds()
        print('op complete', f'[{op_number[0]}/{op_number[1]}]:' if op_number else ':')
        print(
            f'{elapsed}s elapsed',
            f'for {op_name}' if op_name else '',
            f'on {chrom_set.num_chroms} chroms @ {chrom_set.chrom_length} bins/chrom.'
            if chrom_set
            else '.',
        )


def calculate_max_y(args: list[Any]) -> float:
    flat_args = list(itertools.chain.from_iterable(args))
    return np.max(flat_args) * 1.1


def mapThenPrint[T, K](fn: Callable[[T], K], iterable: Iterable[T]):
    for x in map(fn, iterable):
        print(x)
