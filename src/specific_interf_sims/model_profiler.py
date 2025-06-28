import cProfile
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from pstats import SortKey, Stats
from typing import cast

import numpy as np

from src.specific_interf_sims.interf_schemes import BinArray, local_interf
from src.specific_interf_sims.models import induce_COs_via_random_forward_draw
from src.utils.chromosome_set import ChromosomeSet


@contextmanager
def profile_model(profile_name: str):
    profiler = cProfile.Profile()
    profiler.enable()
    try:
        yield profiler
    finally:
        profiler.disable()
        curr_time = datetime.now().strftime('%Y-%m-%d__%H-%M-%S')
        output_file = Path(f'{curr_time}-{profile_name}')
        output_file.touch()
        with output_file.open(mode='a') as f:
            _ = Stats(profiler, stream=f).sort_stats(SortKey.TIME).print_stats()


def example_profiling_run():
    NUM_CHROMS = 100000
    CHROM_LENGTH = 1000
    AVG_COS_PER_CHROM = 1
    STARTING_BINS = cast(
        BinArray, np.zeros(CHROM_LENGTH) + (AVG_COS_PER_CHROM / CHROM_LENGTH)
    )

    testset_control = ChromosomeSet(
        label='control', num_chroms=NUM_CHROMS, chrom_length=CHROM_LENGTH
    )
    testset_attempt_optimise = ChromosomeSet(
        label='attempted optimisation', num_chroms=NUM_CHROMS, chrom_length=CHROM_LENGTH
    )

    rand = np.random.default_rng(77777)

    def nonvectorised_local_interf(co_loc: int, bins: BinArray) -> BinArray:
        for idx in range(len(bins)):
            bins[idx] = bins[idx] * (
                1 - np.exp(-1 * np.abs((idx / CHROM_LENGTH) - (co_loc / CHROM_LENGTH)))
            )
        return bins

    with profile_model('profile-control'):
        induce_COs_via_random_forward_draw(
            testset_control, STARTING_BINS, nonvectorised_local_interf, rand
        )

    with profile_model('profile-attempt-optimise'):
        induce_COs_via_random_forward_draw(
            testset_attempt_optimise, STARTING_BINS, local_interf, rand
        )


if __name__ == '__main__':
    example_profiling_run()
