import logging
from math import inf

import numpy as np
from src.specific_interf_sims.data_transformers import transform_mean_cos_per_chrom_data
from src.specific_interf_sims.interf_schemes import InterferenceScheme
from src.specific_interf_sims.models import induce_COs_via_random_forward_draw
from src.specific_interf_sims.simulator import (
    SimModel,
    SimulationQueue,
    queue_sim,
    run_sims,
)

from croissant._numeric_utils import Interval
from croissant.chromosomes import ChromsetMaker, make_default_chromset


class RecombRateComparer:
    _ref_scheme: InterferenceScheme
    ref_scheme_label: str
    _exp_scheme: InterferenceScheme
    exp_scheme_label: str
    _sim_queue: SimulationQueue
    _chrom_length: int

    def __init__(
        self,
        ref_scheme: InterferenceScheme,
        ref_scheme_label: str,
        exp_scheme: InterferenceScheme,
        exp_scheme_label: str,
        rand_seed: int = 93719,
        chromset_maker: ChromsetMaker | None = make_default_chromset,
        num_chroms: int = 100000,
        chrom_length: int = 1000,
        model: SimModel = induce_COs_via_random_forward_draw,
    ):
        self._ref_scheme = ref_scheme
        self.ref_scheme_label = ref_scheme_label
        self._exp_scheme = exp_scheme
        self.exp_scheme_label = exp_scheme_label
        self._chrom_length = chrom_length
        self._sim_queue = SimulationQueue(
            model,
            rand_seed=rand_seed,
            chroms_per_sim=num_chroms,
            default_chromset_maker=chromset_maker,
        )

        pass

    def match_rate(self, *, target: Interval) -> float:
        return self._match_rate(0, inf, target)

    def _match_rate(self, lower: float, upper: float, target: Interval) -> float:
        curr_avg = (lower * 2) if upper == inf else (lower + upper) / 2
        if lower == 0 and upper == inf:
            curr_avg = 1
        logger = logging.getLogger(__name__)
        logger.info('bounds: [%s, %s]', lower, upper)
        logger.info('current avg: %s', curr_avg)

        ref_set = queue_sim(
            self._sim_queue,
            f'{self.ref_scheme_label} ({curr_avg})',
            self._ref_scheme,
            starting_bins=np.zeros(self._chrom_length)
            + (curr_avg / self._chrom_length),
        )
        exp_set = queue_sim(
            self._sim_queue,
            f'{self.exp_scheme_label} ({curr_avg})',
            self._exp_scheme,
            starting_bins=np.zeros(self._chrom_length)
            + (curr_avg / self._chrom_length),
        )
        run_sims(self._sim_queue)

        (_, ref_data), (_, exp_data) = map(
            transform_mean_cos_per_chrom_data, (ref_set, exp_set)
        )
        logger.info('ref-avg: %s', ref_data)
        logger.info('exp-avg: %s', exp_data)
        if exp_data in target:
            return ref_data

        if exp_data > target:
            return self._match_rate(lower, curr_avg, target)
        else:
            return self._match_rate(curr_avg, upper, target)

    def build_tbl(self, within: Interval) -> str:  # pyright: ignore[reportUnusedParameter]
        """
        If an interval of recomb rates is not provided, the table
        is built using any sims previously run by this comparer
        instance.
        """
        raise NotImplementedError()
