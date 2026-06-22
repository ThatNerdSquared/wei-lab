import logging

from croissant._numeric_utils import Interval
from croissant.recomb_rate_comparer import RecombRateComparer

from src.specific_interf_sims.interf_schemes import (
    no_interf,
    very_fast_rising_local_interf,
)

logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
    recomb_rate_comparer = RecombRateComparer(
        no_interf, 'no interf', very_fast_rising_local_interf, 'very-fast-rising local'
    )
    matched_avg = recomb_rate_comparer.match_rate(
        target=Interval.from_abs_tolerance(4.0, tolerance=0.0001)
    )
    print(f'MATCHED AVG: {matched_avg}')


if __name__ == '__main__':
    main()
