# pyright: reportPrivateUsage=false
from pprint import pprint
from typing import cast

import numpy as np

from src.specific_interf_sims.data_transformers import (
    Region,
    transform_single_refregion_coc_curve_data,
)
from src.utils.chromatid import insert_crossover
from src.utils.chromosome_set import ChromosomeSet
from src.utils.chromosome_with_sister_chromatids import (
    ChromosomeSisterChromatids,
    SequentialSisterSelector,
)
from tests.fixtures import (
    mock_chromset,  # pyright: ignore[reportUnusedImport]
)


def test_single_refregion_coc_calc_blank(
    mock_chromset: ChromosomeSet[ChromosomeSisterChromatids],
):
    _, coc_data = transform_single_refregion_coc_curve_data(mock_chromset)
    expected_data_blank = np.array([-1] * 10 + [0] * 40)
    assert np.array_equal(coc_data, expected_data_blank)


def test_single_refregion_coc_calc_realistic(
    mock_chromset: ChromosomeSet[ChromosomeSisterChromatids],
):
    test_ref_region = Region(10, 20)

    # insert visible refregion CO into 10% (4 chroms => 8 chromatids)
    insert_crossover(mock_chromset.chroms[0], 15)
    insert_crossover(mock_chromset.chroms[2], 16)
    insert_crossover(mock_chromset.chroms[4], 13)
    insert_crossover(mock_chromset.chroms[8], 11)

    # insert invisible refregion CO into 5% (2 chrom => 4 chromatids)
    insert_crossover(mock_chromset.chroms[5], 13)
    cast(
        SequentialSisterSelector, mock_chromset.chroms[5].selectSistersForCO
    )._counter = 0
    insert_crossover(mock_chromset.chroms[5], 17)
    insert_crossover(mock_chromset.chroms[7], 14)
    cast(
        SequentialSisterSelector, mock_chromset.chroms[7].selectSistersForCO
    )._counter = 0
    insert_crossover(mock_chromset.chroms[7], 18)

    # at c = 2: insert CO into 10% (4 chromatids with visible refregion CO, 4 without)
    cast(
        SequentialSisterSelector, mock_chromset.chroms[0].selectSistersForCO
    )._counter = 0
    cast(
        SequentialSisterSelector, mock_chromset.chroms[2].selectSistersForCO
    )._counter = 0

    insert_crossover(mock_chromset.chroms[0], 2)
    insert_crossover(mock_chromset.chroms[2], 2)

    insert_crossover(mock_chromset.chroms[1], 2)
    insert_crossover(mock_chromset.chroms[3], 2)

    # at c = 9: insert CO into 15% (2 with visible refregion CO, 10 without)
    cast(
        SequentialSisterSelector, mock_chromset.chroms[4].selectSistersForCO
    )._counter = 0

    insert_crossover(mock_chromset.chroms[4], 9)
    insert_crossover(mock_chromset.chroms[9], 9)
    insert_crossover(mock_chromset.chroms[10], 9)
    insert_crossover(mock_chromset.chroms[11], 9)
    insert_crossover(mock_chromset.chroms[12], 9)
    insert_crossover(mock_chromset.chroms[15], 9)

    # at c = 39: insert CO into 5% (1 with visible refregion CO, 3 without)
    cast(
        SequentialSisterSelector, mock_chromset.chroms[8].selectSistersForCO
    )._counter = 1
    insert_crossover(mock_chromset.chroms[8], 39)
    insert_crossover(mock_chromset.chroms[16], 39)

    # at c = 41: insert CO into 2.5% (cancel out previous)
    cast(
        SequentialSisterSelector, mock_chromset.chroms[16].selectSistersForCO
    )._counter = 0
    insert_crossover(mock_chromset.chroms[16], 41)

    _, result_data = transform_single_refregion_coc_curve_data(
        mock_chromset, test_ref_region
    )

    expected_coc_before2 = (6 / 80) / (0.1 * (20 / 80))
    expected_coc_before9 = (2 / 80) / (0.1 * 0.15)
    expected_coc_after39 = (1 / 80) / (0.1 * 0.05)
    expected_coc_after41 = (1 / 80) / (0.1 * 0.025)
    expected_data = np.array(
        [expected_coc_before2] * 2
        + [expected_coc_before9] * 7
        + [0]
        + [-1] * 11
        + [0] * 18
        + [expected_coc_after39] * 2
        + [expected_coc_after41] * 9
    )
    pprint(result_data)
    pprint(expected_data)
    assert np.allclose(result_data, expected_data, atol=0.001, rtol=0)


def test_single_refregion_coc_calc_realistic_with_calc_inside_refregion(
    mock_chromset: ChromosomeSet[ChromosomeSisterChromatids],
):
    test_ref_region = Region(10, 20)

    # insert visible refregion CO into 10% (4 chroms => 8 chromatids)
    insert_crossover(mock_chromset.chroms[0], 15)
    insert_crossover(mock_chromset.chroms[2], 16)
    insert_crossover(mock_chromset.chroms[4], 13)
    insert_crossover(mock_chromset.chroms[8], 11)

    # insert invisible refregion CO into 5% (2 chrom => 4 chromatids)
    insert_crossover(mock_chromset.chroms[5], 13)
    cast(
        SequentialSisterSelector, mock_chromset.chroms[5].selectSistersForCO
    )._counter = 0
    insert_crossover(mock_chromset.chroms[5], 17)
    insert_crossover(mock_chromset.chroms[7], 14)
    cast(
        SequentialSisterSelector, mock_chromset.chroms[7].selectSistersForCO
    )._counter = 0
    insert_crossover(mock_chromset.chroms[7], 18)

    # at c = 2: insert CO into 10% (4 chromatids with visible refregion CO, 4 without)
    cast(
        SequentialSisterSelector, mock_chromset.chroms[0].selectSistersForCO
    )._counter = 0
    cast(
        SequentialSisterSelector, mock_chromset.chroms[2].selectSistersForCO
    )._counter = 0

    insert_crossover(mock_chromset.chroms[0], 2)
    insert_crossover(mock_chromset.chroms[2], 2)

    insert_crossover(mock_chromset.chroms[1], 2)
    insert_crossover(mock_chromset.chroms[3], 2)

    # at c = 9: insert CO into 15% (2 with visible refregion CO, 10 without)
    cast(
        SequentialSisterSelector, mock_chromset.chroms[4].selectSistersForCO
    )._counter = 0

    insert_crossover(mock_chromset.chroms[4], 9)
    insert_crossover(mock_chromset.chroms[9], 9)
    insert_crossover(mock_chromset.chroms[10], 9)
    insert_crossover(mock_chromset.chroms[11], 9)
    insert_crossover(mock_chromset.chroms[12], 9)
    insert_crossover(mock_chromset.chroms[15], 9)

    # at c = 39: insert CO into 5% (1 with visible refregion CO, 3 without)
    cast(
        SequentialSisterSelector, mock_chromset.chroms[8].selectSistersForCO
    )._counter = 1
    insert_crossover(mock_chromset.chroms[8], 39)
    insert_crossover(mock_chromset.chroms[16], 39)

    # at c = 41: insert CO into 2.5% (cancel out previous)
    cast(
        SequentialSisterSelector, mock_chromset.chroms[16].selectSistersForCO
    )._counter = 0
    insert_crossover(mock_chromset.chroms[16], 41)

    _, result_data = transform_single_refregion_coc_curve_data(
        mock_chromset, ref_region=test_ref_region, calculate_inside_refregion=True
    )

    expected_coc_before2 = (6 / 80) / (0.1 * (20 / 80))
    expected_coc_before9 = (2 / 80) / (0.1 * 0.15)
    expected_coc_after39 = (1 / 80) / (0.1 * 0.05)
    expected_coc_after41 = (1 / 80) / (0.1 * 0.025)
    expected_data = np.array(
        [expected_coc_before2] * 2
        + [expected_coc_before9] * 7
        + [0, -1, 0, 0]
        + [
            ((2 / 80) / ((6 / 80) * (6 / 80))),
            (4 / 80) / ((8 / 80) * (8 / 80)),
            (4 / 80) / ((10 / 80) * (6 / 80)),
            (4 / 80) / ((12 / 80) * (4 / 80)),
            (2 / 80) / ((10 / 80) * (2 / 80)),
        ]
        + [0] * 2
        + [-1]
        + [0] * 18
        + [expected_coc_after39] * 2
        + [expected_coc_after41] * 9
    )
    pprint(result_data)
    pprint(expected_data)
    assert np.allclose(result_data, expected_data, atol=0.001, rtol=0)
