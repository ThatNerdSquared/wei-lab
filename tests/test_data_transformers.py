# pyright: reportPrivateUsage=false
from pprint import pprint
from typing import cast

import numpy as np
from croissant.chromosomes import (
    ChromosomeSet,
    ChromosomeTetrad,
    SequentialSisterSelector,
)
from tests.fixtures import (
    mock_chromset,  # pyright: ignore[reportUnusedImport]
)

from src.specific_interf_sims.data_transformers import (
    Region,
    transform_single_refregion_coc_curve_data,
)


def test_single_refregion_coc_calc_blank(
    mock_chromset: ChromosomeSet[ChromosomeTetrad],
):
    _, coc_data = transform_single_refregion_coc_curve_data(mock_chromset)
    expected_data_blank = np.array([-1] * 10 + [0] * 40)
    assert np.array_equal(coc_data, expected_data_blank)


def test_single_refregion_coc_calc_realistic(
    mock_chromset: ChromosomeSet[ChromosomeTetrad],
):
    test_ref_region = Region(10, 20)

    # insert visible refregion CO into 10% (4 chroms => 8 chromatids)
    mock_chromset.chroms[0].insert_crossover(15)
    mock_chromset.chroms[2].insert_crossover(16)
    mock_chromset.chroms[4].insert_crossover(13)
    mock_chromset.chroms[8].insert_crossover(11)

    # insert invisible refregion CO into 5% (2 chrom => 4 chromatids)
    mock_chromset.chroms[5].insert_crossover(13)
    cast(
        SequentialSisterSelector, mock_chromset.chroms[5].selectSistersForCO
    )._counter = 0
    mock_chromset.chroms[5].insert_crossover(17)
    mock_chromset.chroms[7].insert_crossover(14)
    cast(
        SequentialSisterSelector, mock_chromset.chroms[7].selectSistersForCO
    )._counter = 0
    mock_chromset.chroms[7].insert_crossover(18)

    # at c = 2: insert CO into 10% (4 chromatids with visible refregion CO, 4 without)
    cast(
        SequentialSisterSelector, mock_chromset.chroms[0].selectSistersForCO
    )._counter = 0
    cast(
        SequentialSisterSelector, mock_chromset.chroms[2].selectSistersForCO
    )._counter = 0

    mock_chromset.chroms[0].insert_crossover(2)
    mock_chromset.chroms[2].insert_crossover(2)

    mock_chromset.chroms[1].insert_crossover(2)
    mock_chromset.chroms[3].insert_crossover(2)

    # at c = 9: insert CO into 15% (2 with visible refregion CO, 10 without)
    cast(
        SequentialSisterSelector, mock_chromset.chroms[4].selectSistersForCO
    )._counter = 0

    mock_chromset.chroms[4].insert_crossover(9)
    mock_chromset.chroms[9].insert_crossover(9)
    mock_chromset.chroms[10].insert_crossover(9)
    mock_chromset.chroms[11].insert_crossover(9)
    mock_chromset.chroms[12].insert_crossover(9)
    mock_chromset.chroms[15].insert_crossover(9)

    # at c = 39: insert CO into 5% (1 with visible refregion CO, 3 without)
    cast(
        SequentialSisterSelector, mock_chromset.chroms[8].selectSistersForCO
    )._counter = 1
    mock_chromset.chroms[8].insert_crossover(39)
    mock_chromset.chroms[16].insert_crossover(39)

    # at c = 41: insert CO into 2.5% (cancel out previous)
    cast(
        SequentialSisterSelector, mock_chromset.chroms[16].selectSistersForCO
    )._counter = 0
    mock_chromset.chroms[16].insert_crossover(41)

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
    mock_chromset: ChromosomeSet[ChromosomeTetrad],
):
    test_ref_region = Region(10, 20)

    # insert visible refregion CO into 10% (4 chroms => 8 chromatids)
    mock_chromset.chroms[0].insert_crossover(15)
    mock_chromset.chroms[2].insert_crossover(16)
    mock_chromset.chroms[4].insert_crossover(13)
    mock_chromset.chroms[8].insert_crossover(11)

    # insert invisible refregion CO into 5% (2 chrom => 4 chromatids)
    mock_chromset.chroms[5].insert_crossover(13)
    cast(
        SequentialSisterSelector, mock_chromset.chroms[5].selectSistersForCO
    )._counter = 0
    mock_chromset.chroms[5].insert_crossover(17)
    mock_chromset.chroms[7].insert_crossover(14)
    cast(
        SequentialSisterSelector, mock_chromset.chroms[7].selectSistersForCO
    )._counter = 0
    mock_chromset.chroms[7].insert_crossover(18)

    # at c = 2: insert CO into 10% (4 chromatids with visible refregion CO, 4 without)
    cast(
        SequentialSisterSelector, mock_chromset.chroms[0].selectSistersForCO
    )._counter = 0
    cast(
        SequentialSisterSelector, mock_chromset.chroms[2].selectSistersForCO
    )._counter = 0

    mock_chromset.chroms[0].insert_crossover(2)
    mock_chromset.chroms[2].insert_crossover(2)

    mock_chromset.chroms[1].insert_crossover(2)
    mock_chromset.chroms[3].insert_crossover(2)

    # at c = 9: insert CO into 15% (2 with visible refregion CO, 10 without)
    cast(
        SequentialSisterSelector, mock_chromset.chroms[4].selectSistersForCO
    )._counter = 0

    mock_chromset.chroms[4].insert_crossover(9)
    mock_chromset.chroms[9].insert_crossover(9)
    mock_chromset.chroms[10].insert_crossover(9)
    mock_chromset.chroms[11].insert_crossover(9)
    mock_chromset.chroms[12].insert_crossover(9)
    mock_chromset.chroms[15].insert_crossover(9)

    # at c = 39: insert CO into 5% (1 with visible refregion CO, 3 without)
    cast(
        SequentialSisterSelector, mock_chromset.chroms[8].selectSistersForCO
    )._counter = 1
    mock_chromset.chroms[8].insert_crossover(39)
    mock_chromset.chroms[16].insert_crossover(39)

    # at c = 41: insert CO into 2.5% (cancel out previous)
    cast(
        SequentialSisterSelector, mock_chromset.chroms[16].selectSistersForCO
    )._counter = 0
    mock_chromset.chroms[16].insert_crossover(41)

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
