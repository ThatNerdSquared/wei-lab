import random
from typing import Callable
from uuid import UUID

import numpy as np

from croissant.chromosomes.chromatid import Chromatid
from croissant.chromosomes.chromosome_tetrad import (
    ChromosomeTetrad,
    SequentialSisterSelector,
    haplotypeSwitchingCOsOnlySelector,
    selectAllSistersForPool,
    selectMostRecombinantForPool,
    selectOnlyRecombinantsForPool,
    selectSingleSisterForPool,
    uniformlyRandomSisterSelector,
)
from croissant.chromosomes.core import Haplotype
from tests.fixtures import (
    ResetRandomCallback,
    mock_chromosome_w_sisters,  # pyright: ignore[reportUnusedImport]
    reset_random_state,  # pyright: ignore[reportUnusedImport]
)


def _random_selector_tester(
    expected_ratio: float, selector: Callable[[], tuple[Chromatid, ...]]
) -> None:
    NUM_TRIALS = 1000

    selection_results: dict[UUID, int] = {}
    for _ in range(1000):
        for chromatid in selector():
            try:
                selection_results[chromatid.id] += 1
            except KeyError:
                selection_results[chromatid.id] = 1
    for chromatid, result in selection_results.items():
        assert (result / NUM_TRIALS) - expected_ratio <= 0.05


def test_uniform_random_sister_selector(
    reset_random_state: ResetRandomCallback,  # pyright: ignore[reportUnusedParameter]
    mock_chromosome_w_sisters: ChromosomeTetrad,
):
    _random_selector_tester(
        0.5, lambda: uniformlyRandomSisterSelector(mock_chromosome_w_sisters, -1)
    )


def test_sequential_sister_selector(
    mock_chromosome_w_sisters: ChromosomeTetrad,
):
    sequentialSisterSelector = SequentialSisterSelector()
    round1_first, round1_second = sequentialSisterSelector(
        mock_chromosome_w_sisters, -1
    )
    assert round1_first == mock_chromosome_w_sisters.a1
    assert round1_second == mock_chromosome_w_sisters.a2

    round2_first, round2_second = sequentialSisterSelector(
        mock_chromosome_w_sisters, -1
    )
    assert round2_first == mock_chromosome_w_sisters.a2
    assert round2_second == mock_chromosome_w_sisters.b1

    round3_first, round3_second = sequentialSisterSelector(
        mock_chromosome_w_sisters, -1
    )
    assert round3_first == mock_chromosome_w_sisters.b1
    assert round3_second == mock_chromosome_w_sisters.b2

    round4_first, round4_second = sequentialSisterSelector(
        mock_chromosome_w_sisters, -1
    )
    assert round4_first == mock_chromosome_w_sisters.b2
    assert round4_second == mock_chromosome_w_sisters.a1


def test_haplotype_switching_only_sister_selector():
    test_chrom = ChromosomeTetrad(
        length=500,
        selectSistersForCO=haplotypeSwitchingCOsOnlySelector,
        selectSistersForPool=selectAllSistersForPool,
    )
    test_chrom.a2.insert_crossover(40)
    test_chrom.b1.insert_crossover(40)

    selections: list[tuple[Chromatid, Chromatid]] = []
    for _ in range(1000):
        selections.append(haplotypeSwitchingCOsOnlySelector(test_chrom, 39))

    for selection in selections:
        assert not (selection[0] == test_chrom.a1 and selection[1] == test_chrom.a2)
        assert not (selection[0] == test_chrom.a2 and selection[1] == test_chrom.a1)
        assert not (selection[0] == test_chrom.b1 and selection[1] == test_chrom.b2)
        assert not (selection[0] == test_chrom.b2 and selection[1] == test_chrom.b1)

    selections_r2: list[tuple[Chromatid, Chromatid]] = []
    for _ in range(1000):
        selections_r2.append(haplotypeSwitchingCOsOnlySelector(test_chrom, 40))

    for selection in selections_r2:
        assert not (selection[0] == test_chrom.a1 and selection[1] == test_chrom.b1)
        assert not (selection[0] == test_chrom.b1 and selection[1] == test_chrom.a1)
        assert not (selection[0] == test_chrom.a2 and selection[1] == test_chrom.b2)
        assert not (selection[0] == test_chrom.b2 and selection[1] == test_chrom.a2)


def test_random_sister_selector_for_pool(
    reset_random_state: ResetRandomCallback,  # pyright: ignore[reportUnusedParameter]
    mock_chromosome_w_sisters: ChromosomeTetrad,
):
    _random_selector_tester(
        0.25, lambda: selectSingleSisterForPool(mock_chromosome_w_sisters)
    )


def test_most_recombinant_pool_selector():
    sequentialSelector = SequentialSisterSelector(init_count=0)
    test_chrom = ChromosomeTetrad(
        length=500,
        selectSistersForCO=sequentialSelector,
        selectSistersForPool=selectMostRecombinantForPool,
    )
    test_chrom.insert_crossover(42)
    test_chrom.insert_crossover(390)
    assert test_chrom.selectSistersForPool(test_chrom) == (test_chrom.b1,)


def test_only_recombinant_pool_selector():
    selected_chromatids: list[Chromatid] = []

    for _ in range(1000):
        test_chrom = ChromosomeTetrad(
            length=500,
            selectSistersForCO=uniformlyRandomSisterSelector,
            selectSistersForPool=selectOnlyRecombinantsForPool,
        )
        test_chrom.insert_crossover(random.randint(0, 500))
        selection_result = test_chrom.selectSistersForPool(test_chrom)
        if selection_result:
            selected_chromatids.append(selection_result[0])

    for selection in selected_chromatids:
        assert selection.get_num_crossovers() > 0


def test_insert_co_for_sister_chromatids(
    reset_random_state: ResetRandomCallback,
    mock_chromosome_w_sisters: ChromosomeTetrad,
):
    mock_chromosome_w_sisters.insert_crossover(400)
    reset_random_state()

    first_selection, second_selection = uniformlyRandomSisterSelector(
        mock_chromosome_w_sisters, -1
    )
    assert first_selection.has_crossover_at(400)
    assert second_selection.has_crossover_at(400)
    reset_random_state()

    mock_chromosome_w_sisters.insert_crossover(3)
    reset_random_state()

    first_selection, second_selection = uniformlyRandomSisterSelector(
        mock_chromosome_w_sisters, -1
    )

    assert first_selection.has_crossover_at(400)
    assert second_selection.has_crossover_at(400)
    assert first_selection.has_crossover_at(3)
    assert second_selection.has_crossover_at(3)
    assert not first_selection.has_crossover_at(4)
    assert not second_selection.has_crossover_at(4)


def test_get_num_cos_for_sister_chromatids(
    mock_chromosome_w_sisters: ChromosomeTetrad,
):
    mock_chromosome_w_sisters.insert_crossover(42)
    assert mock_chromosome_w_sisters.get_num_crossovers() == 1
    mock_chromosome_w_sisters.insert_crossover(99)
    assert mock_chromosome_w_sisters.get_num_crossovers() == 2
    mock_chromosome_w_sisters.insert_crossover(402)
    assert mock_chromosome_w_sisters.get_num_crossovers() == 3


def test_get_co_locations_for_sister_chromatids(
    mock_chromosome_w_sisters: ChromosomeTetrad,
):
    mock_chromosome_w_sisters.insert_crossover(42)
    mock_chromosome_w_sisters.insert_crossover(99)
    mock_chromosome_w_sisters.insert_crossover(402)
    assert mock_chromosome_w_sisters.get_crossover_locations() == [42, 99, 402]


def test_has_co_at_for_sister_chromatids(
    mock_chromosome_w_sisters: ChromosomeTetrad,
):
    mock_chromosome_w_sisters.insert_crossover(42)
    assert mock_chromosome_w_sisters.has_crossover_at(42)
    assert not mock_chromosome_w_sisters.has_crossover_at(99)

    mock_chromosome_w_sisters.insert_crossover(99)
    assert mock_chromosome_w_sisters.has_crossover_at(99)

    mock_chromosome_w_sisters.insert_crossover(402)
    assert mock_chromosome_w_sisters.has_crossover_at(402)

    assert not mock_chromosome_w_sisters.has_crossover_at(43)
    assert not mock_chromosome_w_sisters.has_crossover_at(98)
    assert not mock_chromosome_w_sisters.has_crossover_at(401)


def test_has_visible_co_for_sister_chromatids(
    mock_chromosome_w_sisters: ChromosomeTetrad,
):
    mock_chromosome_w_sisters.selectSistersForCO = SequentialSisterSelector()
    mock_chromosome_w_sisters.insert_crossover(39)
    """
    -------X-----------------
    -------X-----------------
    •••••••••••••••••••••••••
    •••••••••••••••••••••••••
    """
    assert mock_chromosome_w_sisters.get_crossover_locations() == [39]
    assert not mock_chromosome_w_sisters.has_visible_crossover()

    mock_chromosome_w_sisters.insert_crossover(42)
    """
    -------X-----------------
    -------X-----X•••••••••••
    •••••••••••••X-----------
    •••••••••••••••••••••••••
    """
    assert mock_chromosome_w_sisters.get_crossover_locations() == [39, 42]
    assert not mock_chromosome_w_sisters.has_visible_crossover(end=41)
    assert not mock_chromosome_w_sisters.has_visible_crossover(start=39, end=41)

    assert mock_chromosome_w_sisters.has_visible_crossover(start=39)
    assert mock_chromosome_w_sisters.has_visible_crossover(start=41)
    assert mock_chromosome_w_sisters.has_visible_crossover(start=41, end=42)
    assert mock_chromosome_w_sisters.has_visible_crossover()
    assert not mock_chromosome_w_sisters.has_visible_crossover(start=42)

    mock_chromosome_w_sisters.insert_crossover(400)
    """
    -------X-----------------
    -------X-----X•••••••••••
    •••••••••••••X-----X•••••
    •••••••••••••••••••X-----
    """
    assert mock_chromosome_w_sisters.get_crossover_locations() == [39, 42, 400]
    assert mock_chromosome_w_sisters.has_visible_crossover()
    assert mock_chromosome_w_sisters.has_visible_crossover(start=309)

    # test region not including where visible break occurred
    assert not mock_chromosome_w_sisters.has_visible_crossover(end=41)
    assert not mock_chromosome_w_sisters.has_visible_crossover(start=400)

    # double check visibility on chromatid
    assert mock_chromosome_w_sisters.b1.has_visible_crossover(start=42)
    assert not mock_chromosome_w_sisters.b1.has_visible_crossover()


def test_tetrad_into_list(mock_chromosome_w_sisters: ChromosomeTetrad):
    res = mock_chromosome_w_sisters.into_list(6)
    assert len(res) == 6
    for item in res:
        assert item.length == mock_chromosome_w_sisters.length
        assert item.selectSistersForCO == mock_chromosome_w_sisters.selectSistersForCO
        assert (
            item.selectSistersForPool == mock_chromosome_w_sisters.selectSistersForPool
        )


def test_tetrad_into_haplotype_tbl(
    mock_chromosome_w_sisters: ChromosomeTetrad,
):
    holding_arr = np.zeros((16, mock_chromosome_w_sisters.length), dtype=np.int64)
    holding_arr[:] = -1
    empty_row = np.zeros(mock_chromosome_w_sisters.length, dtype=np.int64)
    empty_row[:] = -1

    mock_chromosome_w_sisters.selectSistersForCO = SequentialSisterSelector()
    mock_chromosome_w_sisters.insert_crossover(42)
    mock_chromosome_w_sisters.insert_crossover(339)
    mock_chromosome_w_sisters.into_haplotype_tbl(holding_arr[4:8])

    for i in range(4):
        assert np.array_equal(holding_arr[i], empty_row)
    assert np.array_equal(
        holding_arr[4],
        np.array(
            [mock_chromosome_w_sisters.a1.base_haplotype.value]
            * mock_chromosome_w_sisters.length
        ),
    )
    assert np.array_equal(
        holding_arr[5],
        np.array(
            [mock_chromosome_w_sisters.a2.base_haplotype.value] * 339
            + [Haplotype(not mock_chromosome_w_sisters.a2.base_haplotype).value]
            * (mock_chromosome_w_sisters.length - 339)
        ),
    )
    assert np.array_equal(
        holding_arr[6],
        np.array(
            [mock_chromosome_w_sisters.b2.base_haplotype.value] * 339
            + [Haplotype(not mock_chromosome_w_sisters.b2.base_haplotype).value]
            * (mock_chromosome_w_sisters.length - 339)
        ),
    )
    assert np.array_equal(
        holding_arr[7],
        np.array(
            [mock_chromosome_w_sisters.b2.base_haplotype.value]
            * mock_chromosome_w_sisters.length
        ),
    )
    for i in range(8, 16):
        assert np.array_equal(holding_arr[i], empty_row)
