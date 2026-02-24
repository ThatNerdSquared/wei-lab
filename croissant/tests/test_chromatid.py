from operator import attrgetter

import numpy as np

from croissant.chromosomes.chromatid import (
    Chromatid,
    Crossover,
    CrossoverStack,
    Haplotype,
    add_co_stack_from_exchange,
    convert_chromatid_to_haplotype_tbl,
    get_crossover_locations,
    get_num_crossovers,
    has_crossover_at,
    has_visible_crossover,
    insert_crossover,
    invert_haplotypes_in_co_stack,
    remove_co_stack_to_exchange,
)
from tests.fixtures import (
    mock_chromatid,  # pyright: ignore[reportUnusedImport]
    mock_chromatid_empty,  # pyright: ignore[reportUnusedImport]
    mock_co_stack,  # pyright: ignore[reportUnusedImport]
    mock_crossovers,  # pyright: ignore[reportUnusedImport]
)


def test_invert_haplotypes_in_co_stack(mock_co_stack: CrossoverStack):
    co_stack_after = CrossoverStack(
        [
            Crossover(400, Haplotype.PINK),
            Crossover(42, Haplotype.BLUE),
            Crossover(39, Haplotype.BLUE),
            Crossover(10, Haplotype.PINK),
        ]
    )
    assert invert_haplotypes_in_co_stack(mock_co_stack) == co_stack_after


def test_remove_co_stack(mock_chromatid: Chromatid):
    removed_stack = remove_co_stack_to_exchange(mock_chromatid, 11)
    assert removed_stack == CrossoverStack(
        [
            Crossover(400, Haplotype.BLUE),
            Crossover(42, Haplotype.PINK),
            Crossover(39, Haplotype.PINK),
            Crossover(11, Haplotype.BLUE),
        ]
    )


def test_remove_co_stack_all(mock_chromatid: Chromatid):
    removed_stack = remove_co_stack_to_exchange(mock_chromatid, 9)
    assert removed_stack == CrossoverStack(
        [
            Crossover(400, Haplotype.BLUE),
            Crossover(42, Haplotype.PINK),
            Crossover(39, Haplotype.PINK),
            Crossover(10, Haplotype.BLUE),
            Crossover(9, Haplotype.PINK),
        ]
    )


def test_remove_co_stack_none(mock_chromatid: Chromatid):
    removed_stack = remove_co_stack_to_exchange(mock_chromatid, 401)
    assert removed_stack == CrossoverStack([Crossover(401, Haplotype.BLUE)])


def test_remove_co_stack_empty_chromatid(mock_chromatid_empty: Chromatid):
    removed_stack = remove_co_stack_to_exchange(mock_chromatid_empty, 0)
    assert removed_stack == CrossoverStack([Crossover(0, Haplotype.PINK)])


def test_add_co_stack(mock_chromatid: Chromatid, mock_crossovers: list[Crossover]):
    stack_to_add = CrossoverStack(
        [Crossover(402, Haplotype.PINK), Crossover(401, Haplotype.BLUE)]
    )
    expected_cos = mock_crossovers + list(reversed(stack_to_add))
    add_co_stack_from_exchange(mock_chromatid, stack_to_add)
    assert mock_chromatid.crossovers == expected_cos
    assert mock_chromatid.crossovers == sorted(expected_cos, key=attrgetter('loc'))


def test_convert_chromatid_to_haplotype_tbl(
    mock_chromatid: Chromatid, mock_chromatid_empty: Chromatid
):
    input_array = np.zeros(mock_chromatid_empty.length, dtype=np.int64)
    assert np.sum(input_array) == 0
    convert_chromatid_to_haplotype_tbl(mock_chromatid_empty, input_array)
    assert np.sum(input_array) == mock_chromatid_empty.length

    input_array = np.zeros(mock_chromatid.length, dtype=np.int64)
    assert np.sum(input_array) == 0
    input_array -= 1
    convert_chromatid_to_haplotype_tbl(mock_chromatid, input_array)
    assert -1 not in input_array
    expected_array = np.array([1] * 10 + [0] * 29 + [1] * 3 + [1] * 358 + [0] * 100)
    assert np.array_equal(input_array, expected_array)


def test_insert_co_for_chromatid_empty(mock_chromatid_empty: Chromatid):
    expected_cos = [
        Crossover(77, Haplotype.BLUE),
    ]
    insert_crossover(mock_chromatid_empty, 77)
    assert mock_chromatid_empty.crossovers == expected_cos


def test_insert_co_for_chromatid(mock_chromatid: Chromatid):
    expected_cos = [
        Crossover(10, Haplotype.BLUE),
        Crossover(39, Haplotype.PINK),
        Crossover(40, Haplotype.BLUE),
        Crossover(42, Haplotype.BLUE),
        Crossover(400, Haplotype.PINK),
    ]
    insert_crossover(mock_chromatid, 40)
    assert mock_chromatid.crossovers == expected_cos


def test_get_num_cos(mock_chromatid: Chromatid):
    assert get_num_crossovers(mock_chromatid) == 4


def test_get_co_locs(mock_chromatid: Chromatid):
    assert get_crossover_locations(mock_chromatid) == [10, 39, 42, 400]


def test_has_co_at_loc(mock_chromatid: Chromatid):
    assert has_crossover_at(mock_chromatid, 10)
    assert not has_crossover_at(mock_chromatid, 11)


def test_has_visible_co(mock_chromatid: Chromatid):
    # base is pink, last haplotype is blue
    assert has_visible_crossover(mock_chromatid)

    # start is first bin of pink, last haplotype is blue
    assert has_visible_crossover(mock_chromatid, start=39)

    # start is last bin of pink, last haplotype is blue
    assert has_visible_crossover(mock_chromatid, start=309)

    # base is pink, end is first bin of blue
    assert has_visible_crossover(mock_chromatid, end=10)

    # base is pink, end is last bin of blue
    assert has_visible_crossover(mock_chromatid, end=38)

    # start is first bin of pink, end is first bin of blue
    assert has_visible_crossover(mock_chromatid, start=39, end=400)

    # start is last bin of pink, end is first bin of blue
    assert has_visible_crossover(mock_chromatid, start=309, end=400)

    # start is last bin of pink, end is last bin of blue
    assert has_visible_crossover(mock_chromatid, start=9, end=38)


def test_no_visible_co(mock_chromatid: Chromatid):
    # base is pink, end is first bin of pink
    assert not has_visible_crossover(mock_chromatid, end=39)

    # base is pink, end is last bin of pink
    assert not has_visible_crossover(mock_chromatid, end=309)

    # start is first bin of blue, last haplotype is blue
    assert not has_visible_crossover(mock_chromatid, start=10)

    # start is last bin of blue, end is first bin of blue
    assert not has_visible_crossover(mock_chromatid, start=38, end=400)

    # start is first bin of pink, end is first bin of pink
    assert not has_visible_crossover(mock_chromatid, start=39, end=42)

    # start is last bin of pink, end is first bin of pink
    assert not has_visible_crossover(mock_chromatid, start=9, end=39)
