from operator import attrgetter

from croissant.chromosomes.chromatid import Chromatid, invert_haplotypes_in_co_stack
from croissant.chromosomes.core import Crossover, CrossoverStack, Haplotype
from tests.fixtures import (
    mock_chromatid,  # pyright: ignore[reportUnusedImport]
    mock_chromatid_empty,  # pyright: ignore[reportUnusedImport]
    mock_chromset,  # pyright: ignore[reportUnusedImport]
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
    removed_stack = mock_chromatid.remove_co_stack_to_exchange(11)
    assert removed_stack == CrossoverStack(
        [
            Crossover(400, Haplotype.BLUE),
            Crossover(42, Haplotype.PINK),
            Crossover(39, Haplotype.PINK),
            Crossover(11, Haplotype.BLUE),
        ]
    )


def test_remove_co_stack_all(mock_chromatid: Chromatid):
    removed_stack = mock_chromatid.remove_co_stack_to_exchange(9)
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
    removed_stack = mock_chromatid.remove_co_stack_to_exchange(401)
    assert removed_stack == CrossoverStack([Crossover(401, Haplotype.BLUE)])


def test_remove_co_stack_empty_chromatid(mock_chromatid_empty: Chromatid):
    removed_stack = mock_chromatid_empty.remove_co_stack_to_exchange(0)
    assert removed_stack == CrossoverStack([Crossover(0, Haplotype.PINK)])


def test_add_co_stack(mock_chromatid: Chromatid, mock_crossovers: list[Crossover]):
    stack_to_add = CrossoverStack(
        [Crossover(402, Haplotype.PINK), Crossover(401, Haplotype.BLUE)]
    )
    expected_cos = mock_crossovers + list(reversed(stack_to_add))
    mock_chromatid.add_co_stack_from_exchange(stack_to_add)
    assert mock_chromatid.crossovers == expected_cos
    assert mock_chromatid.crossovers == sorted(expected_cos, key=attrgetter('loc'))


def test_insert_co_for_chromatid_empty(mock_chromatid_empty: Chromatid):
    expected_cos = [
        Crossover(77, Haplotype.BLUE),
    ]
    mock_chromatid_empty.insert_crossover(77)
    assert mock_chromatid_empty.crossovers == expected_cos


def test_insert_co_for_chromatid(mock_chromatid: Chromatid):
    expected_cos = [
        Crossover(10, Haplotype.BLUE),
        Crossover(39, Haplotype.PINK),
        Crossover(40, Haplotype.BLUE),
        Crossover(42, Haplotype.BLUE),
        Crossover(400, Haplotype.PINK),
    ]
    mock_chromatid.insert_crossover(40)
    assert mock_chromatid.crossovers == expected_cos


def test_get_num_cos(mock_chromatid: Chromatid):
    assert mock_chromatid.get_num_crossovers() == 4


def test_get_co_locs(mock_chromatid: Chromatid):
    assert mock_chromatid.get_crossover_locations() == [10, 39, 42, 400]


def test_haplotype_at_for_chromatid(mock_chromatid: Chromatid):
    assert mock_chromatid.haplotype_at(5) == Haplotype.PINK
    assert mock_chromatid.haplotype_at(10) == Haplotype.BLUE
    assert mock_chromatid.haplotype_at(38) == Haplotype.BLUE
    assert mock_chromatid.haplotype_at(39) == Haplotype.PINK
    assert mock_chromatid.haplotype_at(267) == Haplotype.PINK


def test_has_co_at_loc(mock_chromatid: Chromatid):
    assert mock_chromatid.has_crossover_at(10)
    assert not mock_chromatid.has_crossover_at(11)


def test_has_visible_co(mock_chromatid: Chromatid):
    # base is pink, last haplotype is blue
    assert mock_chromatid.has_visible_crossover()

    # start is first bin of pink, last haplotype is blue
    assert mock_chromatid.has_visible_crossover(start=39)

    # start is last bin of pink, last haplotype is blue
    assert mock_chromatid.has_visible_crossover(start=309)

    # base is pink, end is first bin of blue
    assert mock_chromatid.has_visible_crossover(end=10)

    # base is pink, end is last bin of blue
    assert mock_chromatid.has_visible_crossover(end=38)

    # start is first bin of pink, end is first bin of blue
    assert mock_chromatid.has_visible_crossover(start=39, end=400)

    # start is last bin of pink, end is first bin of blue
    assert mock_chromatid.has_visible_crossover(start=309, end=400)

    # start is last bin of pink, end is last bin of blue
    assert mock_chromatid.has_visible_crossover(start=9, end=38)


def test_no_visible_co(mock_chromatid: Chromatid):
    # base is pink, end is first bin of pink
    assert not mock_chromatid.has_visible_crossover(end=39)

    # base is pink, end is last bin of pink
    assert not mock_chromatid.has_visible_crossover(end=309)

    # start is first bin of blue, last haplotype is blue
    assert not mock_chromatid.has_visible_crossover(start=10)

    # start is last bin of blue, end is first bin of blue
    assert not mock_chromatid.has_visible_crossover(start=38, end=400)

    # start is first bin of pink, end is first bin of pink
    assert not mock_chromatid.has_visible_crossover(start=39, end=42)

    # start is last bin of pink, end is first bin of pink
    assert not mock_chromatid.has_visible_crossover(start=9, end=39)


def test_chromatid_into_list(mock_chromatid: Chromatid):
    res = mock_chromatid.into_list(5)
    assert len(res) == 5
    for item in res:
        assert item.base_haplotype == mock_chromatid.base_haplotype
        assert item.length == mock_chromatid.length
        assert not item.crossovers
