import numpy as np

from croissant.chromosomes.legacy_chromosome import LegacyChromosome
from tests.fixtures import (
    mock_legacy_chrom,  # pyright: ignore[reportUnusedImport]
    mock_legacy_chrom_single_co,  # pyright: ignore[reportUnusedImport]
)


def test_legacy_chrom_into_list(mock_legacy_chrom: LegacyChromosome):
    res = mock_legacy_chrom.into_list(9)
    assert len(res) == 9
    for item in res:
        assert item.length == mock_legacy_chrom.length
        assert not item._crossovers  # pyright: ignore[reportPrivateUsage]


def test_legacy_chrom_into_haplotype_tbl(mock_legacy_chrom: LegacyChromosome):
    holding_arr = np.zeros((4, mock_legacy_chrom.length), dtype=np.int64)
    holding_arr[:] = -1
    empty_row = np.zeros(mock_legacy_chrom.length, dtype=np.int64)
    empty_row[:] = -1

    expected_tbl_result = np.zeros(mock_legacy_chrom.length, dtype=np.int64)
    expected_tbl_result[0:57] = 0
    expected_tbl_result[57:293] = 1
    expected_tbl_result[293:] = 0

    assert np.array_equal(holding_arr[2], empty_row)
    mock_legacy_chrom.into_haplotype_tbl(holding_arr[2:3])
    assert np.array_equal(holding_arr[0], empty_row)
    assert np.array_equal(holding_arr[1], empty_row)
    assert np.array_equal(holding_arr[2], expected_tbl_result)
    assert np.array_equal(holding_arr[3], empty_row)


def test_legacy_chrom_into_haplotype_tbl_single_co(
    mock_legacy_chrom_single_co: LegacyChromosome,
):
    holding_arr = np.zeros((4, mock_legacy_chrom_single_co.length), dtype=np.int64)
    holding_arr[:] = -1
    empty_row = np.zeros(mock_legacy_chrom_single_co.length, dtype=np.int64)
    empty_row[:] = -1

    expected_tbl_result = np.zeros(mock_legacy_chrom_single_co.length, dtype=np.int64)
    expected_tbl_result[0:81] = 0
    expected_tbl_result[81:] = 1

    assert np.array_equal(holding_arr[2], empty_row)
    mock_legacy_chrom_single_co.into_haplotype_tbl(holding_arr[2:3])
    assert np.array_equal(holding_arr[0], empty_row)
    assert np.array_equal(holding_arr[1], empty_row)
    assert np.array_equal(holding_arr[2], expected_tbl_result)
    assert np.array_equal(holding_arr[3], empty_row)
