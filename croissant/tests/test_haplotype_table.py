import numpy as np

from croissant.chromosomes import Chromatid, ChromosomeTetrad, LegacyChromosome
from croissant.chromosomes.chromosome_tetrad import (
    SequentialSisterSelector,
    selectSingleSisterForPool,
)
from croissant.chromosomes.core import Haplotype
from croissant.haplotype_table import HaplotypeTable
from tests.fixtures import (
    mock_chromatid,  # pyright: ignore[reportUnusedImport]
    mock_chromatid_empty,  # pyright: ignore[reportUnusedImport]
    mock_chromosome_w_sisters,  # pyright: ignore[reportUnusedImport]
    mock_crossovers,  # pyright: ignore[reportUnusedImport]
    mock_legacy_chrom,  # pyright: ignore[reportUnusedImport]
    mock_legacy_chrom_single_co,  # pyright: ignore[reportUnusedImport]
)


def test_haptbl_from_empty_chromatid(mock_chromatid_empty: Chromatid):
    haptbl = HaplotypeTable.from_chromosomes([mock_chromatid_empty])
    assert np.sum(haptbl.tbl) == mock_chromatid_empty.length


def test_haptbl_from_chromatids(
    mock_chromatid: Chromatid, mock_chromatid_empty: Chromatid
):
    tester_chroms = [
        mock_chromatid_empty,
        mock_chromatid_empty,
        mock_chromatid,
        mock_chromatid_empty,
    ]
    empty_row = np.zeros(mock_chromatid.length, dtype=np.int64)
    empty_row[:] = 1

    expected_tbl_result = np.zeros(mock_chromatid.length, dtype=np.int64)
    expected_tbl_result[0:10] = 1
    expected_tbl_result[10:39] = 0
    expected_tbl_result[39:42] = 1
    expected_tbl_result[42:400] = 1
    expected_tbl_result[400:] = 0

    haptbl = HaplotypeTable.from_chromosomes(tester_chroms)

    assert np.array_equal(haptbl.tbl[0], empty_row)
    assert np.array_equal(haptbl.tbl[1], empty_row)
    assert np.array_equal(haptbl.tbl[2], expected_tbl_result)
    assert np.array_equal(haptbl.tbl[3], empty_row)


def test_haptbl_from_tetrads(
    mock_chromosome_w_sisters: ChromosomeTetrad,
):
    holding_arr = np.zeros((16, mock_chromosome_w_sisters.length), dtype=np.int64)
    holding_arr[:] = -1
    empty_row = np.zeros(mock_chromosome_w_sisters.length, dtype=np.int64)
    empty_row[:] = 1

    empty_tetrad = ChromosomeTetrad(
        length=mock_chromosome_w_sisters.length,
        selectSistersForCO=SequentialSisterSelector(),
        selectSistersForPool=selectSingleSisterForPool,
    )
    for strand in empty_tetrad.as_strands():
        strand.base_haplotype = Haplotype.PINK
    mock_chromosome_w_sisters.selectSistersForCO = SequentialSisterSelector()
    mock_chromosome_w_sisters.insert_crossover(42)
    mock_chromosome_w_sisters.insert_crossover(339)
    mock_chromosome_w_sisters.into_haplotype_tbl(holding_arr[4:8])

    haptbl = HaplotypeTable.from_chromosomes(
        [empty_tetrad, mock_chromosome_w_sisters, empty_tetrad, empty_tetrad]
    )

    for i in range(4):
        assert np.array_equal(haptbl.tbl[i], empty_row)
    assert np.array_equal(
        haptbl.tbl[4],
        np.array(
            [mock_chromosome_w_sisters.a1.base_haplotype.value]
            * mock_chromosome_w_sisters.length
        ),
    )
    assert np.array_equal(
        haptbl.tbl[5],
        np.array(
            [mock_chromosome_w_sisters.a2.base_haplotype.value] * 339
            + [Haplotype(not mock_chromosome_w_sisters.a2.base_haplotype).value]
            * (mock_chromosome_w_sisters.length - 339)
        ),
    )
    assert np.array_equal(
        haptbl.tbl[6],
        np.array(
            [mock_chromosome_w_sisters.b2.base_haplotype.value] * 339
            + [Haplotype(not mock_chromosome_w_sisters.b2.base_haplotype).value]
            * (mock_chromosome_w_sisters.length - 339)
        ),
    )
    assert np.array_equal(
        haptbl.tbl[7],
        np.array(
            [mock_chromosome_w_sisters.b2.base_haplotype.value]
            * mock_chromosome_w_sisters.length
        ),
    )
    for i in range(8, 16):
        assert np.array_equal(haptbl.tbl[i], empty_row)


def test_haptbl_from_legacy_chroms(
    mock_legacy_chrom: LegacyChromosome,
    mock_legacy_chrom_single_co: LegacyChromosome,
):
    empty_legacy_chrom = LegacyChromosome(length=mock_legacy_chrom.length)
    tester_chroms = [
        empty_legacy_chrom,
        mock_legacy_chrom_single_co,
        mock_legacy_chrom,
        empty_legacy_chrom,
    ]
    empty_row = np.zeros(mock_legacy_chrom.length, dtype=np.int64)

    expected_1co_tbl_result = np.zeros(
        mock_legacy_chrom_single_co.length, dtype=np.int64
    )
    expected_1co_tbl_result[0:81] = 0
    expected_1co_tbl_result[81:] = 1
    expected_2co_tbl_result = np.zeros(mock_legacy_chrom.length, dtype=np.int64)
    expected_2co_tbl_result[0:57] = 0
    expected_2co_tbl_result[57:293] = 1
    expected_2co_tbl_result[293:] = 0

    haptbl = HaplotypeTable.from_chromosomes(tester_chroms)

    assert np.array_equal(haptbl.tbl[0], empty_row)
    assert np.array_equal(haptbl.tbl[1], expected_1co_tbl_result)
    assert np.array_equal(haptbl.tbl[2], expected_2co_tbl_result)
    assert np.array_equal(haptbl.tbl[3], empty_row)
