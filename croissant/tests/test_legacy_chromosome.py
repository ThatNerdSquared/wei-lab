from croissant.chromosomes.core import Haplotype
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


def test_legacy_chrom_haplotype_at(mock_legacy_chrom: LegacyChromosome):
    assert mock_legacy_chrom.haplotype_at(10) == Haplotype.BLUE
    assert mock_legacy_chrom.haplotype_at(57) == Haplotype.PINK
    assert mock_legacy_chrom.haplotype_at(199) == Haplotype.PINK
    assert mock_legacy_chrom.haplotype_at(292) == Haplotype.PINK
    assert mock_legacy_chrom.haplotype_at(293) == Haplotype.BLUE
