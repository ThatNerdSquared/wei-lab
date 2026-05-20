from croissant.chromosomes.chromatid import (
    Chromatid as Chromatid,
    convert_chromatid_to_haplotype_tbl as convert_chromatid_to_haplotype_tbl,
)
from croissant.chromosomes.chromosome_set import (
    ChromosomeSet as ChromosomeSet,
    ChromsetMaker as ChromsetMaker,
    make_chromset_all_sisters_in_pool as make_chromset_all_sisters_in_pool,
    make_chromset_custom_sister_selectors as make_chromset_custom_sister_selectors,
    make_chromset_haplotype_switching_only as make_chromset_haplotype_switching_only,
    make_chromset_with_legacy_chromosome as make_chromset_with_legacy_chromosome,
    make_default_chromset as make_default_chromset,
    pool_iter as pool_iter,
    pool_size as pool_size,
)
from croissant.chromosomes.chromosome_tetrad import (
    ChromosomeTetrad as ChromosomeTetrad,
    SequentialSisterSelector as SequentialSisterSelector,
    haplotypeSwitchingCOsOnlySelector as haplotypeSwitchingCOsOnlySelector,
    probabilistic_selector as probabilistic_selector,
    selectMostRecombinantForPool as selectMostRecombinantForPool,
    selectOnlyRecombinantsForPool as selectOnlyRecombinantsForPool,
    selectSingleSisterForPool as selectSingleSisterForPool,
)
from croissant.chromosomes.core import ChromosomeLike
from croissant.chromosomes.legacy_chromosome import LegacyChromosome as LegacyChromosome

CHROM_TYPES: list[type[ChromosomeLike]] = [
    Chromatid,
    ChromosomeTetrad,
    LegacyChromosome,
]
CROISSANT_CHROM_MAPPER: dict[str, type[ChromosomeLike]] = {
    x.__name__: x for x in CHROM_TYPES
}
