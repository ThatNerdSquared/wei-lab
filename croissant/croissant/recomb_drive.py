from croissant.chromosomes import (
    ChromosomeSet,
    ChromosomeTetrad,
    selectSingleSisterForPool,
)
from croissant.chromosomes.core import Strand
from croissant.recomb_sim import Selector, downsample_probabilistic


def make_default_drive_downsamples(
    chrom_set: ChromosomeSet[ChromosomeTetrad],
    drive_type: str,
    drive_selector: Selector,
) -> list[ChromosomeSet[Strand]]:
    res: list[ChromosomeSet[Strand]] = []
    for level in [0.25, 0.5, 0.75, 1.0]:
        res.append(
            downsample_probabilistic(
                chrom_set,
                drive_selector,
                selectSingleSisterForPool,
                p1=level,
                new_label=f'{drive_type} ({int(level * 100)}%)',
            )
        )
    return res
