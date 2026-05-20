import random
from datetime import datetime
from typing import Callable

import pytest

from croissant.chromosomes import LegacyChromosome
from croissant.chromosomes.chromatid import Chromatid
from croissant.chromosomes.chromosome_set import ChromosomeSet
from croissant.chromosomes.chromosome_tetrad import (
    ChromosomeTetrad,
    SequentialSisterSelector,
    selectAllSistersForPool,
    selectSingleSisterForPool,
    uniformlyRandomSisterSelector,
)
from croissant.chromosomes.core import (
    ChromosomeLike,
    Crossover,
    CrossoverStack,
    Haplotype,
)
from croissant.recomb_sim import RecombSim


@pytest.fixture
def mock_legacy_chrom() -> LegacyChromosome:
    res = LegacyChromosome(400)
    res.insert_crossover(57)
    res.insert_crossover(293)
    return res


@pytest.fixture
def mock_legacy_chrom_single_co() -> LegacyChromosome:
    res = LegacyChromosome(400)
    res.insert_crossover(81)
    return res


@pytest.fixture
def mock_crossovers() -> list[Crossover]:
    return [
        Crossover(10, Haplotype.BLUE),
        Crossover(39, Haplotype.PINK),
        Crossover(42, Haplotype.PINK),
        Crossover(400, Haplotype.BLUE),
    ]


@pytest.fixture
def mock_co_stack(mock_crossovers: list[Crossover]) -> CrossoverStack:
    return CrossoverStack(list(reversed(mock_crossovers)))


@pytest.fixture
def mock_chromatid(mock_crossovers: list[Crossover]) -> Chromatid:
    return Chromatid(
        base_haplotype=Haplotype.PINK, length=500, crossovers=mock_crossovers
    )


@pytest.fixture
def mock_chromatid_empty() -> Chromatid:
    return Chromatid(base_haplotype=Haplotype.PINK, length=500)


@pytest.fixture
def mock_chromosome_w_sisters() -> ChromosomeTetrad:
    return ChromosomeTetrad(
        length=500,
        selectSistersForCO=uniformlyRandomSisterSelector,
        selectSistersForPool=selectSingleSisterForPool,
    )


type ResetRandomCallback = Callable[[], None]


@pytest.fixture
def reset_random_state() -> ResetRandomCallback:
    random.seed(77777)
    starting_state = random.getstate()
    return lambda: random.setstate(starting_state)


@pytest.fixture
def mock_chromset() -> ChromosomeSet[ChromosomeTetrad]:
    return ChromosomeSet(
        label='test',
        num_chroms=20,
        chrom_length=50,
        chrom_maker=lambda x: ChromosomeTetrad(
            length=x,
            selectSistersForCO=SequentialSisterSelector(0),
            selectSistersForPool=selectAllSistersForPool,
        ),
    )


def set_recomb_sim_date_to_newyrs[C: ChromosomeLike](sim: RecombSim[C]):
    """
    Sets the given RecombSim's generated time to January 1st, 2000,
    14:50:51.123456.

    NOT TO BE USED OUTSIDE OF UNIT TESTS!!
    """
    sim.time_generated = datetime.fromisoformat('2000-01-01 14:50:51.123456')
