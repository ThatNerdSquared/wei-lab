import random
from typing import Callable

import pytest

from croissant.chromosomes.chromatid import (
    Chromatid,
    Crossover,
    CrossoverStack,
    Haplotype,
)
from croissant.chromosomes.chromosome_set import ChromosomeSet
from croissant.chromosomes.chromosome_tetrad import (
    ChromosomeTetrad,
    SequentialSisterSelector,
    selectAllSistersForPool,
    selectSingleSisterForPool,
    uniformlyRandomSisterSelector,
)


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
