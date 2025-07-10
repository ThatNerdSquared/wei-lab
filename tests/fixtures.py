import random
from typing import Callable

import pytest

from src.utils.chromatid import Chromatid, Crossover, CrossoverStack, Haplotype
from src.utils.chromosome_with_sister_chromatids import (
    ChromosomeSisterChromatids,
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
def mock_chromosome_w_sisters() -> ChromosomeSisterChromatids:
    return ChromosomeSisterChromatids(
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
