import pytest

from src.utils.chromatid import Chromatid, Crossover, CrossoverStack, Haplotype


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
