import pytest

from src.utils.chromatid import Chromatid, make_default_chromatid
from src.utils.chromosome_set import ChromosomeSet
from src.utils.chromosome_with_sister_chromatids import (
    ChromosomeSisterChromatids,
    SequentialSisterSelector,
    make_default_chrom_w_sisters,
    selectSingleSisterForPool,
)


def test_chromset_iterator_when_empty():
    test_chromset = ChromosomeSet(
        label='test',
        num_chroms=0,
        chrom_length=42,
        chrom_maker=make_default_chrom_w_sisters,
    )
    chromset_iter = iter(test_chromset)
    with pytest.raises(StopIteration):
        _ = next(chromset_iter)


def test_chromset_iterator_chromatids():
    test_chromset = ChromosomeSet(
        label='test',
        num_chroms=3,
        chrom_length=42,
        chrom_maker=make_default_chromatid,
    )
    chromset_iter = iter(test_chromset)
    assert test_chromset.chroms == list(chromset_iter)
    chromset_iter = iter(test_chromset)
    _ = next(chromset_iter)
    _ = next(chromset_iter)
    _ = next(chromset_iter)
    with pytest.raises(StopIteration):
        _ = next(chromset_iter)


def test_chromset_iterator_chroms_w_sisters():
    test_chromset = ChromosomeSet(
        label='test',
        num_chroms=3,
        chrom_length=42,
        chrom_maker=lambda x: ChromosomeSisterChromatids(
            length=x,
            selectSistersForCO=SequentialSisterSelector(),
            selectSistersForPool=selectSingleSisterForPool,
        ),
    )
    chromset_iter = iter(test_chromset)
    assert test_chromset.chroms != list(chromset_iter)
    chromset_iter = iter(test_chromset)

    assert isinstance(next(chromset_iter), Chromatid)
    assert isinstance(next(chromset_iter), Chromatid)
    assert isinstance(next(chromset_iter), Chromatid)
    with pytest.raises(StopIteration):
        _ = next(chromset_iter)
