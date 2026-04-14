import filecmp
from pathlib import Path

from croissant.chromosomes import LegacyChromosome, insert_crossover
from croissant.chromosomes.chromatid import Chromatid, Haplotype
from croissant.chromosomes.chromosome_tetrad import (
    ChromosomeTetrad,
    SequentialSisterSelector,
    selectSingleSisterForPool,
)
from croissant.recomb_sim import RecombSim
from tests.fixtures import set_recomb_sim_date_to_newyrs


def test_dump_recomb_sim_legacy_chrom(tmp_path: Path):
    test_recomb_sim = RecombSim.from_template_chrom(
        label='test sim (legacy chroms)',
        description='for use in unittests only - no meaningful biological interpretation.',  # noqa: E501
        num_chroms=10,
        chrom_length=13,
        rand_seed=999,
        template_chrom=LegacyChromosome(length=13),
    )
    set_recomb_sim_date_to_newyrs(test_recomb_sim)
    insert_crossover(test_recomb_sim.chroms[6], 9)
    given_output_fp = test_recomb_sim.to_file(tmp_path)
    expected_output_fp = Path.cwd() / Path(
        'croissant/tests/fakesims/test_sim_legacy_chroms.recombsim'
    )
    assert filecmp.cmp(expected_output_fp, given_output_fp)


def test_dump_recomb_sim_chromatids(tmp_path: Path):
    test_recomb_sim = RecombSim.from_template_chrom(
        label='test sim (chromatids)',
        description='for use in unittests only - no meaningful biological interpretation.',  # noqa: E501
        num_chroms=10,
        chrom_length=42,
        rand_seed=999,
        template_chrom=Chromatid(base_haplotype=Haplotype.BLUE, length=42),
    )
    set_recomb_sim_date_to_newyrs(test_recomb_sim)
    insert_crossover(test_recomb_sim.chroms[6], 9)
    given_output_fp = test_recomb_sim.to_file(tmp_path)
    expected_output_fp = Path.cwd() / Path(
        'croissant/tests/fakesims/test_sim_chromatids.recombsim'
    )
    assert filecmp.cmp(expected_output_fp, given_output_fp)


def test_dump_recomb_sim_tetrads(tmp_path: Path):
    test_recomb_sim = RecombSim.from_template_chrom(
        label='test sim (tetrads)',
        description='for use in unittests only - no meaningful biological interpretation.',  # noqa: E501
        num_chroms=11,
        chrom_length=23,
        rand_seed=999,
        template_chrom=ChromosomeTetrad(
            length=23,
            selectSistersForCO=SequentialSisterSelector(0),
            selectSistersForPool=selectSingleSisterForPool,
        ),
    )
    set_recomb_sim_date_to_newyrs(test_recomb_sim)
    insert_crossover(test_recomb_sim.chroms[6], 9)
    given_output_fp = test_recomb_sim.to_file(tmp_path)
    expected_output_fp = Path.cwd() / Path(
        'croissant/tests/fakesims/test_sim_tetrads.recombsim'
    )
    assert filecmp.cmp(expected_output_fp, given_output_fp)


# def test_read_recomb_sim_legacy_chrom():
#     expected_fp = Path.cwd() / Path('croissant/tests/fakesims/')
#     retrieved_sim = RecombSim.from_file('test sim (legacy chroms)', expected_fp)
#     print(retrieved_sim)
#     assert False
