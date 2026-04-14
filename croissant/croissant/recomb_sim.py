import tomllib
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import tomli_w

from croissant._fmt_utils import assemble_fp
from croissant.chromosomes import CROISSANT_CHROM_MAPPER, ChromosomeLike

_CURR_SIM_VERSION = 0
SIM_FILE_EXT = '.recombsim'


def _assemble_simfile_path(label: str, at_location: Path | None = None) -> Path:
    return assemble_fp(label, SIM_FILE_EXT, at_location)


@dataclass(slots=True, kw_only=True)
class RecombSim[C: ChromosomeLike]:
    label: str
    sim_version: int
    description: str
    time_generated: datetime
    num_chroms: int
    chrom_length: int
    rand_seed: int
    chroms: Sequence[C]
    chrom_type: type[C]

    @staticmethod
    def _dictify(x: list[tuple[Any, Any]]) -> dict[str, Any]:
        res: dict[str, Any] = {}
        for k, v in x:
            if k == 'chroms':
                continue
            if k == 'chrom_type':
                res['chrom_type'] = v.__name__
                continue
            res[k] = v
        return res

    @classmethod
    def from_template_chrom(
        cls: type[RecombSim[ChromosomeLike]],
        *,
        label: str,
        description: str,
        num_chroms: int,
        chrom_length: int,
        rand_seed: int,
        template_chrom: ChromosomeLike,
    ) -> RecombSim[ChromosomeLike]:
        if chrom_length != template_chrom.length:
            raise ValueError(
                f'Provided template chromosome of length {template_chrom.length} does not match specified chromsome length {chrom_length}'  # noqa: E501
            )
        return cls(
            label=label,
            description=description,
            num_chroms=num_chroms,
            chrom_length=chrom_length,
            rand_seed=rand_seed,
            sim_version=_CURR_SIM_VERSION,
            time_generated=datetime.now(),
            chrom_type=template_chrom.__class__,
            chroms=template_chrom.into_list(num_chroms),
        )

    @classmethod
    def from_file(cls, label: str, at_location: Path | None = None):
        fp = _assemble_simfile_path(label, at_location)
        if not fp.exists():
            raise FileNotFoundError(f'No recomb sim file found at {fp}')
        stringified_metadata = ''
        with fp.open() as simfile:
            while True:
                curr = simfile.readline()
                if curr == '':
                    raise RuntimeError(
                        f'Recomb sim file "{fp}" malformed; could not find end of metadata.'  # noqa: E501
                    )
                if curr == '# \n':
                    break
                stringified_metadata += curr[2:]

        metadata = tomllib.loads(stringified_metadata)
        chrom_type = CROISSANT_CHROM_MAPPER[metadata['chrom_type']]

        stringified_chroms_buffer = np.loadtxt(fp, dtype=np.int64, ndmin=2)
        chroms: list[ChromosomeLike] = []
        buffer_length: int = stringified_chroms_buffer.shape[0]
        for idx in range(0, buffer_length, chrom_type.num_strands):
            chroms.append(
                chrom_type.from_haplotype_tbl(
                    stringified_chroms_buffer[idx : idx + chrom_type.num_strands]
                )
            )

        return RecombSim(**metadata, chroms=chroms)  # pyright: ignore[reportAny]

    def to_file(self, at_location: Path | None = None) -> Path:
        fp = _assemble_simfile_path(self.label, at_location)
        metadata = asdict(self, dict_factory=self._dictify)
        stringified_metadata = tomli_w.dumps(metadata)

        buffer_length = self.num_chroms * self.chrom_type.num_strands
        stringified_chroms_buffer = np.zeros(
            (buffer_length, self.chrom_length), dtype=np.int64
        )
        for idx in range(0, buffer_length, self.chrom_type.num_strands):
            self.chroms[int(idx / self.chrom_type.num_strands)].into_haplotype_tbl(
                stringified_chroms_buffer[idx : idx + self.chrom_type.num_strands]
            )

        np.savetxt(
            fp, stringified_chroms_buffer, header=stringified_metadata, fmt='%0d'
        )
        return fp
