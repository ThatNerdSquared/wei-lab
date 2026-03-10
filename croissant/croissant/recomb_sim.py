from dataclasses import InitVar, asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import tomli_w

from croissant.chromosomes import ChromosomeLike

_CURR_SIM_VERSION = 0
SIM_FILE_EXT = '.recombsim'


@dataclass(slots=True, kw_only=True)
class RecombSim[C: ChromosomeLike]:
    label: str
    sim_version: int = field(init=False, default=_CURR_SIM_VERSION)
    description: str
    time_generated: datetime = field(init=False, default_factory=datetime.now)
    num_chroms: int
    chrom_length: int
    rand_seed: int
    chroms: list[C] = field(init=False)
    chrom_type: C = field(init=False)
    template_chrom: InitVar[C]

    _slugify_translate_tbl: ClassVar[dict[int, int | None]] = str.maketrans(
        ' ', '_', '()[]'
    )

    def __post_init__(
        self,
        template_chrom: C,
    ):
        if self.chrom_length != template_chrom.length:
            raise ValueError(
                f'Provided template chromosome of length {template_chrom.length} does not match specified chromsome length {self.chrom_length}'  # noqa: E501
            )
        self.chrom_type = template_chrom.__class__
        self.chroms = template_chrom.into_list(self.num_chroms)

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

    def get_slug(self) -> str:
        return self.label.translate(self._slugify_translate_tbl).strip().lower()

    def to_file(self, at_location: Path | None = None) -> Path:
        folder_to_write_to = at_location if at_location is not None else Path.cwd()
        if not folder_to_write_to.is_dir():
            raise NotADirectoryError(
                'Provided location is not a directory or does not exist!'
            )
        fp = (folder_to_write_to / self.get_slug()).with_suffix(SIM_FILE_EXT)
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

    def from_file(self):
        pass
