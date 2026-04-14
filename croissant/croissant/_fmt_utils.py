from pathlib import Path

_slugify_translate_tbl = str.maketrans(' ', '_', '()[]')


def slugify(label: str) -> str:
    return label.translate(_slugify_translate_tbl).strip().lower()


def assemble_fp(label: str, file_ext: str, at_location: Path | None = None) -> Path:
    folder_to_write_to = at_location if at_location is not None else Path.cwd()
    if not folder_to_write_to.is_dir():
        raise NotADirectoryError(
            f'Provided location "{folder_to_write_to}" is not a directory or does not exist!'  # noqa: E501
        )
    return (folder_to_write_to / slugify(label)).with_suffix(file_ext)
