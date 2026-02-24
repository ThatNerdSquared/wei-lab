# `[WIP]` wei-lab

## libs
- `croissant`
- `analyses` - collection of python scripts

## usage
- `analyses` uses same deps and tooling as `croissant` - installing
  `croissant[dev]` is sufficient:
```bash
python3 -m venv .venv
source .venv/bin/activate
make deps
```
- main script files are in `analyses`, all are standalone and should be runnable
  via `python3 analyses/<script_name>.py` once `croissant` has been installed
