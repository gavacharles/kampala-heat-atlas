"""Load project configuration and resolve project paths."""
from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
OUTPUTS = ROOT / "outputs"


@lru_cache
def load() -> dict:
    with open(ROOT / "config" / "config.yaml") as f:
        return yaml.safe_load(f)


# Large rasters live outside OneDrive (see config: paths.data_root).
DATA = Path(load()["paths"]["data_root"]).expanduser()
for sub in ("raw", "interim", "processed", "carried"):
    (DATA / sub).mkdir(parents=True, exist_ok=True)
FIGURES = Path(load()["paths"]["figures_root"]).expanduser()
FIGURES.mkdir(parents=True, exist_ok=True)
