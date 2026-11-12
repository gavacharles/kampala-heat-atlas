"""Carry over work already done in other repositories (no recomputation).

Copies the datasets into DATA/carried/ and writes a provenance manifest, so every
reused layer can be cited back to the repo, script and paper it came from.
"""
import json
import shutil
import subprocess
from datetime import date
from pathlib import Path

from kampala.config import DATA, OUTPUTS

RAIN = Path.home() / "uganda-rainfall-climate"
FIRE = "gavacharles/kampala-cbd-fire-safety"
OUT = DATA / "carried"

LOCAL = {
    # file in uganda-rainfall-climate : (what it is, producing script, used in module)
    "data/kd/sat_lst.nc": ("Landsat 8/9 dry-season LST + NDVI, early 2014-16 / late 2023-25, 30 m", "src/kd_heat.py + kd_satellite.py", "M2,M3"),
    "data/kd/sat_modis_lst.nc": ("MODIS day/night LST medians 2013-16 / 2022-25, 1 km", "src/kd_heat.py", "M2,M3"),
    "data/kd/sat_iolulc.nc": ("Impact Observatory 10 m LULC 2017-2023", "src/kd_satellite.py", "M1 (reference)"),
    "data/processed/kampala_ghsl_built.nc": ("JRC GHSL built-up surface 1975-2030 (100 m)", "src/fetch_kampala.py", "M1 (long-term context)"),
    "data/processed/kampala_ghs_pop.nc": ("JRC GHS-POP 1975-2030 (100 m)", "src/fetch_kampala.py", "M3,M4"),
    "data/processed/kampala_catchments.tif": ("Catchments + HAND from D8 routing (30 m)", "src/kampala_catchments.py", "M3 (valley vs hill), M4"),
    "data/processed/kampala_worldcover.tif": ("ESA WorldCover 2021 (10 m)", "src/fetch_kampala.py", "M1 (reference labels)"),
    "data/processed/kampala_waterways.geojson": ("OSM drainage channels", "src/fetch_kampala.py", "M3,M4"),
    "results/kd_heat_summary.csv": ("P3 heat-flood exposure results", "src/kd_heat.py", "M3 (comparison)"),
    "results/kd_heat_regression.csv": ("P3 LST regression (low ground vs hills)", "src/kd_heat.py", "M3 (comparison)"),
    "results/kd_heat_by_density.csv": ("P3 heat by built-up quintile", "src/kd_heat.py", "M3 (comparison)"),
    "results/kd_lulc_valley.csv": ("Valley-floor built/wetland change 2017-23", "src/kd_satellite.py", "M1 (comparison)"),
}
GITHUB = {
    "data/gis/building_heights.csv": ("Open Buildings 2.5D p90 height per footprint, 2016 & 2023, CBD", "scripts/gee/01_building_heights.py", "M0 (height calibration)"),
    "data/gis/building_heights_yearly.csv": ("p90 height per footprint, yearly 2016-2023, CBD", "scripts/gee/04_extra_layers.py", "M0"),
    "data/gis/height_validation.csv": ("Height vs OSM building:levels (r = 0.86)", "scripts/gis/04_heights_standards.py", "M0 (calibration)"),
    "data/gis/landsat_lst_dry.tif": ("Landsat LST dry-season median 2019-25, CBD", "scripts/gee/03_remote_sensing.py", "M2 (CBD check)"),
    "data/gis/lst_summary.json": ("CBD LST summary", "scripts/gee/03_remote_sensing.py", "M2"),
}

manifest = []
for rel, (desc, script, module) in LOCAL.items():
    src = RAIN / rel
    dst = OUT / "uganda-rainfall-climate" / Path(rel).name
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    manifest.append(dict(file=str(dst.relative_to(DATA)), source_repo="uganda-rainfall-climate", source_path=rel,
                         script=script, description=desc, used_in=module))
for rel, (desc, script, module) in GITHUB.items():
    dst = OUT / "kampala-cbd-fire-safety" / Path(rel).name
    dst.parent.mkdir(parents=True, exist_ok=True)
    raw = subprocess.run(["gh", "api", f"repos/{FIRE}/contents/{rel}", "-H", "Accept: application/vnd.github.raw"],
                         capture_output=True, check=True).stdout
    dst.write_bytes(raw)
    manifest.append(dict(file=str(dst.relative_to(DATA)), source_repo="kampala-cbd-fire-safety", source_path=rel,
                         script=script, description=desc, used_in=module))

(OUTPUTS / "tables").mkdir(parents=True, exist_ok=True)
(OUTPUTS / "tables" / "carried_over_manifest.json").write_text(
    json.dumps({"date": str(date.today()), "items": manifest}, indent=2))
for m in manifest:
    print(f"{m['used_in']:28s} {m['file']}")
