"""Module 0 (new parts): citywide FABDEM terrain and 2.5D building metrics.

Carried over, not recomputed: CBD footprint heights 2016-2023 and the p90 calibration
(kampala-cbd-fire-safety), HAND/catchments (uganda-rainfall-climate).
"""
from kampala import cube, ee_utils
from kampala.config import DATA

ee_utils.init()
ext = ee_utils.extended_area()
out = DATA / "processed" / "m0"

ee_utils.download(cube.terrain(ext), out / "terrain_fabdem_30m.tif", ext, 30)

for year in (2016, 2020, 2023):   # Open Buildings 2.5D years nearest the 2017/2021/2025 epochs
    ee_utils.download(cube.building_metrics(year, ext), out / f"buildings_{year}_30m.tif", ext, 30,
                      tile_mb=2, tile_dim=128)
