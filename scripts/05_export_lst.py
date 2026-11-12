"""Module 2 (new parts): thermal inputs for the three thesis epochs.

Carried over (not recomputed): Landsat LST 2014-16 / 2023-25 and MODIS LST 2013-16 / 2022-25
(uganda-rainfall-climate), CBD LST 2019-25 (kampala-cbd-fire-safety).
New: gap-free calibrated Landsat LST + anomaly for 2017 / 2021 / 2025, MODIS day/night for the same
windows, global LCZ map. Calibration coefficients -> outputs/tables/m2_lst_calibration.csv
"""
import ee
import pandas as pd

from kampala import config, ee_utils, landsat
from kampala.config import DATA, OUTPUTS

ee_utils.init()
ext = ee_utils.extended_area()
out = DATA / "processed" / "m2"
rows = []
for ep, spec in config.load()["epochs"].items():
    img = landsat.lst_composite(ep, ext)
    props = img.toDictionary(["calib_intercept", "calib_slope", "area_mean_LST"]).getInfo()
    both = img.select(["LST", "LST_st"])
    r = both.reduceRegion(ee.Reducer.pearsonsCorrelation(), ext, 300, maxPixels=1e10).getInfo()["correlation"]
    d = both.select("LST").subtract(both.select("LST_st")).pow(2).reduceRegion(ee.Reducer.mean(), ext, 300).getInfo()["LST"]
    cov = img.select("LST_st").mask().reduceRegion(ee.Reducer.mean(), ext, 300).getInfo()["LST_st"]
    rows.append(dict(epoch=ep, year=spec["year"], **props, r_vs_C2ST=r, rmse_vs_C2ST=d ** 0.5, C2ST_coverage=cov))
    print(rows[-1], flush=True)
    ee_utils.download(img, out / f"lst_landsat_{ep}_{spec['year']}_30m.tif", ext, 30, min_valid=0.97)
    ee_utils.download(landsat.modis_lst(ep, ext), out / f"lst_modis_{ep}_{spec['year']}_1km.tif", ext, 1000)
pd.DataFrame(rows).to_csv(OUTPUTS / "tables" / "m2_lst_calibration.csv", index=False)
ee_utils.download(landsat.lcz(ext), out / "lcz_demuzere_100m.tif", ext, 100, dtype="uint8")
