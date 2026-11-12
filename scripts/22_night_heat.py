"""Night-time heat by urban form (Paper 1): MODIS Aqua 1 km, ~01:30 (night) and ~13:30 (day),
dry-season medians for 2017 / 2021 / 2025 (three-year windows, as for Landsat).

Each 1 km MODIS cell gets the area shares of: packed low-rise, other dense, SDI 2014 informal
settlements, built-up (M0), tree cover (M1), water/wetland (M1) and its mean FABDEM elevation.
Cells with > 10 % water are dropped (the lake is the warmest night surface and would dominate).
1. group means = share-weighted mean anomaly (anomaly = cell minus land mean of that pass)
2. OLS: anomaly ~ packed + other_dense + sdi + built + tree + elevation (per pass and epoch),
   reported as degrees C per +10 % share, with HC1 standard errors.
"""
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import statsmodels.api as sm
from rasterio.features import rasterize
from rasterio.warp import Resampling, reproject

from kampala import config
from kampala.config import DATA, OUTPUTS

P = DATA / "processed"
EP = config.load()["epochs"]
BY = {"t1": 2016, "t2": 2020, "t3": 2023}


def onto(src_arr, src_tr, src_crs, prof, how=Resampling.average):
    out = np.zeros((prof["height"], prof["width"]), "float32")
    reproject(src_arr.astype("float32"), out, src_transform=src_tr, src_crs=src_crs, dst_transform=prof["transform"],
              dst_crs=prof["crs"], resampling=how)
    return out


sdi = gpd.read_file(P / "m3" / "sdi_2014_settlements.gpkg")
rows, coefs = [], []
for ep, spec in EP.items():
    y = spec["year"]
    with rasterio.open(P / "m2" / f"lst_modis_{ep}_{y}_1km.tif") as r:
        day, night, prof = r.read(1), r.read(2), r.profile
    with rasterio.open(P / "m3" / f"settlement_groups_{ep}_{y}_90m.tif") as r:
        g, gt, gc = r.read(1), r.transform, r.crs
    packed = onto(g == 1, gt, gc, prof); dense = onto(g == 2, gt, gc, prof)
    sdi10 = rasterize(((geom, 1) for geom in sdi.to_crs(gc).geometry), out_shape=g.shape, transform=gt, fill=0, dtype="uint8")
    sdis = onto(sdi10, gt, gc, prof)
    with rasterio.open(P / "m0" / f"buildings_{BY[ep]}_30m.tif") as r:
        built = onto(r.read(1, masked=True).filled(0), r.transform, r.crs, prof)
    with rasterio.open(P / "m1" / f"landcover_{ep}_{y}_10m.tif") as r:
        lc = r.read(1)[::3, ::3]; t30 = r.transform * rasterio.Affine.scale(3)
        tree = onto(lc == 3, t30, r.crs, prof); water = onto(np.isin(lc, [6, 7]), t30, r.crs, prof)
    with rasterio.open(P / "m0" / "terrain_fabdem_30m.tif") as r:
        elev = onto(r.read(1), r.transform, r.crs, prof)
    ok = np.isfinite(day) & np.isfinite(night) & (water < 0.10)
    for nm, arr in (("night", night), ("day", day)):
        anom = arr - np.nanmean(arr[ok])
        for gname, w in (("packed low-rise", packed), ("other dense", dense), ("SDI informal (2014)", sdis),
                         ("non-built", np.clip(1 - built * 3, 0, 1))):
            m = ok & (w > 0)
            rows.append(dict(year=y, pass_=nm, group=gname, weighted_anom=float(np.sum(anom[m] * w[m]) / np.sum(w[m])),
                             effective_km2=float(w[m].sum())))
        X = pd.DataFrame({"packed": packed[ok], "other_dense": dense[ok], "sdi": sdis[ok], "built": built[ok],
                          "tree": tree[ok], "elev_100m": elev[ok] / 100})
        fit = sm.OLS(anom[ok], sm.add_constant(X)).fit(cov_type="HC1")
        for k in ("packed", "other_dense", "sdi", "built", "tree"):
            coefs.append(dict(year=y, pass_=nm, term=k, per_10pct=fit.params[k] / 10, se=fit.bse[k] / 10,
                              p=fit.pvalues[k], r2=fit.rsquared, n=int(ok.sum())))
        coefs.append(dict(year=y, pass_=nm, term="elev_100m", per_10pct=fit.params["elev_100m"], se=fit.bse["elev_100m"],
                          p=fit.pvalues["elev_100m"], r2=fit.rsquared, n=int(ok.sum())))

g = pd.DataFrame(rows); c = pd.DataFrame(coefs)
g.to_csv(OUTPUTS / "tables" / "m3_night_by_group.csv", index=False)
c.to_csv(OUTPUTS / "tables" / "m3_night_regression.csv", index=False)
w = g.pivot_table(index=["pass_", "group"], columns="year", values="weighted_anom")
print(w.round(2)); print("\nchange 2017->2025:\n", (w[2025] - w[2017]).round(2))
print(c.pivot_table(index=["pass_", "term"], columns="year", values="per_10pct").round(3))
