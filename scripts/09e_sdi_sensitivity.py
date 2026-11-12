"""Sensitivity of the SDI-reference results to georeferencing error: shift every polygon by
250 m in 8 directions (plus no shift) and recompute agreement and heat."""
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from shapely.affinity import translate

from kampala.config import DATA, OUTPUTS

P = DATA / "processed"
sdi = gpd.read_file(P / "m3" / "sdi_2014_settlements.gpkg")
with rasterio.open(P / "m2" / "lst_landsat_t3_2025_30m.tif") as r:
    a25, tr = r.read(2, masked=True).filled(np.nan), r.transform
with rasterio.open(P / "m2" / "lst_landsat_t1_2017_30m.tif") as r:
    a17 = r.read(2, masked=True).filled(np.nan)
with rasterio.open(P / "m0" / "buildings_2023_30m.tif") as r:
    bf = r.read(1, masked=True).filled(0)
with rasterio.open(P / "m3" / "settlement_groups_t3_2025_90m.tif") as r:
    g90, tr90 = r.read(1), r.transform
hull = sdi.union_all().convex_hull.buffer(2000)        # comparison area: the mapped part of Kampala
area30 = rasterize([(hull, 1)], out_shape=a25.shape, transform=tr, fill=0, dtype="uint8").astype(bool)
area90 = rasterize([(hull, 1)], out_shape=g90.shape, transform=tr90, fill=0, dtype="uint8").astype(bool)
rows = []
for dx, dy in [(0, 0)] + [(250 * np.cos(a), 250 * np.sin(a)) for a in np.radians(np.arange(0, 360, 45))]:
    geoms = [translate(g, dx, dy) for g in sdi.geometry]
    ref = rasterize(((g, 1) for g in geoms), out_shape=a25.shape, transform=tr, fill=0, dtype="uint8").astype(bool)
    ref90 = rasterize(((g, 1) for g in geoms), out_shape=g90.shape, transform=tr90, fill=0, dtype="uint8").astype(bool)
    dense = area30 & ~ref & (bf >= 0.35)
    ours = (g90 == 1) & area90
    rows.append(dict(shift_m=round(np.hypot(dx, dy)), dir_deg=round(np.degrees(np.arctan2(dy, dx))) % 360,
                     sdi_anom_2025=np.nanmean(a25[ref]), dense_anom_2025=np.nanmean(a25[dense]),
                     sdi_change=np.nanmean(a25[ref]) - np.nanmean(a17[ref]), dense_change=np.nanmean(a25[dense]) - np.nanmean(a17[dense]),
                     ref_flagged_informal=(ours & ref90).sum() / ref90.sum(),
                     ours_inside_ref=(ours & ref90).sum() / max(ours.sum(), 1),
                     ref_share_of_area=ref90[area90].mean()))
df = pd.DataFrame(rows)
df.to_csv(OUTPUTS / "tables" / "m3_sdi_sensitivity.csv", index=False)
print(df.round(3).to_string(index=False))
print("\nrange over shifts:"); print(df.iloc[1:].agg(["min", "max"]).round(3).T)
