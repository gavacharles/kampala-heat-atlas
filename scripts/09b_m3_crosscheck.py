"""Module 3 cross-check: informal-type morphology from Overture footprints (independent of the
Open Buildings 2.5D counts) vs the M0-based rule for 2025, and a plausibility check against
well-known informal neighbourhoods (locations from Overture divisions).
"""
import duckdb
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from scipy import ndimage
from shapely import wkb

from kampala.config import DATA, OUTPUTS

M3 = DATA / "processed" / "m3"
RULE = dict(built=0.35, footprint=80.0, density=30.0, height=8.0, min_cells=3)

with rasterio.open(M3 / "settlement_groups_t3_2025_90m.tif") as r:
    grp, prof = r.read(1), r.profile
T, H, W = prof["transform"], prof["height"], prof["width"]

b = gpd.read_parquet(DATA / "raw" / "overture_buildings.parquet").to_crs(prof["crs"])
b = b[b.geometry.type.isin(["Polygon", "MultiPolygon"])]
area = b.geometry.area.values
c = b.geometry.centroid
rows, cols = rasterio.transform.rowcol(T, c.x.values, c.y.values)
rows, cols = np.asarray(rows), np.asarray(cols)
ok = (rows >= 0) & (rows < H) & (cols >= 0) & (cols < W) & (area >= 5)
idx = rows[ok] * W + cols[ok]
n = np.bincount(idx, minlength=H * W).reshape(H, W).astype(float)
roof = np.bincount(idx, weights=area[ok], minlength=H * W).reshape(H, W)

with rasterio.open(DATA / "processed" / "m0" / "buildings_2023_30m.tif") as r:
    hp = r.read(3, masked=True).filled(np.nan)
h3 = hp[: H * 3, : W * 3].reshape(H, 3, W, 3)
hp90 = np.nanmean(h3, axis=(1, 3))

bf = roof / 8100
density = n / 0.81
fp = np.where(n > 0, roof / np.maximum(n, 1), np.nan)
cand = (bf >= RULE["built"]) & (fp <= RULE["footprint"]) & (density >= RULE["density"]) & (hp90 <= RULE["height"])
lab, k = ndimage.label(cand)
sizes = ndimage.sum(cand, lab, range(1, k + 1))
ov_inf = np.isin(lab, np.where(sizes >= RULE["min_cells"])[0] + 1)
ob_inf = grp == 1

inter, union = (ov_inf & ob_inf).sum(), (ov_inf | ob_inf).sum()
res = dict(overture_informal_km2=ov_inf.sum() * 0.0081, openbuildings_informal_km2=ob_inf.sum() * 0.0081,
           overlap_km2=inter * 0.0081, IoU=inter / union if union else np.nan,
           share_of_OB_confirmed=inter / ob_inf.sum() if ob_inf.sum() else np.nan)
print(pd.Series(res).round(3).to_string())

# named neighbourhoods
w, s, e, nn = 32.35, 0.10, 32.85, 0.55
con = duckdb.connect(); con.sql("INSTALL httpfs; LOAD httpfs; SET s3_region='us-west-2';")
names = ["Kisenyi", "Katanga", "Bwaise", "Kalerwe", "Kivulu", "Namuwongo", "Kifumbira", "Kasokoso",
         "Kinawataka", "Nateete", "Ndeeba", "Kazo", "Kawempe", "Mulago", "Banda", "Kibuli"]
q = " OR ".join([f"names.primary ILIKE '%{x}%'" for x in names])
df = con.sql(f"""
  SELECT names.primary AS name, subtype, geometry FROM read_parquet(
    's3://overturemaps-us-west-2/release/2026-10-21.0/theme=divisions/type=division/*', hive_partitioning=1)
  WHERE bbox.xmin > {w} AND bbox.xmax < {e} AND bbox.ymin > {s} AND bbox.ymax < {nn} AND ({q})
""").df()
pts = gpd.GeoDataFrame(df, geometry=df.pop("geometry").apply(lambda g: wkb.loads(bytes(g))), crs="EPSG:4326").to_crs(prof["crs"])
out = []
for _, p in pts.iterrows():
    r_, c_ = rasterio.transform.rowcol(T, p.geometry.x, p.geometry.y)
    win = (slice(max(r_ - 5, 0), r_ + 6), slice(max(c_ - 5, 0), c_ + 6))   # ~1 km window
    out.append(dict(name=p["name"], subtype=p["subtype"],
                    OB_informal_pct=100 * ob_inf[win].mean(), Overture_informal_pct=100 * ov_inf[win].mean()))
named = pd.DataFrame(out).drop_duplicates("name").sort_values("Overture_informal_pct", ascending=False)
named.to_csv(OUTPUTS / "tables" / "m3_named_settlement_check.csv", index=False)
pd.Series(res).to_csv(OUTPUTS / "tables" / "m3_overture_crosscheck.csv", header=["value"])
print(named.round(1).to_string(index=False))
