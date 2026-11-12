"""Module 4 (new): where should Kampala plant trees?

Priority (30 m cells) = heat x people x canopy deficit, only where there is room to plant.
    heat        percentile rank of 2025 LST anomaly (M2)
    people      percentile rank of population (GHS-POP 2020, carried from uganda-rainfall-climate),
                x 1.5 inside informal-type settlement (M3)
    deficit     1 - tree share (M1 2025 map, 10 m -> 30 m)
    plantable   share of open grass/cropland/bare pixels (M1) on slopes < 30 deg; wetlands and water
                are never plantable (wetlands are protected and already cool)
Cooling model: OLS of LST on tree share, built share, elevation and NDVI-free covariates
    across built-up cells, giving the surface cooling per +10 % canopy.
Scenario: plant half of the plantable space in very-high and high priority cells.
"""
import numpy as np
import pandas as pd
import rasterio
import statsmodels.api as sm
import xarray as xr
from rasterio.warp import Resampling, reproject

from kampala import config
from kampala.config import DATA, OUTPUTS

P = DATA / "processed"
M4 = P / "m4"
M4.mkdir(parents=True, exist_ok=True)
y3 = config.load()["epochs"]["t3"]["year"]

with rasterio.open(P / "m2" / f"lst_landsat_t3_{y3}_30m.tif") as r:
    grid = r.profile
    lst = r.read(1, masked=True).filled(np.nan)
    anom = r.read(2, masked=True).filled(np.nan)
H, W = grid["height"], grid["width"]


def to_grid(src_arr, src_transform, src_crs, resampling, dtype="float32"):
    out = np.zeros((H, W), dtype=dtype)
    reproject(src_arr, out, src_transform=src_transform, src_crs=src_crs, dst_transform=grid["transform"],
              dst_crs=grid["crs"], resampling=resampling)
    return out


# land cover 10 m -> shares at 30 m
with rasterio.open(P / "m1" / f"landcover_t3_{y3}_10m.tif") as r:
    lc = r.read(1)
    share = {k: to_grid((lc == k).astype("float32"), r.transform, r.crs, Resampling.average) for k in range(1, 8)}
tree = share[3]
open_land = share[4] + share[5]
water_wet = share[6] + share[7]

with rasterio.open(P / "m0" / "buildings_2023_30m.tif") as r:
    built = r.read(1, masked=True).filled(0)
with rasterio.open(P / "m0" / "terrain_fabdem_30m.tif") as r:
    elev = to_grid(r.read(1), r.transform, r.crs, Resampling.bilinear)
    slope = to_grid(r.read(2), r.transform, r.crs, Resampling.bilinear)

pop_nc = xr.open_dataarray(DATA / "carried" / "uganda-rainfall-climate" / "kampala_ghs_pop.nc").sel(epoch=2020)
pop_nc = pop_nc.sortby("lat", ascending=False)
lat, lon = pop_nc["lat"].values, pop_nc["lon"].values
dx = (lon[-1] - lon[0]) / (len(lon) - 1)   # grid spacing is slightly uneven; use the mean
dy = (lat[0] - lat[-1]) / (len(lat) - 1)
src_t = rasterio.transform.from_origin(lon[0] - dx / 2, lat[0] + dy / 2, dx, dy)
# people per m2 -> resample -> people per 30 m cell
cell_m2 = (dx * 111320) * (dy * 110574)
pop = to_grid((np.nan_to_num(pop_nc.values) / cell_m2).astype("float32"), src_t, "EPSG:4326", Resampling.bilinear) * 900

with rasterio.open(P / "m3" / f"settlement_groups_t3_{y3}_90m.tif") as r:
    informal = to_grid((r.read(1) == 1).astype("float32"), r.transform, r.crs, Resampling.nearest) > 0.5


def prank(a, mask):
    out = np.full(a.shape, np.nan)
    out[mask] = pd.Series(a[mask]).rank(pct=True).values
    return out


land = np.isfinite(lst) & (water_wet < 0.5)
plantable = np.where(slope < 30, open_land, 0) * land
heat = prank(anom, land)
people = prank(pop * np.where(informal, 1.5, 1.0), land)
priority = heat * people * (1 - tree)
priority[plantable < 0.10] = np.nan

q = np.nanpercentile(priority, [70, 85, 95])
cls = np.zeros((H, W), dtype="uint8")
ok = np.isfinite(priority)
cls[ok] = np.digitize(priority[ok], q) + 1          # 1 low .. 4 very high
p = grid.copy(); p.update(count=1, dtype="uint8", nodata=0)
with rasterio.open(M4 / "tree_priority_classes_30m.tif", "w", **p) as dst:
    dst.write(cls, 1)
p.update(dtype="float32", nodata=np.nan)
with rasterio.open(M4 / "tree_priority_score_30m.tif", "w", **p) as dst:
    dst.write(priority.astype("float32"), 1)

# cooling model on built-up cells
m = land & (built >= 0.05) & np.isfinite(lst)
X = pd.DataFrame({"tree": tree[m], "built": built[m], "elev": elev[m], "open": open_land[m]})
X = sm.add_constant(X)
fit = sm.OLS(lst[m], X).fit()
cool10 = -fit.params["tree"] * 0.10
print(fit.summary().tables[1])
print(f"surface cooling per +10 % canopy: {cool10:.2f} C (R2 {fit.rsquared:.2f})")

# scenario: plant half of the plantable space in high (3) and very high (4) cells
target = cls >= 3
added = np.where(target, 0.5 * plantable, 0)
d_lst = -fit.params["tree"] * added
names = {1: "low", 2: "medium", 3: "high", 4: "very high"}
summary = pd.DataFrame([dict(priority=names[k], km2=(cls == k).sum() * 900 / 1e6,
                             plantable_km2=float((plantable * (cls == k)).sum() * 900 / 1e6),
                             people=float(np.nansum(pop[cls == k])),
                             mean_LST_anom=float(np.nanmean(anom[cls == k])),
                             current_tree_pct=100 * float(np.nanmean(tree[cls == k])))
                        for k in names])
summary.to_csv(OUTPUTS / "tables" / "m4_priority_summary.csv", index=False)
scen = dict(cells=int(target.sum()), new_canopy_km2=float(added.sum() * 900 / 1e6),
            people_in_planted_cells=float(np.nansum(pop[target])),
            mean_cooling_C=float(np.nanmean(d_lst[target])), max_cooling_C=float(np.nanmax(d_lst[target])),
            cooling_per_10pct_canopy_C=float(cool10), model_R2=float(fit.rsquared))
pd.Series(scen).to_csv(OUTPUTS / "tables" / "m4_planting_scenario.csv", header=["value"])
print(summary.round(2).to_string(index=False))
print(pd.Series(scen).round(3).to_string())
