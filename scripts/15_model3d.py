"""Run the 3D urban climate model.

1. pick three 2 x 2 km focus districts (CBD fixed; informal and frontier chosen from the rasters)
2. per district: 2 m DSM, shadows for 12 months x daylight hours, SVF, annual irradiance,
   pedestrian shade hours, per-roof irradiance with 3D shading, lake-breeze potential flow
3. city-wide: building orientation, wall-sun penalty, frontal area index for the lake breeze,
   joined to settlement type and surface heat
Outputs: DATA/processed/m6/<district>.npz, roofs_<district>.parquet, tables m6_*.csv
"""
import json

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from scipy import ndimage

from kampala import config
from kampala import model3d as M
from kampala.config import DATA, OUTPUTS

P = DATA / "processed"
OUT = P / "m6"; OUT.mkdir(parents=True, exist_ok=True)
TAB = OUTPUTS / "tables"
Y3 = config.load()["epochs"]["t3"]["year"]
CBD = (451300.0, 34400.0)   # Kampala Road, UTM 36N


def best_window(arr, transform, cell, centre, rmin, rmax, size_m=2000):
    k = max(1, int(size_m / cell))
    score = ndimage.uniform_filter(arr.astype(float), size=k)
    rr, cc = np.mgrid[0:arr.shape[0], 0:arr.shape[1]]
    x = transform.c + (cc + 0.5) * transform.a; y = transform.f + (rr + 0.5) * transform.e
    d = np.hypot(x - centre[0], y - centre[1])
    score[(d < rmin) | (d > rmax)] = -1
    i = np.unravel_index(np.argmax(score), score.shape)
    return float(x[i]), float(y[i]), float(score[i])


with rasterio.open(P / "m3" / f"settlement_groups_t3_{Y3}_90m.tif") as r:
    inf = r.read(1) == 1
    inf_xy = best_window(inf, r.transform, 90, CBD, 1500, 8000)
with rasterio.open(P / "m1" / "growth_frontier_10m.tif") as r:
    fr = r.read(1)[::3, ::3]
    t = r.transform * rasterio.Affine.scale(3)
    new = (fr >= 2)
    fr_xy = best_window(new, t, 30, CBD, 6000, 15000)

districts = {"cbd": CBD, "informal": inf_xy[:2], "frontier": fr_xy[:2]}
meta = {k: dict(x=v[0], y=v[1]) for k, v in districts.items()}
meta["informal"]["informal_share"] = inf_xy[2]; meta["frontier"]["new_built_share"] = fr_xy[2]
print(json.dumps(meta, indent=1))
(TAB / "m6_districts.json").write_text(json.dumps(meta, indent=1))

rows = []
for name, (cx, cy) in districts.items():
    print(f"== {name}", flush=True)
    m = M.district_dsm(cx, cy)
    s = M.solar_run(m, log=lambda t: print(t, flush=True))
    w = M.potential_flow(m, wind_from=180.0)
    roofs = M.roof_table(m, s)
    roofs.drop(columns="geometry").to_parquet(OUT / f"roofs_{name}.parquet")
    roofs[["geometry", "h", "roof_kwh_m2", "shading_loss_pct"]].to_file(OUT / f"buildings_{name}.gpkg", driver="GPKG")
    frames = np.stack([s["shadow_frames"][k] for k in sorted(s["shadow_frames"])])
    np.savez_compressed(OUT / f"{name}.npz", dsm=m["dsm"], terrain=m["terrain"], bh=m["bh"], bid=m["bid"],
                        irr=s["irr"], svf=s["svf"], shade_hours=s["shade_hours"], sunlit=s["sunlit"],
                        u=w["u"], v=w["v"], speed=w["speed"], solid=w["solid"], frames=frames,
                        frame_keys=np.array(sorted(s["shadow_frames"])), origin=np.array([cx - 1000, cy + 1000]))
    ground = m["bh"] == 0
    open_w = ~w["solid"]
    rows.append(dict(district=name, buildings=len(m["buildings"]),
                     plan_area_fraction=float((m["bh"] > 0).mean()),
                     mean_height_m=float(m["buildings"]["h"].mean()),
                     height_from_overture_pct=100 * float((m["buildings"]["h_source"] != "default 3.5 m").mean()),
                     ground_svf=float(s["svf"][ground].mean()),
                     ground_shade_hours_10_16=float(s["shade_hours"][ground].mean()),
                     ground_irr_kwh_m2=float(s["irr"][ground].mean()),
                     roof_irr_kwh_m2=float(roofs["roof_kwh_m2"].mean()),
                     roof_shading_loss_pct=float(roofs["shading_loss_pct"].mean()),
                     roofs_shaded_over_10pct=100 * float((roofs["shading_loss_pct"] > 10).mean()),
                     wind_speed_ratio_open=float(w["speed"][open_w].mean()),
                     wind_stagnant_share_pct=100 * float((w["speed"][open_w] < 0.5).mean())))
    print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(TAB / "m6_district_summary.csv", index=False)

# ---------------------------------------------------------------- city-wide metrics
print("== city-wide building metrics", flush=True)
crs = config.load()["study_area"]["crs"]
b = gpd.read_parquet(DATA / "raw" / "overture_buildings.parquet").to_crs(crs)
b = b[b.geometry.type == "Polygon"].copy()
b = b[b.geometry.area.between(10, 50000)]
az, L, Wd = M.building_orientation(b)
lookup = M.wall_sun_lookup()
wall, penalty = M.wall_sun_penalty(az, L, Wd, lookup)
c = b.geometry.centroid
h = b["height"].astype(float).fillna(b["num_floors"].astype(float) * 3.2).fillna(3.5).clip(2.5, 120).values
minx, miny, maxx, maxy = shapely_b = b.geometry.bounds.T.values
city = pd.DataFrame({"x": c.x.values, "y": c.y.values, "az": az, "length": L, "width": Wd, "h": h,
                     "elongation": L / np.maximum(Wd, 0.1), "wall_sun": wall, "wall_penalty": penalty,
                     "frontal_S": (maxx - minx) * h})       # facade width facing a southerly wind x height
pd.Series(lookup).to_csv(TAB / "m6_wall_sun_by_azimuth.csv", header=["kwh_m2_yr"])

with rasterio.open(P / "m3" / f"settlement_groups_t3_{Y3}_90m.tif") as r:
    grp = r.read(1); gt = r.transform
    rr, cc = rasterio.transform.rowcol(gt, city.x.values, city.y.values)
    rr = np.clip(np.asarray(rr), 0, grp.shape[0] - 1); cc = np.clip(np.asarray(cc), 0, grp.shape[1] - 1)
    city["settlement"] = pd.Series(grp[rr, cc]).map({1: "informal-type", 2: "formal dense", 3: "sparse built", 4: "non-built"}).values
    # frontal area index on the 90 m grid (lambda_f = sum frontal area / cell area)
    lam = np.zeros(grp.shape); np.add.at(lam, (rr, cc), city["frontal_S"].values); lam /= 90 * 90
with rasterio.open(P / "m1" / "growth_frontier_10m.tif") as r:
    fr = r.read(1)
    r2, c2 = rasterio.transform.rowcol(r.transform, city.x.values, city.y.values)
    r2 = np.clip(np.asarray(r2), 0, fr.shape[0] - 1); c2 = np.clip(np.asarray(c2), 0, fr.shape[1] - 1)
    city["built_period"] = pd.Series(fr[r2, c2]).map({1: f"by 2017", 2: "2017-21", 3: "2021-25"}).values
city.to_parquet(OUT / "city_buildings.parquet")
with rasterio.open(P / "m3" / f"settlement_groups_t3_{Y3}_90m.tif") as r:
    prof = r.profile
prof.update(dtype="float32", nodata=np.nan)
with rasterio.open(OUT / "frontal_area_index_S_90m.tif", "w", **prof) as dst:
    dst.write(lam.astype("float32"), 1)

ew = lambda a: ((a > 67.5) & (a < 112.5)).mean() * 100    # long axis within 22.5 deg of east-west
summ = city.groupby("settlement").agg(buildings=("az", "size"), ew_aligned_pct=("az", ew),
                                      wall_sun_kwh_m2=("wall_sun", "mean"), wall_penalty_kwh_m2=("wall_penalty", "mean"),
                                      elongation=("elongation", "median"))
summ.to_csv(TAB / "m6_orientation_by_settlement.csv")
per = city.groupby("built_period").agg(buildings=("az", "size"), ew_aligned_pct=("az", ew),
                                       wall_penalty_kwh_m2=("wall_penalty", "mean"))
per.to_csv(TAB / "m6_orientation_by_period.csv")
print(summ.round(1)); print(per.round(1))

# lambda_f vs settlement and LST anomaly (90 m)
with rasterio.open(P / "m2" / f"lst_landsat_t3_{Y3}_30m.tif") as r:
    a = r.read(2, masked=True).filled(np.nan)
hh, ww = (a.shape[0] // 3) * 3, (a.shape[1] // 3) * 3
a90 = np.nanmean(a[:hh, :ww].reshape(hh // 3, 3, ww // 3, 3), axis=(1, 3))
H, W = min(a90.shape[0], lam.shape[0]), min(a90.shape[1], lam.shape[1])
dfv = pd.DataFrame({"lambda_f": lam[:H, :W].ravel(), "lst_anom": a90[:H, :W].ravel(), "grp": grp[:H, :W].ravel()})
dfv = dfv[(dfv.lambda_f > 0) & np.isfinite(dfv.lst_anom)]
dfv["lambda_bin"] = pd.cut(dfv.lambda_f, [0, 0.05, 0.1, 0.2, 0.3, 0.5, 5])
vb = dfv.groupby("lambda_bin", observed=True).agg(cells=("lst_anom", "size"), lst_anom=("lst_anom", "mean"))
vb.to_csv(TAB / "m6_lst_by_frontal_area.csv")
lg = dfv.groupby("grp").lambda_f.mean().rename({1: "informal-type", 2: "formal dense", 3: "sparse built", 4: "non-built"})
lg.to_csv(TAB / "m6_frontal_area_by_settlement.csv")
print(vb.round(2)); print(lg.round(3))
