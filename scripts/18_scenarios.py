"""What-if scenarios on the 3D model (M6).

A  Trees: plant street trees in the CBD and informal districts and re-run shade and heat stress.
   Candidate spots: open ground 2-15 m from a building (where people walk), currently < 0.5 h of
   midday shade; one tree per 12 m; up to 600 per district (KCCA plants >= 8,000 a year citywide).
   Tree = solid crown, 9 m high, 5 m radius (a mature street tree, e.g. Terminalia or Markhamia); the ground
   under a crown is shaded with sky view 0.15. Porosity is ignored, so shade is an upper bound.
B  Walking heat (UTCI, 14:30 on 21 March) before and after trees.
   Air: ERA5-Land dry-season 14:00, hot afternoon (90th percentile 27.1 C, RH 67 %), wind 2.3 m/s
   x potential-flow speed ratio. Mean radiant temperature: simplified SOLWEIG-style budget for a
   standing person (projection factor, SVF-weighted sky and wall long-wave, sunlit ground +15 C).
   ERA5-Land air temperature is a rural grid-cell value, so absolute UTCI is conservative.
C  Rotated frontier: turn every frontier building to an east-west long axis about its centroid and
   re-run the breeze; wall-sun penalty before and after.
D  Solar over a day: hourly rooftop generation (21 Mar / 21 Jun / 21 Dec) for the CBD district with
   3D shading, and the city-wide daily shape scaled to the 33 TWh/yr technical potential.
"""
import json

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
import shapely.affinity
from pythermalcomfort.models import utci
from rasterio.features import rasterize
from rasterio.transform import from_origin
from scipy import ndimage

from kampala import model3d as M
from kampala.config import DATA, OUTPUTS

M6 = DATA / "processed" / "m6"
OUT = DATA / "processed" / "m7"; OUT.mkdir(parents=True, exist_ok=True)
TAB = OUTPUTS / "tables"
W14 = pd.read_csv(DATA / "interim" / "era5land_14h_dryseason.csv", index_col=0)["value"]
TA, RH, V10 = float(W14["ta_p90"]), float(W14["rh_mean"]), float(W14["v10_mean"])
SIGMA = 5.67e-8
SUN = M.sun_positions()


def load(name):
    d = dict(np.load(M6 / f"{name}.npz"))
    x0, y0 = d["origin"]
    d["transform"] = from_origin(x0, y0, 2.0, 2.0)
    d["res"] = 2.0
    return d


def plant_trees(d, n_max=600, spacing=14.0, height=9.0, radius=5.0):
    bh = d["bh"]
    dist = ndimage.distance_transform_edt(bh == 0) * 2.0
    cand = (bh == 0) & (dist >= 2) & (dist <= 15) & (d["shade_hours"] < 0.5)
    # pedestrian proxy: number of building cells within 30 m (busier streets first)
    busy = ndimage.uniform_filter((bh > 0).astype(float), size=31)
    score = np.where(cand, busy, -1)
    order = np.argsort(score.ravel())[::-1]
    taken = np.zeros_like(cand)
    trees = []
    step = int(spacing / 2.0)
    for idx in order:
        if score.flat[idx] <= 0 or len(trees) >= n_max:
            break
        r, c = divmod(idx, bh.shape[1])
        if taken[max(0, r - step):r + step + 1, max(0, c - step):c + step + 1].any():
            continue
        taken[r, c] = True
        trees.append((r, c))
    rr, cc = np.ogrid[:bh.shape[0], :bh.shape[1]]
    crown = np.zeros_like(bh, dtype=bool)
    k = int(radius / 2.0)
    for r, c in trees:
        r0, r1, c0, c1 = max(0, r - k), r + k + 1, max(0, c - k), c + k + 1
        sub = (rr[r0:r1] - r) ** 2 + (cc[:, c0:c1] - c) ** 2 <= k * k
        crown[r0:r1, c0:c1] |= sub
    dsm = d["dsm"].copy()
    dsm[crown] = np.maximum(dsm[crown], d["terrain"][crown] + height)
    return dsm, crown, np.array(trees)


def midday_shade(dsm, crown=None):
    sh_h = np.zeros(dsm.shape, dtype="float32")
    for t, row in SUN.iterrows():
        if 10 <= t.hour < 16:
            sh = M.shadow(dsm, row.azimuth, row.elevation)
            if crown is not None:
                sh |= crown                      # standing under the crown = in shade
            sh_h += row.weight * sh / 365.0
    return sh_h


def utci_map(dsm, svf, speed, when=(3, 14), crown=None):
    row = SUN[(SUN.index.month == when[0]) & (SUN.index.hour == when[1])].iloc[0]
    sh = M.shadow(dsm, row.azimuth, row.elevation)
    if crown is not None:                        # pedestrian under a crown: shaded, little open sky
        sh |= crown
        svf = np.where(crown, 0.15, svf)
    e = np.radians(row.elevation)
    I, D = row.dni * row.k, row.dhi * row.k
    G = I * np.sin(e) + D
    fp = 0.308 * np.cos(e * (1 - row.elevation ** 2 / 48402))          # standing person (ASHRAE)
    S = 0.7 * (fp * I * (~sh) + 0.5 * D * svf + 0.5 * 0.2 * G)
    ta_k = TA + 273.15
    tg = ta_k + np.where(sh, 3.0, 15.0)                                  # ground surface
    L = 0.97 * (0.5 * (0.85 * SIGMA * ta_k ** 4 * svf + SIGMA * (ta_k + 4) ** 4 * (1 - svf)) + 0.5 * 0.95 * SIGMA * tg ** 4)
    tmrt = ((S + L) / (0.97 * SIGMA)) ** 0.25 - 273.15
    sp = np.kron(speed, np.ones((2, 2)))[: dsm.shape[0], : dsm.shape[1]]
    v = np.clip(V10 * sp, 0.5, 17)
    u = utci(tdb=np.full(tmrt.shape, TA).ravel(), tr=tmrt.ravel(), v=v.ravel(), rh=np.full(tmrt.size, RH), limit_inputs=False)
    return np.asarray(u["utci"]).reshape(tmrt.shape).astype("float32"), tmrt.astype("float32")


rows, utci_rows = [], []
for name in ("cbd", "informal"):
    d = load(name)
    ground = d["bh"] == 0
    walk = ground & (ndimage.distance_transform_edt(ground) * 2.0 <= 15)
    dsm_t, crown, trees = plant_trees(d)
    sh_before = d["shade_hours"]
    sh_after = midday_shade(dsm_t, crown)
    svf_t = M.sky_view_factor(dsm_t)
    u0, t0 = utci_map(d["dsm"], d["svf"], d["speed"])
    u1, t1 = utci_map(dsm_t, svf_t, d["speed"], crown=crown)
    frames = {h: M.shadow(dsm_t, r.azimuth, r.elevation) for h, r in
              ((t.hour, r) for t, r in SUN[SUN.index.month == 3].iterrows())}
    np.savez_compressed(OUT / f"trees_{name}.npz", dsm_t=dsm_t, crown=crown, trees=trees, sh_after=sh_after,
                        svf_t=svf_t, utci0=u0, utci1=u1, tmrt0=t0, tmrt1=t1,
                        frames=np.stack([frames[h] for h in sorted(frames)]), hours=np.array(sorted(frames)))
    w = walk
    rows.append(dict(district=name, trees=len(trees),
                     walk_shade_h_before=float(sh_before[w].mean()), walk_shade_h_after=float(sh_after[w].mean()),
                     walk_area_2h_shade_before_pct=100 * float((sh_before[w] >= 2).mean()),
                     walk_area_2h_shade_after_pct=100 * float((sh_after[w] >= 2).mean()),
                     shade_m2h_per_tree_per_day=float(((sh_after - sh_before)[ground].sum() * 4) / max(len(trees), 1)),
                     utci_walk_before=float(u0[w].mean()), utci_walk_after=float(u1[w].mean()),
                     very_strong_stress_pct_before=100 * float((u0[w] > 38).mean()),
                     very_strong_stress_pct_after=100 * float((u1[w] > 38).mean()),
                     tmrt_walk_before=float(t0[w].mean()), tmrt_walk_after=float(t1[w].mean())))
    print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(TAB / "m7_tree_scenario_3d.csv", index=False)

# local effect: where the trees put people in the shade (under crowns or in their new shadow)
loc_rows = []
for name in ("cbd", "informal"):
    d = dict(np.load(M6 / f"{name}.npz")); t = dict(np.load(OUT / f"trees_{name}.npz"))
    ground = d["bh"] == 0
    zone = ground & ((t["sh_after"] - d["shade_hours"] > 0.25) | t["crown"])
    cooled = ground & (t["utci1"] < t["utci0"] - 0.5)
    loc_rows.append(dict(district=name, trees=len(t["trees"]), shaded_zone_m2=int(zone.sum() * 4),
                         shade_h_before_in_tree_zone=float(d["shade_hours"][zone].mean()),
                         shade_h_after_in_tree_zone=float(t["sh_after"][zone].mean()),
                         cooled_m2_at_1430=int(cooled.sum() * 4),
                         utci_1430_sun=float(t["utci0"][cooled].mean()), utci_1430_tree_shade=float(t["utci1"][cooled].mean()),
                         tmrt_sun=float(t["tmrt0"][cooled].mean()), tmrt_shade=float(t["tmrt1"][cooled].mean())))
loc = pd.DataFrame(loc_rows); loc.to_csv(TAB / "m7_tree_local_effect.csv", index=False)
print(loc.round(2).T, flush=True)

# ---------------------------------------------------------------- C rotated frontier
print("== rotated frontier", flush=True)
lookup = pd.read_csv(TAB / "m6_wall_sun_by_azimuth.csv", index_col=0)["kwh_m2_yr"].values
d = load("frontier")
b = gpd.read_file(M6 / "buildings_frontier.gpkg")
b = b[b.geometry.type == "Polygon"].reset_index(drop=True)
az, L, Wd = M.building_orientation(b)
_, pen0 = M.wall_sun_penalty(az, L, Wd, lookup)
rot = [shapely.affinity.rotate(g, -(90 - a), origin="centroid") for g, a in zip(b.geometry, az)]   # clockwise to E-W
b_rot = b.copy(); b_rot["geometry"] = rot
az1, L1, W1 = M.building_orientation(b_rot)
_, pen1 = M.wall_sun_penalty(az1, L1, W1, lookup)
n = d["bh"].shape[0]
bh_rot = rasterize(((g, h) for g, h in zip(b_rot.geometry, b_rot["h"])), out_shape=(n, n), transform=d["transform"], fill=0, dtype="float32")
m_rot = dict(bh=bh_rot, res=2.0)
w0 = M.potential_flow(dict(bh=d["bh"], res=2.0)); w1 = M.potential_flow(m_rot)
b_rot[["geometry", "h"]].to_file(OUT / "frontier_rotated.gpkg", driver="GPKG")
np.savez_compressed(OUT / "frontier_rotation.npz", bh0=d["bh"], bh1=bh_rot, speed0=w0["speed"], speed1=w1["speed"],
                    solid0=w0["solid"], solid1=w1["solid"], u1=w1["u"], v1=w1["v"])
rotation = dict(buildings=len(b), ew_aligned_before_pct=100 * float(((az > 67.5) & (az < 112.5)).mean()),
                wall_penalty_before=float(pen0.mean()), wall_penalty_after=float(pen1.mean()),
                wall_sun_saved_mwh_per_yr=float(((pen0 - pen1) * 2 * (L + Wd) * b["h"].values).sum() / 1000),
                stagnant_open_before_pct=100 * float((w0["speed"][~w0["solid"]] < 0.5).mean()),
                stagnant_open_after_pct=100 * float((w1["speed"][~w1["solid"]] < 0.5).mean()))
print(rotation, flush=True)
(TAB / "m7_rotation_scenario.json").write_text(json.dumps(rotation, indent=1))

# ---------------------------------------------------------------- D solar over a day
print("== solar over a day", flush=True)
d = load("cbd")
roofs = pd.read_parquet(M6 / "roofs_cbd.parquet")
bid = d["bid"]; ids = bid[bid > 0]; nb = len(roofs) + 1
keys = [tuple(k) for k in d["frame_keys"]]
EFF, PR = 0.21, 0.80
hourly = []
for (mo, h) in keys:
    row = SUN[(SUN.index.month == mo) & (SUN.index.hour == h)].iloc[0]
    sh = d["frames"][keys.index((mo, h))]
    irr = row.k * (row.dni * np.sin(np.radians(row.elevation)) * (~sh) + row.dhi * d["svf"])   # W/m2
    kw = np.bincount(ids, weights=irr[bid > 0], minlength=nb)[1:] * 4.0 * 0.65 * EFF * PR / 1000   # 2 m cells, usable 65 %
    hourly.append(dict(month=mo, hour=h, district_mw=float(kw.sum() / 1000)))
hourly = pd.DataFrame(hourly)
city = pd.read_csv(TAB / "m5_solar_totals.csv", index_col=0)["value"]
for mo in (3, 6, 12):
    m = hourly.month == mo
    hourly.loc[m, "city_share_of_day"] = hourly.loc[m, "district_mw"] / hourly.loc[m, "district_mw"].sum()
hourly["city_gw"] = hourly["city_share_of_day"] * float(city["generation_GWh_yr"]) / 365.0     # GWh in that hour ~ GW
hourly.to_csv(TAB / "m7_solar_hourly.csv", index=False)
print(hourly.pivot(index="hour", columns="month", values="city_gw").round(1))
