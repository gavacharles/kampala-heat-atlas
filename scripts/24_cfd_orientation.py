"""Viscous 2D flow (lattice Boltzmann) for the three districts, and an orientation search at the
growth frontier: every building turned about its centroid to one long-axis bearing (0 = N-S,
90 = E-W) and scored on wall sun (vs the best bearing) and on breeze (share of open space below
half the inflow speed, median speed ratio). Outputs: DATA/processed/m8/*.npz, tables m8_*.csv.
"""
import geopandas as gpd
import numpy as np
import pandas as pd
import shapely.affinity
from rasterio.features import rasterize
from rasterio.transform import from_origin

from kampala import lbm
from kampala import model3d as M
from kampala.config import DATA, OUTPUTS

M6, OUT, TAB = DATA / "processed" / "m6", DATA / "processed" / "m8", OUTPUTS / "tables"
OUT.mkdir(parents=True, exist_ok=True)
RES, STEPS, AVG = 4.0, 30000, 18000
lookup = pd.read_csv(TAB / "m6_wall_sun_by_azimuth.csv", index_col=0)["kwh_m2_yr"].values


def solid_from(geoms, origin, n=500):
    tr = from_origin(origin[0], origin[1], RES, RES)
    return rasterize(((g, 1) for g in geoms), out_shape=(n, n), transform=tr, fill=0, dtype="uint8").astype(bool)


def score(spd, solid):
    open_ = ~solid
    open_[:6] = False; open_[-6:] = False                # exclude the boundary buffer strips
    return dict(open_median_speed=float(np.median(spd[open_])), weak_breeze_pct=100 * float((spd[open_] < 0.5).mean()),
                plan_area_fraction=float(solid.mean()))


rows = []
for name in ("cbd", "informal", "frontier"):
    d = np.load(M6 / f"{name}.npz"); origin = d["origin"]
    b = gpd.read_file(M6 / f"buildings_{name}.gpkg")
    sol = solid_from(b.geometry, origin)
    spd = lbm.run(sol, steps=STEPS, avg_from=AVG, log=lambda s: print(name, s, flush=True))
    np.savez_compressed(OUT / f"lbm_{name}.npz", speed=spd, solid=sol)
    rows.append(dict(case=name, layout="as built", **score(spd, sol)))
    print(rows[-1], flush=True)

d = np.load(M6 / "frontier.npz"); origin = d["origin"]
b = gpd.read_file(M6 / "buildings_frontier.gpkg"); b = b[b.geometry.type == "Polygon"].reset_index(drop=True)
az, L, Wd = M.building_orientation(b)
_, pen0 = M.wall_sun_penalty(az, L, Wd, lookup)
rows[-1]["wall_penalty_kwh_m2"] = float(pen0.mean())
for theta in (0, 30, 60, 90, 120, 150):
    rot = [shapely.affinity.rotate(g, -(theta - a), origin="centroid") for g, a in zip(b.geometry, az)]
    _, pen = M.wall_sun_penalty(np.full(len(b), float(theta)), L, Wd, lookup)
    sol = solid_from(rot, origin)
    spd = lbm.run(sol, steps=STEPS, avg_from=AVG, log=lambda s: print("theta", theta, s, flush=True))
    np.savez_compressed(OUT / f"lbm_frontier_theta{theta:03d}.npz", speed=spd, solid=sol)
    rows.append(dict(case="frontier", layout=f"all long axes at {theta} deg", wall_penalty_kwh_m2=float(pen.mean()), **score(spd, sol)))
    print(rows[-1], flush=True)
res = pd.DataFrame(rows)
pf = pd.read_csv(TAB / "m6_district_summary.csv").set_index("district")["wind_stagnant_share_pct"]
res["potential_flow_weak_pct"] = [pf.get(r.case) if r.layout == "as built" else np.nan for r in res.itertuples()]
res.to_csv(TAB / "m8_cfd_orientation.csv", index=False)
print(res.round(2).to_string(index=False))
