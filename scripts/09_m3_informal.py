"""Module 3 (new): informal-settlement morphology and its thermal trajectory, 2017 -> 2021 -> 2025.

Settlement delineation (per year, 90 m cells = 3x3 of the 30 m M0 grid, Open Buildings 2.5D):
    informal-type morphology if   built share >= 35 %, mean footprint <= 80 m2,
                                  density >= 30 buildings/ha, p90 roof height <= 8 m,
    in connected clusters of >= 3 cells (~2.4 ha). Thresholds follow the morphology literature
    (Kuffer et al. 2016; Taubenböck et al. 2018) and must be validated against reference
    boundaries (ACTogether/SDI profiles) before the chapter is final.
Comparison groups: informal-type, other dense built-up (formal), sparse built-up, non-built.

Carried over, not repeated: valley-floor vs hill heat exposure (P3, uganda-rainfall-climate) -
used here as a covariate (HAND) so M3 isolates the settlement-type effect.
"""
import numpy as np
import pandas as pd
import rasterio
from scipy import ndimage

from kampala import config
from kampala.config import DATA, OUTPUTS

M0, M2, M3 = DATA / "processed" / "m0", DATA / "processed" / "m2", DATA / "processed" / "m3"
M3.mkdir(parents=True, exist_ok=True)
EPOCHS = config.load()["epochs"]
BYEAR = {"t1": 2016, "t2": 2020, "t3": 2023}
RULE = dict(built=0.35, footprint=80.0, density=30.0, height=8.0, min_cells=3)
GROUPS = {1: "informal-type", 2: "formal dense", 3: "sparse built", 4: "non-built"}


def block(a, f=3, how="mean"):
    h, w = (a.shape[0] // f) * f, (a.shape[1] // f) * f
    v = a[:h, :w].reshape(h // f, f, w // f, f)
    return np.nanmean(v, axis=(1, 3)) if how == "mean" else np.nansum(v, axis=(1, 3))


def read(path):
    with rasterio.open(path) as r:
        return r.read(masked=True).filled(np.nan), r.profile


rows, groups = [], {}
for ep, spec in EPOCHS.items():
    b, prof = read(M0 / f"buildings_{BYEAR[ep]}_30m.tif")
    lst, lprof = read(M2 / f"lst_landsat_{ep}_{spec['year']}_30m.tif")
    assert b.shape[1:] == lst.shape[1:], "M0 and M2 grids differ"
    bf, hp, n = block(b[0]), block(b[2]), block(b[3], how="sum")
    cell_ha = 0.81
    density = n / cell_ha
    footprint = np.where(n > 0.5, bf * 8100 / np.maximum(n, 1e-6), np.nan)
    cand = (bf >= RULE["built"]) & (footprint <= RULE["footprint"]) & (density >= RULE["density"]) & (hp <= RULE["height"])
    lab, k = ndimage.label(cand)
    sizes = ndimage.sum(cand, lab, range(1, k + 1))
    informal = np.isin(lab, np.where(sizes >= RULE["min_cells"])[0] + 1)

    g = np.full(bf.shape, 4, dtype="uint8")
    g[bf >= 0.05] = 3
    g[bf >= RULE["built"]] = 2
    g[informal] = 1
    groups[ep] = g

    anom = block(lst[1])
    for k_, nm in GROUPS.items():
        m = (g == k_) & np.isfinite(anom)
        rows.append(dict(epoch=ep, year=spec["year"], group=nm, km2=m.sum() * cell_ha / 100,
                         LST_anom_mean=float(np.nanmean(anom[m])), LST_anom_p90=float(np.nanpercentile(anom[m], 90))))
    p = prof.copy()
    t = p["transform"]
    p.update(count=1, dtype="uint8", nodata=0, height=g.shape[0], width=g.shape[1],
             transform=rasterio.Affine(t.a * 3, t.b, t.c, t.d, t.e * 3, t.f))
    with rasterio.open(M3 / f"settlement_groups_{ep}_{spec['year']}_90m.tif", "w", **p) as dst:
        dst.write(g, 1)
    print(f"{spec['year']}: informal-type {informal.sum() * cell_ha / 100:.1f} km2 in {(sizes >= RULE['min_cells']).sum()} clusters")

res = pd.DataFrame(rows)
res.to_csv(OUTPUTS / "tables" / "m3_lst_by_settlement_group.csv", index=False)
print(res.pivot(index="group", columns="year", values="LST_anom_mean").round(2).loc[list(GROUPS.values())])
print(res.pivot(index="group", columns="year", values="km2").round(1).loc[list(GROUPS.values())])

# trajectories: cells informal in all three years vs formal-dense in all three years
lsts = {}
for ep, spec in EPOCHS.items():
    lst, _ = read(M2 / f"lst_landsat_{ep}_{spec['year']}_30m.tif")
    lsts[ep] = block(lst[1])
stable_inf = np.logical_and.reduce([groups[e] == 1 for e in EPOCHS])
stable_for = np.logical_and.reduce([groups[e] == 2 for e in EPOCHS])
new_inf = (groups["t1"] >= 3) & (groups["t3"] == 1)
traj = pd.DataFrame({
    nm: [float(np.nanmean(lsts[e][m])) for e in EPOCHS] for nm, m in
    {"always informal-type": stable_inf, "always formal dense": stable_for, "became informal-type": new_inf}.items()
}, index=[s["year"] for s in EPOCHS.values()]).T
traj["change_2017_2025"] = traj.iloc[:, -1] - traj.iloc[:, 0]
traj["km2"] = [m.sum() * 0.0081 for m in (stable_inf, stable_for, new_inf)]
traj.to_csv(OUTPUTS / "tables" / "m3_thermal_trajectories.csv")
print(traj.round(2))
