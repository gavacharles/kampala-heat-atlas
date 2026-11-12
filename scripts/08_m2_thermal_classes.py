"""Module 2 (new): thermal classification map of Kampala for 2017 / 2021 / 2025.

Classes from the standardised LST anomaly over land (mean +/- 0.5 and 1.5 SD; Chen et al. 2006,
widely used for surface UHI zoning):
    1 very cool  z < -1.5        2 cool  -1.5..-0.5     3 moderate  -0.5..0.5
    4 warm       0.5..1.5        5 hot   z > 1.5
Water (NDVI < 0 or LCZ G) is excluded. Outputs: class rasters, area table, 2017->2025 transition
matrix, persistent-hotspot map, LST by Local Climate Zone.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.colors import ListedColormap
from rasterio.warp import Resampling, reproject

from kampala import config
from kampala.config import DATA, OUTPUTS, FIGURES

M2 = DATA / "processed" / "m2"
EPOCHS = config.load()["epochs"]
NAMES = {1: "very cool", 2: "cool", 3: "moderate", 4: "warm", 5: "hot"}
COLOURS = ["#2c7bb6", "#abd9e9", "#ffffbf", "#fdae61", "#d7191c"]
LCZ_NAMES = {1: "1 compact high-rise", 2: "2 compact mid-rise", 3: "3 compact low-rise", 4: "4 open high-rise",
             5: "5 open mid-rise", 6: "6 open low-rise", 7: "7 lightweight low-rise", 8: "8 large low-rise",
             9: "9 sparsely built", 10: "10 heavy industry", 11: "A dense trees", 12: "B scattered trees",
             13: "C bush, scrub", 14: "D low plants", 15: "E bare rock/paved", 16: "F bare soil/sand", 17: "G water"}


def read(path, band=1):
    with rasterio.open(path) as r:
        return r.read(band, masked=True).filled(np.nan), r.profile


def lcz_on_grid(profile):
    with rasterio.open(M2 / "lcz_demuzere_100m.tif") as r:
        out = np.zeros((profile["height"], profile["width"]), dtype="uint8")
        reproject(r.read(1), out, src_transform=r.transform, src_crs=r.crs,
                  dst_transform=profile["transform"], dst_crs=profile["crs"], resampling=Resampling.nearest)
    return out


classes, rows, lcz_rows = {}, [], []
for ep, spec in EPOCHS.items():
    f = M2 / f"lst_landsat_{ep}_{spec['year']}_30m.tif"
    lst, prof = read(f, 1)
    ndvi, _ = read(f, 4)
    lcz = lcz_on_grid(prof)
    land = np.isfinite(lst) & (ndvi > 0) & (lcz != 17)
    z = np.full(lst.shape, np.nan)
    z[land] = (lst[land] - lst[land].mean()) / lst[land].std()
    c = np.zeros(lst.shape, dtype="uint8")
    c[land] = np.digitize(z[land], [-1.5, -0.5, 0.5, 1.5]) + 1
    classes[ep] = c

    p = prof.copy(); p.update(count=1, dtype="uint8", nodata=0)
    with rasterio.open(M2 / f"thermal_classes_{ep}_{spec['year']}_30m.tif", "w", **p) as dst:
        dst.write(c, 1)
    for k, nm in NAMES.items():
        rows.append(dict(epoch=ep, year=spec["year"], cls=k, name=nm, km2=(c == k).sum() * 900 / 1e6,
                         mean_LST=float(np.nanmean(lst[c == k]))))
    for k, nm in LCZ_NAMES.items():
        m = land & (lcz == k)
        if m.sum() > 100:
            lcz_rows.append(dict(year=spec["year"], lcz=nm, km2=m.sum() * 900 / 1e6,
                                 LST_anom_mean=float(np.nanmean(lst[m] - lst[land].mean())),
                                 share_hot_pct=100 * float((c[m] == 5).mean())))
    print(ep, "land mean LST %.1f C, sd %.2f" % (lst[land].mean(), lst[land].std()))

tab = OUTPUTS / "tables"
area = pd.DataFrame(rows)
area.to_csv(tab / "m2_thermal_class_area.csv", index=False)
print(area.pivot(index="name", columns="year", values="km2").round(1).loc[list(NAMES.values())])

lczt = pd.DataFrame(lcz_rows)
lczt.to_csv(tab / "m2_lst_by_lcz.csv", index=False)
print(lczt.pivot(index="lcz", columns="year", values="LST_anom_mean").round(2))

# 2017 -> 2025 transitions and persistent hotspots
a, b = classes["t1"], classes["t3"]
ok = (a > 0) & (b > 0)
tm = pd.crosstab(pd.Series(a[ok], name="2017").map(NAMES), pd.Series(b[ok], name="2025").map(NAMES)) * 900 / 1e6
tm = tm.reindex(index=NAMES.values(), columns=NAMES.values()).fillna(0)
tm.to_csv(tab / "m2_thermal_transitions_2017_2025_km2.csv")
persist = ((classes["t1"] == 5).astype("uint8") + (classes["t2"] == 5) + (classes["t3"] == 5)).astype("uint8")
p = prof.copy(); p.update(count=1, dtype="uint8", nodata=255)
with rasterio.open(M2 / "hot_persistence_count_30m.tif", "w", **p) as dst:
    dst.write(persist, 1)
print("hot in all three epochs: %.1f km2" % ((persist == 3).sum() * 900 / 1e6))

# figure
fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), constrained_layout=True)
cmap = ListedColormap(["white"] + COLOURS)
for ax, (ep, spec) in zip(axes, EPOCHS.items()):
    ax.imshow(classes[ep], cmap=cmap, vmin=0, vmax=5, interpolation="nearest")
    ax.set_title(f"{spec['year']} dry season"); ax.set_axis_off()
handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in COLOURS]
fig.legend(handles, NAMES.values(), loc="lower center", ncol=5, frameon=False)
fig.suptitle("Kampala thermal classes (Landsat LST anomaly, mean ± 0.5/1.5 SD)")
fig.savefig(FIGURES / "m2_thermal_classes.png", dpi=200)
