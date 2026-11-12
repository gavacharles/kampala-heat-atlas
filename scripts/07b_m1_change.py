"""Module 1 (new): three-date semantic change detection 2017 -> 2021 -> 2025.

Outputs
  tables   m1_class_area.csv, m1_transitions_<a>_<b>_km2.csv, m1_change_summary.csv
  rasters  growth_frontier_10m.tif   1 = built by 2017, 2 = new built 2017-21, 3 = new built 2021-25
           wetland_loss_10m.tif      wetland in 2017 that became built / cropland by 2025
  points   m1_validation_points_2025.gpkg / .kml  (stratified, 50 per class, for Google Earth Pro;
           fill ref_class by eye -> independent accuracy, replacing the weak-label agreement)
Temporal consistency: built -> vegetation is almost always a classification error at 10 m over 4-8
years; its area is reported as an error indicator rather than silently 'corrected'.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.colors import ListedColormap

from kampala import config, landcover
from kampala.config import DATA, OUTPUTS, FIGURES

M1 = DATA / "processed" / "m1"
EP = config.load()["epochs"]
CL = config.load()["landcover_classes"]
maps, prof = {}, None
for ep, spec in EP.items():
    with rasterio.open(M1 / f"landcover_{ep}_{spec['year']}_10m.tif") as r:
        maps[ep], prof = r.read(1), r.profile
tab = OUTPUTS / "tables"

area = pd.DataFrame({spec["year"]: [(maps[ep] == k).sum() * 100 / 1e6 for k in CL] for ep, spec in EP.items()},
                    index=list(CL.values()))
area.to_csv(tab / "m1_class_area.csv")
print("class area (km2)\n", area.round(1))

summary = []
for a, b in (("t1", "t2"), ("t2", "t3"), ("t1", "t3")):
    ya, yb = EP[a]["year"], EP[b]["year"]
    tm = landcover.transitions(maps[a], maps[b])
    tm.to_csv(tab / f"m1_transitions_{ya}_{yb}_km2.csv")
    built_a = np.isin(maps[a], [1, 2]); built_b = np.isin(maps[b], [1, 2])
    summary.append(dict(period=f"{ya}-{yb}",
                        new_built_km2=(~built_a & built_b & (maps[a] > 0)).sum() * 1e-4,
                        sparse_to_dense_km2=((maps[a] == 2) & (maps[b] == 1)).sum() * 1e-4,
                        wetland_to_built_km2=((maps[a] == 6) & built_b).sum() * 1e-4,
                        tree_to_built_km2=((maps[a] == 3) & built_b).sum() * 1e-4,
                        built_to_veg_km2_error_indicator=(built_a & np.isin(maps[b], [3, 4, 6])).sum() * 1e-4))
summ = pd.DataFrame(summary)
summ.to_csv(tab / "m1_change_summary.csv", index=False)
print(summ.round(2).to_string(index=False))

# growth frontier and wetland loss rasters
b1, b2, b3 = (np.isin(maps[e], [1, 2]) for e in ("t1", "t2", "t3"))
front = np.zeros(b1.shape, dtype="uint8")
front[b1] = 1
front[~b1 & b2] = 2
front[~b1 & ~b2 & b3] = 3
wet_loss = ((maps["t1"] == 6) & np.isin(maps["t3"], [1, 2, 4, 5])).astype("uint8")
p = prof.copy(); p.update(dtype="uint8", nodata=0, count=1, compress="deflate")
for name, arr in (("growth_frontier_10m.tif", front), ("wetland_loss_10m.tif", wet_loss)):
    with rasterio.open(M1 / name, "w", **p) as dst:
        dst.write(arr, 1)

# validation points for manual interpretation
pts = landcover.validation_points(M1 / f"landcover_t3_{EP['t3']['year']}_10m.tif")
pts.to_file(OUTPUTS / "tables" / "m1_validation_points_2025.gpkg", driver="GPKG")
pts.rename(columns={"map_name": "Name"}).to_file(OUTPUTS / "tables" / "m1_validation_points_2025.kml", driver="KML")
print(f"validation points: {len(pts)}")

# figure: 2017 / 2025 maps + growth frontier
cols = ["#d7301f", "#fc8d59", "#1a9850", "#d9ef8b", "#bdbdbd", "#41b6c4", "#2c7fb8"]
fig, ax = plt.subplots(1, 3, figsize=(16, 5.5), constrained_layout=True)
step = 4  # display decimation
for a_, ep in zip(ax[:2], ("t1", "t3")):
    a_.imshow(maps[ep][::step, ::step], cmap=ListedColormap(["white"] + cols), vmin=0, vmax=7, interpolation="nearest")
    a_.set_title(f"Land cover {EP[ep]['year']}"); a_.set_axis_off()
ax[2].imshow(front[::step, ::step], cmap=ListedColormap(["white", "#969696", "#fd8d3c", "#bd0026"]), vmin=0, vmax=3,
             interpolation="nearest")
ax[2].set_title("Growth frontier"); ax[2].set_axis_off()
h = [plt.Rectangle((0, 0), 1, 1, color=c) for c in cols]
fig.legend(h, list(CL.values()), loc="lower left", ncol=7, frameon=False)
h2 = [plt.Rectangle((0, 0), 1, 1, color=c) for c in ("#969696", "#fd8d3c", "#bd0026")]
fig.legend(h2, ["built by 2017", "new 2017-21", "new 2021-25"], loc="lower right", frameon=False)
fig.savefig(FIGURES / "m1_landcover_change.png", dpi=200)
