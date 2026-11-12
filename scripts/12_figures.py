"""Thesis figures (static maps + charts) -> outputs/figures/fig_*.png

Maps : thermal classes x3, LST change 2017->2025, land cover + growth, informal-type settlements
       on 2025 LST, tree-planting priority, rooftop solar density, 3D cube of the CBD.
Charts: LST by LCZ (dumbbell), settlement thermal trajectories, land-cover change (slope),
        solar adoption vs demand, tree-priority people & plantable land.
"""
import duckdb
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.colors import ListedColormap
from rasterio.warp import Resampling, reproject
from shapely import wkb

from kampala import config
from kampala import viz as V
from kampala.config import DATA, OUTPUTS, FIGURES

V.style()
P, FIG, TAB = DATA / "processed", FIGURES, OUTPUTS / "tables"
TERR = P / "m0" / "terrain_fabdem_30m.tif"
EP = config.load()["epochs"]
YEARS = [s["year"] for s in EP.values()]


def save(fig, name):
    fig.savefig(FIG / name, dpi=220)
    plt.close(fig)
    print("wrote", name)


# ---------------------------------------------------------------- 1 thermal classes x3
fig, axes = plt.subplots(1, 3, figsize=(15, 5.6), constrained_layout=True)
for ax, (ep, s) in zip(axes, EP.items()):
    a, ext = V.read(P / "m2" / f"thermal_classes_{ep}_{s['year']}_30m.tif")
    V.hillshade_under(ax, TERR)
    ax.imshow(V.masked(a), cmap=V.THERMAL, vmin=0.5, vmax=5.5, extent=ext, interpolation="nearest", alpha=0.92)
    V.map_axes(ax, f"{s['year']} dry season", extent=ext)
V.legend_patches(axes[-1], V.THERMAL.colors, V.THERMAL_NAMES, title="LST anomaly class")
fig.suptitle("Kampala thermal classes, Landsat land surface temperature (mean ± 0.5 / 1.5 SD)",
             x=0.01, ha="left", fontsize=13, fontweight="semibold", color=V.INK)
save(fig, "fig_m2_thermal_classes.png")

# ---------------------------------------------------------------- 2 LST anomaly change
a1, ext = V.read(P / "m2" / f"lst_landsat_t1_{YEARS[0]}_30m.tif", 2)
a3, _ = V.read(P / "m2" / f"lst_landsat_t3_{YEARS[2]}_30m.tif", 2)
lc3, _ = V.read(P / "m1" / f"landcover_t3_{YEARS[2]}_10m.tif", 1, step=3)
d = (a3 - a1)
water = np.zeros(d.shape, bool)
h, w = min(d.shape[0], lc3.shape[0]), min(d.shape[1], lc3.shape[1])
water[:h, :w] = np.ma.filled(lc3, 0)[:h, :w] == 7
d = np.ma.masked_where(water, d)
fig, ax = plt.subplots(figsize=(9, 8.2), constrained_layout=True)
V.hillshade_under(ax, TERR)
im = ax.imshow(d, cmap=V.DIVERGING, vmin=-4, vmax=4, extent=ext, interpolation="nearest", alpha=0.9)
V.map_axes(ax, f"Change in relative surface heat, {YEARS[0]} → {YEARS[2]}", extent=ext)
ax.text(0.01, -0.02, "Difference of LST anomalies (each epoch relative to its own land mean). The western third relies most on "
        "the gap-filled TOA LST;\nbroad east-west banding there may include compositing differences, so read local contrasts, "
        "not the regional band.", transform=ax.transAxes, fontsize=7.5, color=V.INK2, va="top")
cb = fig.colorbar(im, ax=ax, shrink=0.6, pad=0.01)
cb.set_label("Δ LST anomaly (°C)  ·  red = warmed relative to the city")
cb.outline.set_visible(False)
save(fig, "fig_m2_lst_change.png")

# ---------------------------------------------------------------- 3 land cover 2025 + growth frontier
lc, ext = V.read(P / "m1" / f"landcover_t3_{YEARS[2]}_10m.tif", 1, step=2)
fr, _ = V.read(P / "m1" / "growth_frontier_10m.tif", 1, step=2)
fig, axes = plt.subplots(1, 2, figsize=(15, 7.2), constrained_layout=True)
axes[0].imshow(V.masked(lc), cmap=ListedColormap(V.LANDCOVER), vmin=0.5, vmax=7.5, extent=ext, interpolation="nearest")
V.map_axes(axes[0], f"Land cover {YEARS[2]} (Sentinel-2, 10 m)", extent=ext)
V.legend_patches(axes[0], V.LANDCOVER, V.LC_NAMES)
V.hillshade_under(axes[1], TERR, alpha=0.45)
axes[1].imshow(V.masked(fr), cmap=ListedColormap(["#c3c2b7", V.ORANGES[3], V.REDS[5]]), vmin=0.5, vmax=3.5,
               extent=ext, interpolation="nearest")
V.map_axes(axes[1], "Where Kampala grew", extent=ext)
V.legend_patches(axes[1], ["#c3c2b7", V.ORANGES[3], V.REDS[5]],
                 [f"built by {YEARS[0]}", f"new {YEARS[0]}–{YEARS[1]}", f"new {YEARS[1]}–{YEARS[2]}"])
save(fig, "fig_m1_landcover_growth.png")

# ---------------------------------------------------------------- 4 informal-type settlements on LST
lst3, ext = V.read(P / "m2" / f"lst_landsat_t3_{YEARS[2]}_30m.tif", 2)
g, gext = V.read(P / "m3" / f"settlement_groups_t3_{YEARS[2]}_90m.tif")
cx, cy, half = 452000, 34500, 9000      # UTM 36N window around central-north Kampala
fig, ax = plt.subplots(figsize=(9.5, 9), constrained_layout=True)
im = ax.imshow(lst3, cmap=V.DIVERGING, vmin=-8, vmax=8, extent=ext, interpolation="nearest", alpha=0.75)
inf = np.ma.masked_not_equal(np.ma.filled(g, 0), 1)
ax.imshow(inf, cmap=ListedColormap([V.INK]), extent=gext, interpolation="nearest", alpha=0.85)
ax.set_xlim(cx - half, cx + half); ax.set_ylim(cy - half, cy + half)
con = duckdb.connect(); con.sql("INSTALL httpfs; LOAD httpfs; SET s3_region='us-west-2';")
names = ["Kisenyi", "Katanga", "Bwaise", "Kalerwe", "Kivulu", "Namuwongo", "Kifumbira", "Kazo Angola", "Mulago", "Kawempe Mbogo"]
q = " OR ".join([f"names.primary = '{n}'" for n in names])
df = con.sql(f"""SELECT names.primary AS name, geometry FROM read_parquet(
  's3://overturemaps-us-west-2/release/2026-10-21.0/theme=divisions/type=division/*', hive_partitioning=1)
  WHERE bbox.xmin > 32.45 AND bbox.xmax < 32.70 AND bbox.ymin > 0.25 AND bbox.ymax < 0.42 AND ({q})""").df()
pts = gpd.GeoDataFrame(df, geometry=df.pop("geometry").apply(lambda x: wkb.loads(bytes(x))), crs="EPSG:4326")
pts = pts.to_crs(config.load()["study_area"]["crs"])
pts["d"] = pts.geometry.centroid.distance(gpd.points_from_xy([451300], [34400])[0])   # Kampala Road
pts = pts.sort_values("d").drop_duplicates("name")  # several places share a name; keep the central one
for _, p in pts.iterrows():
    c = p.geometry.centroid
    ax.annotate(p["name"], (c.x, c.y), xytext=(4, 4), textcoords="offset points", fontsize=8, color=V.INK,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8))
    ax.plot(c.x, c.y, "o", ms=4, color=V.INK, mec="white", mew=1)
V.map_axes(ax, f"Informal-type morphology (black) on {YEARS[2]} surface heat", km_bar=False)
cb = fig.colorbar(im, ax=ax, shrink=0.55, pad=0.01); cb.set_label("LST anomaly (°C)"); cb.outline.set_visible(False)
save(fig, "fig_m3_informal_on_heat.png")

# ---------------------------------------------------------------- 5 tree priority
tp, ext = V.read(P / "m4" / "tree_priority_classes_30m.tif")
fig, ax = plt.subplots(figsize=(9, 8.2), constrained_layout=True)
V.hillshade_under(ax, TERR, alpha=0.4)
cols = [V.GREENS[1], V.GREENS[2], V.GREENS[4], V.GREENS[6]]
ax.imshow(V.masked(tp), cmap=ListedColormap(cols), vmin=0.5, vmax=4.5, extent=ext, interpolation="nearest")
V.map_axes(ax, "Where to plant trees: heat × people × canopy deficit, where there is room", extent=ext)
V.legend_patches(ax, cols, ["low", "medium", "high", "very high"], title="Planting priority")
save(fig, "fig_m4_tree_priority.png")

# ---------------------------------------------------------------- 6 rooftop solar density (GWh / km2 on 250 m grid)
sol = pd.read_parquet(P / "m5" / "rooftop_solar_buildings.parquet")
b = gpd.read_parquet(DATA / "raw" / "overture_buildings.parquet", columns=["geometry"]).to_crs(config.load()["study_area"]["crs"])
c = b.geometry.centroid.loc[sol.index]
with rasterio.open(P / "m0" / "terrain_fabdem_30m.tif") as r:
    ext = (r.bounds.left, r.bounds.right, r.bounds.bottom, r.bounds.top)
cell = 250
nx, ny = int((ext[1] - ext[0]) // cell), int((ext[3] - ext[2]) // cell)
ix = ((c.x.values - ext[0]) // cell).astype(int); iy = ((ext[3] - c.y.values) // cell).astype(int)
ok = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
grid = np.zeros((ny, nx))
np.add.at(grid, (iy[ok], ix[ok]), sol["kwh_yr"].values[ok] / 1e6 / (cell * cell / 1e6))  # GWh per km2
fig, ax = plt.subplots(figsize=(9, 8.2), constrained_layout=True)
V.hillshade_under(ax, TERR, alpha=0.35)
im = ax.imshow(np.ma.masked_less(grid, 0.5), cmap=V.seq(V.ORANGES, "sol"), vmin=0, vmax=np.nanpercentile(grid[grid > 0], 99),
               extent=(ext[0], ext[0] + nx * cell, ext[3] - ny * cell, ext[3]), interpolation="nearest")
V.map_axes(ax, "Rooftop solar technical potential", extent=ext)
cb = fig.colorbar(im, ax=ax, shrink=0.6, pad=0.01); cb.set_label("GWh per km² per year"); cb.outline.set_visible(False)
save(fig, "fig_m5_solar_density.png")

# ---------------------------------------------------------------- 7 3D cube of the CBD (terrain + buildings, coloured by LST)
from matplotlib import cm
from matplotlib.colors import Normalize


if __name__ == "__main__":
    z, dsm, lst = V.cube_arrays(P / "m0" / "buildings_2023_30m.tif", TERR, P / "m2" / f"lst_landsat_t3_{YEARS[2]}_30m.tif")
    X, Y = np.meshgrid(np.arange(z.shape[1]) * 30, np.arange(z.shape[0])[::-1] * 30)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(projection="3d")
    rel = lst - np.nanmedian(lst)                       # heat relative to central Kampala itself
    norm = Normalize(-4, 4)
    ax.plot_surface(X, Y, dsm, facecolors=V.DIVERGING(norm(rel)), rstride=1, cstride=1, linewidth=0, antialiased=False, shade=False)
    ax.set_box_aspect((1, 1, 0.18)); ax.view_init(elev=32, azim=-60)
    ax.set_axis_off()
    m = cm.ScalarMappable(norm=norm, cmap=V.DIVERGING); m.set_array([])
    cb = fig.colorbar(m, ax=ax, shrink=0.5, pad=0.02); cb.set_label("LST 2025 relative to the window median (°C)"); cb.outline.set_visible(False)
    ax.set_title("The Kampala cube: FABDEM terrain + Open Buildings heights, coloured by surface heat\n"
                 "central Kampala, 5.4 × 5.4 km; heights capped at 45 m; vertical exaggerated", loc="left", fontsize=11)
    save(fig, "fig_m0_cube_cbd.png")

    # ------------------------------------------------------------ charts
    # C1 LST by LCZ dumbbell
    l = pd.read_csv(TAB / "m2_lst_by_lcz.csv").pivot(index="lcz", columns="year", values="LST_anom_mean")
    l = l.sort_values(YEARS[2])
    fig, ax = plt.subplots(figsize=(8, 5.2), constrained_layout=True)
    yy = np.arange(len(l))
    ax.hlines(yy, l[YEARS[0]], l[YEARS[2]], color=V.AXIS, lw=2, zorder=1)
    ax.scatter(l[YEARS[0]], yy, s=60, color=V.SERIES[0], edgecolor="white", lw=1.5, zorder=2, label=str(YEARS[0]))
    ax.scatter(l[YEARS[2]], yy, s=60, color=V.SERIES[1], edgecolor="white", lw=1.5, zorder=3, label=str(YEARS[2]))
    ax.set_yticks(yy, l.index); ax.axvline(0, color=V.AXIS, lw=1)
    ax.set_xlabel("LST anomaly vs city land mean (°C)"); ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right"); ax.set_title("Surface heat by Local Climate Zone")
    save(fig, "fig_m2_lcz_dumbbell.png")

    # C2 settlement trajectories
    t = pd.read_csv(TAB / "m3_thermal_trajectories.csv", index_col=0)
    fig, ax = plt.subplots(figsize=(7.5, 4.8), constrained_layout=True)
    for i, (name, row) in enumerate(t.iterrows()):
        vals = [row[str(y)] for y in YEARS]
        ax.plot(YEARS, vals, color=V.SERIES[i], marker="o", ms=8, mec="white", mew=1.5, label=name)
        ax.annotate(f"{name}  ({row['change_2017_2025']:+.1f} °C)", (YEARS[-1], vals[-1]), xytext=(8, 0),
                    textcoords="offset points", va="center", fontsize=9, color=V.INK2)
    ax.set_xticks(YEARS); ax.set_xlim(YEARS[0] - 0.5, YEARS[-1] + 5.5)
    ax.set_ylabel("Mean LST anomaly (°C)"); ax.legend(loc="upper left", fontsize=8)
    ax.set_title("Thermal trajectories by settlement type")
    save(fig, "fig_m3_trajectories.png")

    # C3 land cover slope chart
    a = pd.read_csv(TAB / "m1_class_area.csv", index_col=0)
    fig, ax = plt.subplots(figsize=(7, 5.5), constrained_layout=True)
    for i, (cls, row) in enumerate(a.iterrows()):
        col = V.LANDCOVER[i]
        ax.plot([0, 1, 2], row.values, color=col, marker="o", ms=7, mec="white", mew=1.2)
        ax.annotate(f"{V.LC_NAMES[i]}  {row.iloc[-1]:.0f} km² ({row.iloc[-1] - row.iloc[0]:+.0f})", (2, row.iloc[-1]),
                    xytext=(8, 0), textcoords="offset points", va="center", fontsize=8.5, color=V.INK2)
    ax.set_xticks([0, 1, 2], [str(y) for y in YEARS]); ax.set_xlim(-0.2, 3.3)
    ax.set_ylabel("Area (km²)"); ax.set_title("Land-cover change in the study area")
    save(fig, "fig_m1_class_change.png")

    # C4 solar adoption vs demand
    s = pd.read_csv(TAB / "m5_solar_scenarios.csv")
    fig, ax = plt.subplots(figsize=(7.5, 4.6), constrained_layout=True)
    x = np.arange(len(s))
    ax.bar(x - 0.2, s["largest_first_GWh"] / 1000, 0.38, color=V.SERIES[1], label="largest roofs first")
    ax.bar(x + 0.2, s["random_GWh"] / 1000, 0.38, color=V.SERIES[0], label="random roofs")
    ax.axhline(4.674, color=V.INK, lw=1.2, ls="--")
    ax.text(len(s) - 0.5, 4.674, " national sales 2024 (Umeme): 4.7 TWh", va="bottom", ha="right", fontsize=8.5, color=V.INK)
    ax.set_xticks(x, s["adoption"]); ax.set_ylabel("Annual generation (TWh)"); ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left"); ax.set_title("Rooftop solar: technical potential under adoption scenarios")
    save(fig, "fig_m5_scenarios.png")

    # C5 tree priority people / plantable
    tpz = pd.read_csv(TAB / "m4_priority_summary.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    cols = [V.GREENS[1], V.GREENS[2], V.GREENS[4], V.GREENS[6]]
    for ax, col, lab in ((axes[0], "people", "Residents (thousands)"), (axes[1], "plantable_km2", "Plantable land (km²)")):
        vals = tpz[col] / (1000 if col == "people" else 1)
        ax.barh(tpz["priority"], vals, color=cols, edgecolor="white")
        for yv, v in enumerate(vals):
            ax.text(v, yv, f" {v:,.0f}", va="center", fontsize=9, color=V.INK2)
        ax.set_xlabel(lab); ax.grid(axis="y", visible=False)
    axes[0].set_title("Who lives in each planting-priority class")
    save(fig, "fig_m4_priority_bars.png")
