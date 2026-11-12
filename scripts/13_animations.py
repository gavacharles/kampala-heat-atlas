"""Thesis animations -> outputs/figures/anim_*.gif

A1 thermal classes 2017 -> 2021 -> 2025 (cross-fade)
A2 urban growth: built-up land appearing epoch by epoch over hillshade
A3 the diurnal wind cycle (ERA5-Land 2015-2024): the Lake Victoria breeze switching on
A4 the Kampala cube rotating
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import ListedColormap, Normalize

from kampala import config
from kampala import viz as V
from kampala.config import DATA, OUTPUTS, FIGURES

V.style()
P, FIG = DATA / "processed", FIGURES
TERR = P / "m0" / "terrain_fabdem_30m.tif"
EP = config.load()["epochs"]
YEARS = [s["year"] for s in EP.values()]
hs, ext = V.read(TERR, 4, step=2)

# ------------------------------------------------------------ A1 thermal class cross-fade
cls = [np.ma.filled(V.read(P / "m2" / f"thermal_classes_{e}_{s['year']}_30m.tif", step=2)[0], 0).astype(float)
       for e, s in EP.items()]
rgb = []
for c in cls:
    img = np.ones(c.shape + (4,))
    for k, col in enumerate(V.THERMAL.colors, start=1):
        img[c == k] = plt.matplotlib.colors.to_rgba(col)
    img[c == 0] = (1, 1, 1, 0)
    rgb.append(img)
frames = []
for i in range(len(rgb) - 1):
    for t in np.linspace(0, 1, 10):
        frames.append(((1 - t) * rgb[i] + t * rgb[i + 1], YEARS[i] if t < 0.5 else YEARS[i + 1]))
    frames += [(rgb[i + 1], YEARS[i + 1])] * 6
frames = [(rgb[0], YEARS[0])] * 6 + frames
fig, ax = plt.subplots(figsize=(6.4, 6.2), constrained_layout=True)
ax.imshow(hs, cmap="gray", extent=ext, alpha=0.35)
im = ax.imshow(frames[0][0], extent=ext, interpolation="nearest")
V.map_axes(ax, None, extent=ext)
title = ax.set_title("", loc="left")
V.legend_patches(ax, V.THERMAL.colors, V.THERMAL_NAMES, title="LST class")


def upd(i):
    im.set_data(frames[i][0]); title.set_text(f"Kampala thermal classes · {frames[i][1]}")
    return im, title


FuncAnimation(fig, upd, frames=len(frames), blit=False).save(FIG / "anim_m2_thermal_classes.gif", writer=PillowWriter(fps=8))
plt.close(fig); print("wrote anim_m2_thermal_classes.gif")

# ------------------------------------------------------------ A2 growth
fr, fext = V.read(P / "m1" / "growth_frontier_10m.tif", step=4)
fr = np.ma.filled(fr, 0)
cols = ["#c3c2b7", V.ORANGES[3], V.REDS[5]]
fig, ax = plt.subplots(figsize=(6.4, 6.2), constrained_layout=True)
ax.imshow(hs, cmap="gray", extent=ext, alpha=0.45)
layer = ax.imshow(np.ma.masked_all(fr.shape), cmap=ListedColormap(cols), vmin=0.5, vmax=3.5, extent=fext, interpolation="nearest")
V.map_axes(ax, None, extent=ext)
title = ax.set_title("", loc="left")
V.legend_patches(ax, cols, [f"built by {YEARS[0]}", f"new {YEARS[0]}–{YEARS[1]}", f"new {YEARS[1]}–{YEARS[2]}"])
seq = [1] * 8 + [2] * 8 + [3] * 14


def upd2(i):
    k = seq[i]
    layer.set_data(np.ma.masked_where((fr == 0) | (fr > k), fr))
    title.set_text(f"Built-up Kampala · {YEARS[k - 1]}")
    return layer, title


FuncAnimation(fig, upd2, frames=len(seq)).save(FIG / "anim_m1_growth.gif", writer=PillowWriter(fps=4))
plt.close(fig); print("wrote anim_m1_growth.gif")

# ------------------------------------------------------------ A3 diurnal wind cycle
w = pd.read_csv(DATA / "interim" / "m0_era5land_wind_kampala.csv")
w["t"] = pd.to_datetime(w["time"], utc=True).dt.tz_convert("Africa/Kampala")
w["hour"] = w["t"].dt.hour
w["sector"] = (((w["dir_from"] + 11.25) % 360) // 22.5).astype(int)
theta = np.radians(np.arange(16) * 22.5)
rmax = 100 * w.groupby("hour")["sector"].value_counts(normalize=True).max() * 1.05
fig = plt.figure(figsize=(5.6, 6))
ax = fig.add_subplot(projection="polar")
ax.set_theta_zero_location("N"); ax.set_theta_direction(-1)


def upd3(h):
    ax.clear(); ax.set_theta_zero_location("N"); ax.set_theta_direction(-1)
    d = w[w.hour == h]
    f = d["sector"].value_counts(normalize=True).reindex(range(16), fill_value=0) * 100
    ax.bar(theta, f.values, width=np.radians(20), color=V.SERIES[0] if 7 <= h < 19 else V.SERIES[6],
           edgecolor="white", linewidth=1)
    ax.set_ylim(0, rmax); ax.set_yticklabels([]); ax.grid(color=V.GRID)
    ax.set_xticks(np.radians([0, 90, 180, 270]), ["N", "E", "S\n(Lake Victoria)", "W"])
    ax.set_title(f"Wind comes FROM… · {h:02d}:00 local\nmean {d['speed'].mean():.1f} m/s · ERA5-Land 2015–2024",
                 loc="center", fontsize=10)


FuncAnimation(fig, upd3, frames=range(24)).save(FIG / "anim_m0_diurnal_wind.gif", writer=PillowWriter(fps=3))
plt.close(fig); print("wrote anim_m0_diurnal_wind.gif")

# ------------------------------------------------------------ A4 rotating cube
z, dsm, lst = V.cube_arrays(P / "m0" / "buildings_2023_30m.tif", TERR, P / "m2" / f"lst_landsat_t3_{YEARS[2]}_30m.tif")
rel = lst - np.nanmedian(lst)
X, Y = np.meshgrid(np.arange(z.shape[1]) * 30, np.arange(z.shape[0])[::-1] * 30)
fig = plt.figure(figsize=(6.4, 5.4))
ax = fig.add_subplot(projection="3d")
ax.plot_surface(X, Y, dsm, facecolors=V.DIVERGING(Normalize(-4, 4)(rel)), rstride=2, cstride=2, linewidth=0,
                antialiased=False, shade=False)
ax.set_box_aspect((1, 1, 0.18)); ax.set_axis_off()
fig.suptitle("The Kampala cube · central Kampala · coloured by 2025 surface heat", fontsize=10, x=0.02, ha="left")


def upd4(i):
    ax.view_init(elev=30, azim=-60 + i * 6)
    return ()


FuncAnimation(fig, upd4, frames=60).save(FIG / "anim_m0_cube_rotate.gif", writer=PillowWriter(fps=12))
plt.close(fig); print("wrote anim_m0_cube_rotate.gif")
