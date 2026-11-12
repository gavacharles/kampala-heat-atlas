"""Scenario visuals -> FIGURES/fig7_*.png, anim7_*.gif

1 Before/after trees in 3D (informal close-up), ground coloured by walking heat (UTCI, 14:30)
2 Walking heat maps, CBD and informal, before/after trees
3 Equinox day with the new trees (shadow animation)
4 Rotated frontier: buildings and breeze before/after
5 Solar over a day: CBD roofs lighting up hour by hour, with the city's output curve vs average demand
6 Day/night heat images for the swipe page (MODIS 2025, same sensor and grid)
"""
import json

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyvista as pv
from matplotlib.colors import ListedColormap
from PIL import Image

from kampala import model3d as M
from kampala import viz as V
from kampala.config import DATA, FIGURES, OUTPUTS

pv.OFF_SCREEN = True
V.style()
M6, M7, TAB = DATA / "processed" / "m6", DATA / "processed" / "m7", OUTPUTS / "tables"
SUN = M.sun_positions()
CROP, VEX = 400, 2.0
UTCI_CMAP = V.seq(["#cde2fb", "#fbe7c6", "#f8b89c", "#ec7372", "#c23434", "#701b1b"], "utci")   # 26 -> 38 C
LABEL = {"cbd": "Central business district", "informal": "Informal district (Kawempe)", "frontier": "Growth frontier"}


CENTRE = None   # (row, col) on the 2 m grid; None = district centre


def crop(a, f=1):
    n = a.shape[0] // 1
    k = CROP // (2 * f)
    if CENTRE is None:
        r = c = a.shape[0] // 2
    else:
        r, c = CENTRE[0] // f, CENTRE[1] // f
    r = int(np.clip(r, k, a.shape[0] - k)); c = int(np.clip(c, k, a.shape[1] - k))
    return a[r - k:r + k, c - k:c + k]


def grid(dsm, ter):
    n = dsm.shape[0]
    X, Y = np.meshgrid(np.arange(n) * 2.0, (n - 1 - np.arange(n)) * 2.0)
    return pv.StructuredGrid(X, Y, (dsm - ter.min()) * VEX)


def scene(d, dsm, crown, ground_val, title, clim=(29, 36)):
    p = pv.Plotter(off_screen=True, window_size=(1100, 820)); p.set_background("white")
    ter = crop(d["terrain"]); g = grid(crop(dsm), ter)
    bh = crop(d["bh"]) > 0; cr = crop(crown)
    g["kind"] = np.where(bh, 2.0, np.where(cr, 1.0, 0.0)).ravel(order="F")
    g["utci"] = np.clip(crop(ground_val), *clim).ravel(order="F")
    p.add_mesh(g.threshold(0.5, scalars="kind", invert=True), scalars="utci", cmap=UTCI_CMAP, clim=clim,
               scalar_bar_args=dict(title="Walking heat at 14:30, UTCI (°C)", color="black", vertical=False,
                                    position_x=0.25, position_y=0.03, width=0.5, fmt="%.0f"))
    p.add_mesh(g.threshold([1.5, 2.5], scalars="kind"), color="#c3c2b7")
    if cr.any():
        p.add_mesh(g.threshold([0.5, 1.5], scalars="kind"), color="#2f8f3a")
    p.camera_position = [(-260, -520, 520), (400, 400, 10), (0, 0, 1)]
    p.add_text(title, position="upper_left", font_size=11, color="black")
    img = p.screenshot(return_img=True); p.close()
    return img


# ---------------------------------------------------------------- 1 before/after 3D
d = dict(np.load(M6 / "informal.npz")); t = dict(np.load(M7 / "trees_informal.npz"))
# centre the close-up on the densest cluster of planted trees
_tr = t["trees"]; _dens = [((np.abs(_tr - p_) < 200).all(axis=1)).sum() for p_ in _tr]
CENTRE = tuple(_tr[int(np.argmax(_dens))])
a = scene(d, d["dsm"], np.zeros_like(t["crown"]), t["utci0"], "Today · 21 March, 14:30 · 800 m · heights ×2")
b = scene(d, t["dsm_t"], t["crown"], t["utci1"], f"With {len(t['trees'])} street trees (green) · same hour")
imageio.imwrite(FIGURES / "fig7_trees_before_after_3d.png", np.concatenate([a, b], axis=1))
print("wrote fig7_trees_before_after_3d.png")

# ---------------------------------------------------------------- 2 walking heat, street-level zoom
loc = pd.read_csv(TAB / "m7_tree_local_effect.csv").set_index("district")
fig, axes = plt.subplots(2, 3, figsize=(15, 10.6), constrained_layout=True)
for row, name in enumerate(("cbd", "informal")):
    d = dict(np.load(M6 / f"{name}.npz")); t = dict(np.load(M7 / f"trees_{name}.npz"))
    tr = t["trees"]; dens = [((np.abs(tr - p_) < 75).all(axis=1)).sum() for p_ in tr]
    r0, c0 = tr[int(np.argmax(dens))]; k = 75                      # 300 m window on the densest planting
    r0, c0 = int(np.clip(r0, k, 1000 - k)), int(np.clip(c0, k, 1000 - k))
    win = (slice(r0 - k, r0 + k), slice(c0 - k, c0 + k))
    bh, crown = d["bh"][win] > 0, t["crown"][win]
    for col, (arr, lab, cmap, vmin, vmax) in enumerate((
            (t["utci0"][win], "today", UTCI_CMAP, 28, 36),
            (t["utci1"][win], f"with street trees", UTCI_CMAP, 28, 36),
            (t["utci1"][win] - t["utci0"][win], "change in felt heat", V.seq(["#184f95", "#86b6ef", "#f0efec"], "dU"), -6, 0))):
        ax = axes[row, col]
        im = ax.imshow(np.ma.masked_where(bh, arr), cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.imshow(np.ma.masked_where(~bh, bh), cmap=ListedColormap(["#4f524d"]), interpolation="nearest")
        if col:
            ax.contour(crown.astype(float), levels=[0.5], colors="#1f6b2a", linewidths=0.9)
        V.map_axes(ax, f"{LABEL[name]} · {lab}", km_bar=False)
        ax.plot([8, 58], [140, 140], color=V.INK, lw=2.5); ax.text(33, 136, "100 m", ha="center", fontsize=8)
        if row == 1:
            cb = fig.colorbar(im, ax=ax, location="bottom", shrink=0.8, pad=0.02); cb.outline.set_visible(False)
            cb.set_label("UTCI change (°C)" if col == 2 else "Walking heat at 14:30, UTCI (°C)")
fig.suptitle(f"Street trees take {loc.loc['informal', 'utci_1430_sun'] - loc.loc['informal', 'utci_1430_tree_shade']:.1f} °C off the felt heat where they shade "
             f"(UTCI {loc.loc['informal', 'utci_1430_sun']:.1f} → {loc.loc['informal', 'utci_1430_tree_shade']:.1f} °C, informal district, 14:30, 21 March)\n"
             "300 m street-level windows · green outlines = tree crowns (9 m high, 10 m wide) · grey = buildings", x=0.01, ha="left", fontsize=12, fontweight="bold")
fig.savefig(FIGURES / "fig7_walking_heat_maps.png", dpi=170); plt.close(fig)
print("wrote fig7_walking_heat_maps.png")

# ---------------------------------------------------------------- 3 equinox day with trees
d = dict(np.load(M6 / "informal.npz")); t = dict(np.load(M7 / "trees_informal.npz"))
arc = SUN[SUN.index.month == 3]
frames = []
for i, h in enumerate(t["hours"]):
    sh = crop(t["frames"][i]); ter = crop(d["terrain"])
    g = grid(crop(t["dsm_t"]), ter)
    roof = crop(d["bh"]) > 0; cr = crop(t["crown"])
    g["val"] = np.where(cr, 4.0, np.where(sh, 0.0, 1.0) + np.where(roof, 2.0, 0.0)).ravel(order="F")
    p = pv.Plotter(off_screen=True, window_size=(1100, 800)); p.set_background("white")
    p.add_mesh(g, scalars="val", cmap=["#3a4a63", "#f2d39b", "#5e6b80", "#fbe7c6", "#2f8f3a"], clim=[0, 4], show_scalar_bar=False)
    s = arc[arc.index.hour == h].iloc[0]
    e, az = np.radians(s.elevation), np.radians(s.azimuth)
    sun = np.array([400 + 560 * np.cos(e) * np.sin(az), 400 + 560 * np.cos(e) * np.cos(az), 560 * np.sin(e)])
    p.add_mesh(pv.Sphere(radius=18, center=sun), color="#eda100")
    p.camera_position = [(-300, -560, 560), (400, 400, 10), (0, 0, 1)]
    p.add_text(f"Informal district with {len(t['trees'])} street trees · 21 Mar · {h:02d}:30 · sun {s.elevation:.0f}° high\n"
               "green = tree crowns · dark = shade", position="upper_left", font_size=11, color="black")
    frames.append(p.screenshot(return_img=True)); p.close()
imageio.mimsave(FIGURES / "anim7_trees_shadows.gif", frames, duration=0.7, loop=0)
print("wrote anim7_trees_shadows.gif")

CENTRE = None
# ---------------------------------------------------------------- 4 rotated frontier
r = dict(np.load(M7 / "frontier_rotation.npz")); rs = json.loads((TAB / "m7_rotation_scenario.json").read_text())
fig, axes = plt.subplots(1, 2, figsize=(14, 7.4), constrained_layout=True)
for ax, (bh, sp, solid, lab) in zip(axes, ((r["bh0"], r["speed0"], r["solid0"], f"As built · weak breeze in {rs['stagnant_open_before_pct']:.0f}% of open space"),
                                           (r["bh1"], r["speed1"], r["solid1"], f"All turned east–west · weak breeze in {rs['stagnant_open_after_pct']:.0f}%"))):
    im = ax.imshow(np.ma.masked_where(solid, sp), cmap=V.seq(["#c23434", "#f4a3a2", "#f0efec", "#86b6ef", "#1c5cab"], "spd"),
                   vmin=0, vmax=1.4, extent=(0, 2000, 0, 2000), interpolation="nearest")
    ax.imshow(np.ma.masked_where(bh == 0, bh), cmap=ListedColormap(["#1b1b1f"]), extent=(0, 2000, 0, 2000), interpolation="nearest")
    V.map_axes(ax, lab, km_bar=False)
    ax.annotate("lake breeze from the south ↑", (0.02, 0.02), xycoords="axes fraction", fontsize=9, color=V.INK,
                bbox=dict(fc="white", ec="none", alpha=0.85))
cb = fig.colorbar(im, ax=axes, shrink=0.55, pad=0.01); cb.set_label("breeze speed / free stream (red = stagnant)"); cb.outline.set_visible(False)
fig.suptitle(f"The orientation trade-off at the growth frontier: east–west turns cut wall sun by {rs['wall_penalty_before']:.0f} kWh/m²/yr "
             f"but block the southerly breeze", x=0.01, ha="left", fontsize=13, fontweight="bold")
fig.savefig(FIGURES / "fig7_frontier_rotation.png", dpi=170); plt.close(fig)
print("wrote fig7_frontier_rotation.png")

# ---------------------------------------------------------------- 5 solar over a day (3D roofs + curve)
d = dict(np.load(M6 / "cbd.npz")); hourly = pd.read_csv(TAB / "m7_solar_hourly.csv")
keys = [tuple(k) for k in d["frame_keys"]]
mar = hourly[hourly.month == 3].set_index("hour")
avg_demand = 4674 / 8760    # GW, Umeme 2024 sales spread evenly
frames = []
for h in mar.index:
    row = SUN[(SUN.index.month == 3) & (SUN.index.hour == h)].iloc[0]
    sh = d["frames"][keys.index((3, h))]
    irr = row.k * (row.dni * np.sin(np.radians(row.elevation)) * (~sh) + row.dhi * d["svf"])
    ter = crop(d["terrain"]); g = grid(crop(d["dsm"]), ter)
    g["irr"] = np.where(crop(d["bh"]) > 0, crop(irr), np.nan).ravel(order="F")
    p = pv.Plotter(off_screen=True, window_size=(1100, 760)); p.set_background("white")
    p.add_mesh(g, scalars="irr", cmap=V.seq(["#3a4a63", "#c95222", "#eda100", "#fff3c4"], "pv"), clim=[0, 900],
               nan_color="#e1e0d9", show_scalar_bar=False)
    p.camera_position = [(-260, -520, 520), (400, 400, 10), (0, 0, 1)]
    scene_img = Image.fromarray(p.screenshot(return_img=True)); p.close()
    fig, ax = plt.subplots(figsize=(11, 3.2), dpi=100, constrained_layout=True)
    ax.fill_between(mar.index, mar.city_gw, color=V.ORANGES[1], alpha=0.6)
    ax.plot(mar.index, mar.city_gw, color=V.SERIES[1])
    ax.plot(h, mar.loc[h, "city_gw"], "o", ms=10, color=V.SERIES[1], mec="white", mew=2)
    ax.axhline(avg_demand, color=V.INK, ls="--", lw=1.2)
    ax.text(18.4, avg_demand + 0.4, f"Uganda's average demand {avg_demand:.2f} GW", ha="right", fontsize=9)
    ax.set_xlim(6.5, 18.6); ax.set_ylim(0, 14); ax.set_xticks(range(7, 19))
    ax.set_ylabel("GW"); ax.set_title(f"All 2.0 M Kampala roofs, 21 March · {h:02d}:30 → {mar.loc[h, 'city_gw']:.1f} GW", fontsize=11)
    fig.canvas.draw()
    chart = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()); plt.close(fig)
    canvas = Image.new("RGB", (1100, scene_img.height + chart.height), "white")
    canvas.paste(scene_img, (0, 0)); canvas.paste(chart.resize((1100, chart.height)), (0, scene_img.height))
    frames.append(np.asarray(canvas))
imageio.mimsave(FIGURES / "anim7_solar_day.gif", frames, duration=0.8, loop=0)
print("wrote anim7_solar_day.gif")

# ---------------------------------------------------------------- 6 day/night images (MODIS 2025, 1 km)
import rasterio
with rasterio.open(DATA / "processed" / "m2" / "lst_modis_t3_2025_1km.tif") as src:
    day, night = src.read(1), src.read(2)
for arr, name in ((day, "day"), (night, "night")):
    rel = arr - np.nanmedian(arr)
    rgba = (V.DIVERGING(np.clip((rel + 6) / 12, 0, 1)) * 255).astype("uint8")
    Image.fromarray(rgba).resize((arr.shape[1] * 16, arr.shape[0] * 16), Image.NEAREST).save(FIGURES / f"modis_{name}_2025.png")
print("wrote modis day/night images")
