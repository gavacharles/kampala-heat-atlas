"""3D model visuals -> outputs/figures/fig3d_*.png, anim3d_*.gif

Renders (PyVista, off-screen): each district as a 2 m surface (terrain + buildings) coloured by
annual irradiance, with the sun's path for 21 Jun / 21 Mar / 21 Dec and lake-breeze arrows.
Animations: equinox shadow sweep with a moving sun (CBD, informal), orbit of the CBD,
lake-breeze particles (top view). Orientation: rose of building long axes by settlement type,
wall sun by facing direction, and a district map of the orientation penalty.
"""
import geopandas as gpd
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyvista as pv
from matplotlib.collections import PatchCollection

from kampala import model3d as M
from kampala import viz as V
from kampala.config import DATA, OUTPUTS, FIGURES

pv.OFF_SCREEN = True
V.style()
M6, FIG, TAB = DATA / "processed" / "m6", FIGURES, OUTPUTS / "tables"   # resolve: VTK cannot write through the symlink
LABEL = {"cbd": "Central business district", "informal": "Informal settlement district", "frontier": "Growth frontier"}
SUN = M.sun_positions()
VE = 1.0   # no vertical exaggeration: real heights
# irradiance: shaded (blue) -> open sky (orange), diverging around ~1,300 kWh/m2
IRR_CMAP = V.seq(["#0d366b", "#3987e5", "#cde2fb", "#fde0d2", "#eb6834", "#7a3013"], "irr")


CROP = 400          # cells of 2 m -> 800 m close-up window centred in the district
VEX = 2.0           # vertical exaggeration for the close-ups (stated on every image)


def crop(a, f=1):
    n = a.shape[0]; c = n // 2; k = CROP // (2 * f)
    return a[c - k:c + k, c - k:c + k]


def surface(d, step=1, scalars=None, vex=VEX):
    dsm, ter = crop(d["dsm"])[::step, ::step], crop(d["terrain"])
    n, m = dsm.shape
    x = np.arange(m) * 2.0 * step; y = (n - 1 - np.arange(n)) * 2.0 * step
    X, Y = np.meshgrid(x, y)
    g = pv.StructuredGrid(X, Y, (dsm - ter.min()) * vex)
    if scalars is not None:
        g["val"] = crop(scalars)[::step, ::step].ravel(order="F")
    return g


SIZE = CROP * 2.0
CENTRE = (SIZE / 2, SIZE / 2)


def sun_xyz(az, el, centre=CENTRE, r=560.0):
    a, e = np.radians(az), np.radians(el)
    return np.array([centre[0] + r * np.cos(e) * np.sin(a), centre[1] + r * np.cos(e) * np.cos(a), r * np.sin(e)])


def sun_arcs(plotter):
    for month, col in ((6, V.SERIES[1]), (3, V.SERIES[3]), (12, V.SERIES[0])):
        s = SUN[SUN.index.month == month]
        pts = np.array([sun_xyz(a, e) for a, e in zip(s.azimuth, s.elevation)])
        plotter.add_mesh(pv.Spline(pts, 200).tube(radius=3), color=col)
        noon = s.iloc[np.argmax(s.elevation.values)]
        plotter.add_mesh(pv.Sphere(radius=12, center=sun_xyz(noon.azimuth, noon.elevation)), color=col)


def wind_glyphs(d, every=8):
    u, v, spd, solid = (crop(d[k], 2) for k in ("u", "v", "speed", "solid"))
    ter = crop(d["terrain"])
    n, m = u.shape
    rr, cc = np.mgrid[every // 2:n:every, every // 2:m:every]
    keep = ~solid[rr, cc] & (spd[rr, cc] > 0.05)
    rr, cc = rr[keep], cc[keep]
    z = (ter[np.clip(rr * 2, 0, ter.shape[0] - 1), np.clip(cc * 2, 0, ter.shape[1] - 1)] - ter.min()) * VEX + 3
    pts = pv.PolyData(np.c_[cc * 4.0 + 2, (n - 1 - rr) * 4.0 + 2, z])
    vec = np.c_[u[rr, cc], v[rr, cc], np.zeros(len(rr))]
    pts["vec"] = vec / np.maximum(np.linalg.norm(vec, axis=1, keepdims=True), 1e-6)   # unit arrows
    pts["spd"] = np.clip(spd[rr, cc], 0, 1.5)
    return pts.glyph(orient="vec", scale=False, factor=20, geom=pv.Arrow(tip_length=0.35, tip_radius=0.18, shaft_radius=0.06))


def render_district(name, d):
    p = pv.Plotter(off_screen=True, window_size=(1600, 1150))
    p.set_background("white")
    roof = crop(d["bh"]) > 0
    g = surface(d)
    g["isroof"] = roof.astype(float).ravel(order="F")
    g["shade"] = np.clip(crop(d["shade_hours"]), 0, 3).ravel(order="F")
    g["height"] = np.clip(crop(d["bh"]), 0, 30).ravel(order="F")
    bar = dict(color="black", vertical=False, position_y=0.04, width=0.38, n_labels=4, fmt="%.0f")
    p.add_mesh(g.threshold(0.5, scalars="isroof", invert=True), scalars="shade",
               cmap=V.seq(["#fbe7c6", "#cde2fb", "#86b6ef", "#3987e5", "#184f95"], "g"), clim=[0, 3],
               scalar_bar_args=dict(title="Ground shade, hours a day 10:00-16:00", position_x=0.05, **{**bar, "fmt": "%.1f"}))
    p.add_mesh(g.threshold(0.5, scalars="isroof"), scalars="height", cmap=V.seq(["#e1e0d9", "#898781", "#3b3a37"], "h"),
               clim=[0, 30], scalar_bar_args=dict(title="Building height (m)", position_x=0.5, **bar))
    sun_arcs(p)
    p.add_mesh(wind_glyphs(d), scalars="spd", cmap=V.seq(["#c3c2b7", "#e34948", "#701b1b"], "w"), clim=[0, 1.5],
               scalar_bar_args=dict(title="Breeze / free stream", color="black", vertical=True, position_x=0.86, position_y=0.3, height=0.4, fmt="%.1f"))
    p.camera_position = [(-260, -520, 520), (CENTRE[0], CENTRE[1], 10), (0, 0, 1)]
    p.add_text(f"{LABEL[name]} · 800 × 800 m close-up · heights ×{VEX:.0f}\n"
               "sun paths: Jun (orange), Mar/Sep (yellow), Dec (blue) · arrows: daytime lake breeze from the south",
               position="upper_left", font_size=11, color="black")
    p.screenshot(FIG / f"fig3d_{name}_sun_wind.png")
    p.close()
    print("wrote", f"fig3d_{name}_sun_wind.png")


def shadow_sweep(name, d, month=3):
    keys = [tuple(k) for k in d["frame_keys"]]
    hours = [h for (mo, h) in keys if mo == month]
    frames = []
    base = surface(d)
    roof = crop(d["bh"]) > 0
    arc = SUN[SUN.index.month == month]
    for h in hours:
        sh = crop(d["frames"][keys.index((month, h))])
        p = pv.Plotter(off_screen=True, window_size=(1100, 800))
        p.set_background("white")
        g = base.copy()
        g["val"] = (np.where(sh, 0.0, 1.0) + np.where(roof, 2.0, 0.0)).ravel(order="F")
        p.add_mesh(g, scalars="val", cmap=["#3a4a63", "#f2d39b", "#5e6b80", "#fbe7c6"], clim=[0, 3], show_scalar_bar=False)
        s = arc[arc.index.hour == h].iloc[0]
        p.add_mesh(pv.Spline(np.array([sun_xyz(a, e) for a, e in zip(arc.azimuth, arc.elevation)]), 120).tube(radius=2.5), color="#c9b38b")
        p.add_mesh(pv.Sphere(radius=18, center=sun_xyz(s.azimuth, s.elevation)), color="#eda100")
        p.camera_position = [(-300, -560, 560), (CENTRE[0], CENTRE[1], 10), (0, 0, 1)]
        mon = ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][month]
        p.add_text(f"{LABEL[name]} · 21 {mon} · {h:02d}:30 · sun {s.elevation:.0f}° above the horizon\n"
                   f"800 × 800 m · heights ×{VEX:.0f} · dark = in shadow", position="upper_left", font_size=11, color="black")
        frames.append(p.screenshot(return_img=True))
        p.close()
    imageio.mimsave(FIG / f"anim3d_{name}_shadows_m{month:02d}.gif", frames, duration=0.7, loop=0)
    print("wrote", f"anim3d_{name}_shadows_m{month:02d}.gif")


def orbit(name, d, n=40):
    frames = []
    g = surface(d, scalars=d["irr"])
    for i in range(n):
        p = pv.Plotter(off_screen=True, window_size=(900, 650))
        p.set_background("white")
        p.add_mesh(g, scalars="val", cmap=IRR_CMAP,
                   clim=[600, 2000], show_scalar_bar=False)
        a = np.radians(i * 360 / n)
        p.camera_position = [(CENTRE[0] + 900 * np.sin(a), CENTRE[1] - 900 * np.cos(a), 560), (CENTRE[0], CENTRE[1], 10), (0, 0, 1)]
        p.add_text(f"{LABEL[name]} · annual sunlight on roofs and ground · 800 m · heights x{VEX:.0f}", position="upper_left", font_size=10, color="black")
        frames.append(p.screenshot(return_img=True)); p.close()
    imageio.mimsave(FIG / f"anim3d_{name}_orbit.gif", frames, duration=0.08, loop=0)
    print("wrote", f"anim3d_{name}_orbit.gif")


def wind_particles(name, d, n_part=2500, steps=90):
    u, v, solid = d["u"], d["v"], d["solid"]
    n, m = u.shape
    rng = np.random.default_rng(1)
    py, px = rng.uniform(0, n, n_part), rng.uniform(0, m, n_part)
    fig, ax = plt.subplots(figsize=(6.4, 6.4), constrained_layout=True)
    ax.imshow(solid, cmap=plt.matplotlib.colors.ListedColormap(["#f3f4f1", "#4f524d"]), interpolation="nearest")
    sc = ax.scatter(px, py, s=2, c=V.SERIES[0])
    ax.set_axis_off()
    ax.set_title(f"{LABEL[name]} · lake breeze from the south (potential flow, 4 m)", loc="left", fontsize=10)
    frames = []
    for _ in range(steps):
        iy, ix = np.clip(py.astype(int), 0, n - 1), np.clip(px.astype(int), 0, m - 1)
        px += u[iy, ix] * 3.0; py -= v[iy, ix] * 3.0
        dead = (py < 0) | (py >= n) | (px < 0) | (px >= m) | solid[np.clip(py.astype(int), 0, n - 1), np.clip(px.astype(int), 0, m - 1)]
        dead |= rng.random(n_part) < 0.01
        py[dead] = rng.uniform(n * 0.85, n, dead.sum()); px[dead] = rng.uniform(0, m, dead.sum())
        sc.set_offsets(np.c_[px, py])
        sc.set_color(V.seq(V.BLUE)(np.clip(d["speed"][np.clip(py.astype(int), 0, n - 1), np.clip(px.astype(int), 0, m - 1)], 0, 1.4) / 1.4))
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
    plt.close(fig)
    imageio.mimsave(FIG / f"anim3d_{name}_breeze.gif", frames, duration=0.07, loop=0)
    print("wrote", f"anim3d_{name}_breeze.gif")


if __name__ == "__main__":
    data = {k: dict(np.load(M6 / f"{k}.npz")) for k in ("cbd", "informal", "frontier")}
    import sys
    only3d = "--3d-only" in sys.argv
    for k, d in data.items():
        render_district(k, d)
        if not only3d:
            wind_particles(k, d)
        shadow_sweep(k, d, 3)
    shadow_sweep("informal", data["informal"], 6)
    shadow_sweep("informal", data["informal"], 12)
    orbit("cbd", data["cbd"])
    if only3d:
        raise SystemExit

    # ---------------- orientation rose by settlement type + wall sun curve
    city = pd.read_parquet(M6 / "city_buildings.parquet")
    lookup = pd.read_csv(TAB / "m6_wall_sun_by_azimuth.csv", index_col=0)["kwh_m2_yr"].values
    groups = ["informal-type", "formal dense", "sparse built"]
    fig = plt.figure(figsize=(14, 4.6), constrained_layout=True)
    bins = np.radians(np.arange(0, 181, 10))
    for i, gname in enumerate(groups):
        ax = fig.add_subplot(1, 4, i + 1, projection="polar")
        a = np.radians(city.loc[city.settlement == gname, "az"].values)
        a = np.concatenate([a, a + np.pi])
        h, e = np.histogram(a, bins=np.radians(np.arange(0, 361, 10)))
        ax.bar(e[:-1] + np.radians(5), h / h.sum() * 100, width=np.radians(9), color=V.SERIES[i], edgecolor="white", lw=0.5)
        ax.set_theta_zero_location("N"); ax.set_theta_direction(-1); ax.set_yticklabels([])
        ew = ((city.loc[city.settlement == gname, "az"] > 67.5) & (city.loc[city.settlement == gname, "az"] < 112.5)).mean() * 100
        ax.set_title(f"{gname}\n{ew:.0f}% long axis within 22.5° of E–W", fontsize=10)
    ax = fig.add_subplot(1, 4, 4, projection="polar")
    ax.plot(np.radians(np.arange(361)), np.r_[lookup, lookup[0]], color=V.SERIES[1])
    ax.fill(np.radians(np.arange(361)), np.r_[lookup, lookup[0]], color=V.SERIES[1], alpha=0.15)
    ax.set_theta_zero_location("N"); ax.set_theta_direction(-1)
    ax.set_title("Direct sun on a wall facing each way\n(kWh/m²/yr, ERA5-scaled)", fontsize=10)
    fig.suptitle("Which way do Kampala's buildings face, and what does the sun do to their walls?", x=0.01, ha="left",
                 fontsize=12, fontweight="bold")
    fig.savefig(FIG / "fig3d_orientation_rose.png", dpi=200); plt.close(fig); print("wrote fig3d_orientation_rose.png")

    # ---------------- district map: buildings coloured by orientation penalty
    for k in ("informal", "frontier"):
        b = gpd.read_file(M6 / f"buildings_{k}.gpkg")
        az, L, W = M.building_orientation(b)
        _, pen = M.wall_sun_penalty(az, L, W, lookup)
        b["pen"] = pen
        fig, ax = plt.subplots(figsize=(8, 8), constrained_layout=True)
        b.plot(ax=ax, column="pen", cmap=V.seq(V.REDS[:6]), vmin=0, vmax=max(1, np.nanpercentile(pen, 95)), linewidth=0,
               legend=True, legend_kwds=dict(label="extra direct sun on walls vs an E–W layout (kWh/m²/yr)", shrink=0.6))
        V.map_axes(ax, f"{LABEL[k]}: orientation penalty of each building", km_bar=False)
        fig.savefig(FIG / f"fig3d_{k}_orientation_penalty.png", dpi=200); plt.close(fig); print("wrote", f"fig3d_{k}_orientation_penalty.png")
