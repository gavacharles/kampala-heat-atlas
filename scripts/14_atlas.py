"""Build the interactive 'Kampala Heat Atlas' (single self-contained HTML page).

    python scripts/14_atlas.py <output_dir>

Every raster layer is rendered as a transparent PNG on the shared 30 m grid (same bounds), shown
over the FABDEM hillshade in Leaflet (CRS.Simple, pan/zoom). Headline numbers are read from the
result tables so the page always matches the latest run.
"""
import base64
import io
import json
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from matplotlib.colors import to_rgba
from PIL import Image

from kampala import config
from kampala import viz as V
from kampala.config import DATA, OUTPUTS, FIGURES

OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
P, TAB, FIG = DATA / "processed", OUTPUTS / "tables", FIGURES
EP = config.load()["epochs"]; Y = [s["year"] for s in EP.values()]

with rasterio.open(P / "m0" / "terrain_fabdem_30m.tif") as r:
    H, W, T = r.height, r.width, r.transform
    hs = r.read(4).astype(float)


def fit(a):
    """Crop / pad any array to the master grid shape."""
    out = np.zeros((H, W), dtype=a.dtype)
    h, w = min(H, a.shape[0]), min(W, a.shape[1])
    out[:h, :w] = a[:h, :w]
    return out


def png(rgba, max_w=1100):
    im = Image.fromarray((np.clip(rgba, 0, 1) * 255).astype("uint8"), "RGBA")
    if im.width > max_w:
        im = im.resize((max_w, round(im.height * max_w / im.width)), Image.NEAREST)
    buf = io.BytesIO(); im.save(buf, "PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def classes(a, colors):
    rgba = np.zeros(a.shape + (4,))
    for k, c in enumerate(colors, start=1):
        rgba[a == k] = to_rgba(c)
    return rgba


def continuous(a, cmap, vmin, vmax, mask):
    rgba = cmap(np.clip((a - vmin) / (vmax - vmin), 0, 1))
    rgba[mask] = 0
    return rgba


def rd(path, band=1, step=1):
    with rasterio.open(path) as r:
        return r.read(band)[::step, ::step]


layers = {}
g = np.clip((hs - np.nanpercentile(hs, 2)) / (np.nanpercentile(hs, 98) - np.nanpercentile(hs, 2)), 0, 1)
base = np.dstack([0.55 + 0.4 * g] * 3 + [np.ones_like(g)])
_im = Image.fromarray((base[..., :3] * 255).astype("uint8"), "RGB")
_im = _im.resize((1100, round(_im.height * 1100 / _im.width)), Image.LANCZOS)
_buf = io.BytesIO(); _im.save(_buf, "JPEG", quality=78)
BASE = "data:image/jpeg;base64," + base64.b64encode(_buf.getvalue()).decode()   # opaque base -> JPEG

for ep, s in EP.items():
    layers[f"thermal-{s['year']}"] = png(classes(fit(rd(P / "m2" / f"thermal_classes_{ep}_{s['year']}_30m.tif")), V.THERMAL.colors))
a1 = fit(rd(P / "m2" / f"lst_landsat_t1_{Y[0]}_30m.tif", 2)); a3 = fit(rd(P / "m2" / f"lst_landsat_t3_{Y[2]}_30m.tif", 2))
lc25 = fit(rd(P / "m1" / f"landcover_t3_{Y[2]}_10m.tif", 1, 3))
layers["lst-change"] = png(continuous(a3 - a1, V.DIVERGING, -4, 4, (lc25 == 7) | ~np.isfinite(a3 - a1)))
layers["lst-2025"] = png(continuous(a3, V.DIVERGING, -8, 8, (lc25 == 7) | ~np.isfinite(a3)))
layers["landcover-2017"] = png(classes(fit(rd(P / "m1" / f"landcover_t1_{Y[0]}_10m.tif", 1, 3)), V.LANDCOVER))
layers["landcover-2025"] = png(classes(lc25, V.LANDCOVER))
layers["growth"] = png(classes(fit(rd(P / "m1" / "growth_frontier_10m.tif", 1, 3)), ["#a9a8a0", V.ORANGES[3], V.REDS[5]]))
inf = np.kron(rd(P / "m3" / f"settlement_groups_t3_{Y[2]}_90m.tif"), np.ones((3, 3), dtype="uint8"))
layers["informal-2025"] = png(classes(fit((inf == 1).astype("uint8")), ["#1b1b1f"]))
layers["trees"] = png(classes(fit(rd(P / "m4" / "tree_priority_classes_30m.tif")),
                              [V.GREENS[1], V.GREENS[2], V.GREENS[4], V.GREENS[6]]))

# rooftop solar density: GWh per km2 on 240 m blocks of the master grid
sol = pd.read_parquet(P / "m5" / "rooftop_solar_buildings.parquet")
b = gpd.read_parquet(DATA / "raw" / "overture_buildings.parquet", columns=["geometry"]).to_crs(config.load()["study_area"]["crs"])
c = b.geometry.centroid.loc[sol.index]
rows, cols = rasterio.transform.rowcol(T, c.x.values, c.y.values)
rows, cols = np.asarray(rows), np.asarray(cols)
ok = (rows >= 0) & (rows < H) & (cols >= 0) & (cols < W)
grid = np.zeros((H, W)); np.add.at(grid, (rows[ok], cols[ok]), sol["kwh_yr"].values[ok])
k = 8
hb, wb = H // k, W // k
blk = grid[:hb * k, :wb * k].reshape(hb, k, wb, k).sum(axis=(1, 3)) / 1e6 / ((k * 30) ** 2 / 1e6)
sd = fit(np.kron(blk, np.ones((k, k))))
layers["solar"] = png(continuous(sd, V.seq(V.ORANGES), 0, float(np.percentile(blk[blk > 0], 99)), sd < 0.5))
solar_max = float(np.percentile(blk[blk > 0], 99))


def img_b64(path, width=900):
    im = Image.open(path)
    if im.width > width:
        im = im.resize((width, round(im.height * width / im.width)), Image.LANCZOS)
    buf = io.BytesIO(); im.convert("RGB").quantize(256).save(buf, "PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def gif_b64(path, width=520, every=1):
    im = Image.open(path); frames, durs = [], []
    try:
        while True:
            if im.tell() % every == 0:
                f = im.convert("RGB"); frames.append(f.resize((width, round(f.height * width / f.width)), Image.LANCZOS))
                durs.append(im.info.get("duration", 100) * every)
            im.seek(im.tell() + 1)
    except EOFError:
        pass
    buf = io.BytesIO()
    frames[0].save(buf, "GIF", save_all=True, append_images=frames[1:], duration=durs, loop=0, optimize=True)
    return "data:image/gif;base64," + base64.b64encode(buf.getvalue()).decode()


charts = {n: img_b64(FIG / f) for n, f in {
    "lcz": "fig_m2_lcz_dumbbell.png", "traj": "fig_m3_trajectories.png", "class": "fig_m1_class_change.png",
    "solar": "fig_m5_scenarios.png", "trees": "fig_m4_priority_bars.png", "cube": "fig_m0_cube_cbd.png"}.items()}
anims = {"wind": gif_b64(FIG / "anim_m0_diurnal_wind.gif", 440), "cube": gif_b64(FIG / "anim_m0_cube_rotate.gif", 460, every=2)}
for _k, _v in {**{f"layer:{a}": b for a, b in layers.items()}, **{f"chart:{a}": b for a, b in charts.items()},
               **{f"anim:{a}": b for a, b in anims.items()}, "base": BASE}.items():
    print(f"  {_k:24s} {len(_v) / 1e6:.2f} MB")

# ---- numbers
area = pd.read_csv(TAB / "m1_class_area.csv", index_col=0)
built = area.loc[["built_dense", "built_sparse"]].sum()
th = pd.read_csv(TAB / "m2_thermal_class_area.csv")
hot = th[th.name == "hot"].set_index("year")["km2"]
traj = pd.read_csv(TAB / "m3_thermal_trajectories.csv", index_col=0)
grp = pd.read_csv(TAB / "m3_lst_by_settlement_group.csv")
g25 = grp[grp.year == Y[2]].set_index("group")["LST_anom_mean"]
tree = pd.read_csv(TAB / "m4_planting_scenario.csv", index_col=0)["value"]
pri = pd.read_csv(TAB / "m4_priority_summary.csv").set_index("priority")
st = pd.read_csv(TAB / "m5_solar_totals.csv", index_col=0)["value"]
cal = pd.read_csv(TAB / "m2_lst_calibration.csv")
diag = [json.loads((TAB / f"m1_{e}_diagnostics.json").read_text()) for e in EP]
num = dict(
    built0=f"{built.iloc[0]:.0f}", built2=f"{built.iloc[2]:.0f}", builtpct=f"{100 * (built.iloc[2] / built.iloc[0] - 1):.0f}",
    hot0=f"{hot.loc[Y[0]]:.0f}", hot2=f"{hot.loc[Y[2]]:.0f}",
    inf=f"{g25['informal-type']:.1f}", formal=f"{g25['formal dense']:.1f}",
    became=f"{traj.loc['became informal-type', 'change_2017_2025']:+.1f}",
    formalchg=f"{traj.loc['always formal dense', 'change_2017_2025']:+.1f}",
    cool10=f"{tree['cooling_per_10pct_canopy_C']:.2f}", treecool=f"{tree['mean_cooling_C']:.1f}",
    treepeople=f"{tree['people_in_planted_cells'] / 1000:.0f}", newcanopy=f"{tree['new_canopy_km2']:.0f}",
    vh_people=f"{pri.loc['very high', 'people'] / 1000:.0f}", vh_km2=f"{pri.loc['very high', 'km2']:.0f}",
    gwp=f"{st['capacity_GWp']:.0f}", twh=f"{st['generation_GWh_yr'] / 1000:.0f}", roofs=f"{st['buildings'] / 1e6:.1f}",
    yield_=f"{st['specific_yield_kWh_per_kWp']:,.0f}",
    r_min=f"{cal.r_vs_C2ST.min():.2f}", r_max=f"{cal.r_vs_C2ST.max():.2f}",
    agree=f"{min(d['label_agreement'] for d in diag) * 100:.0f}–{max(d['label_agreement'] for d in diag) * 100:.0f}",
)

LAYERS = [
    # id, group, label, legend-type, legend payload, caption
    ("thermal-2025", "Heat", f"Thermal classes {Y[2]}", "classes", list(zip(V.THERMAL.colors, V.THERMAL_NAMES)),
     "Landsat 8/9 land surface temperature, dry-season median over three years, classed by distance from the city land mean (±0.5 and ±1.5 standard deviations)."),
    ("thermal-2021", "Heat", f"Thermal classes {Y[1]}", "classes", list(zip(V.THERMAL.colors, V.THERMAL_NAMES)), "Same method, 2021 window."),
    ("thermal-2017", "Heat", f"Thermal classes {Y[0]}", "classes", list(zip(V.THERMAL.colors, V.THERMAL_NAMES)), "Same method, 2017 window."),
    ("lst-2025", "Heat", f"Surface heat {Y[2]} (°C)", "ramp", ["-8 °C", "city mean", "+8 °C", "diverging"],
     "Land surface temperature minus the city land mean. Gap-free: computed from the thermal band with NDVI-based emissivity and calibrated to the Collection 2 product."),
    ("lst-change", "Heat", f"Heat change {Y[0]}–{Y[2]}", "ramp", ["cooled 4 °C", "no change", "warmed 4 °C", "diverging"],
     "Difference of the two relative-heat maps. Read local contrasts; broad regional bands in the west may include compositing differences."),
    ("landcover-2025", "Growth", f"Land cover {Y[2]}", "classes", list(zip(V.LANDCOVER, V.LC_NAMES)),
     "Sentinel-2 random forest, 10 m, trained on Dynamic World labels; built-up requires building evidence within about 30 m."),
    ("landcover-2017", "Growth", f"Land cover {Y[0]}", "classes", list(zip(V.LANDCOVER, V.LC_NAMES)), "Same method, 2017 (Sentinel-2 L1C)."),
    ("growth", "Growth", "Where the city grew", "classes",
     list(zip(["#a9a8a0", V.ORANGES[3], V.REDS[5]], [f"built by {Y[0]}", f"new {Y[0]}–{Y[1]}", f"new {Y[1]}–{Y[2]}"])),
     "First epoch in which each 10 m pixel is mapped as built-up."),
    ("informal-2025", "People", "Informal-type settlement", "classes", [("#1b1b1f", "informal-type morphology")],
     "90 m cells with ≥35 % roof cover, mean footprint ≤80 m², ≥30 buildings/ha, low roofs, in clusters ≥2.4 ha. A morphology proxy that misses mixed-use areas such as Kisenyi."),
    ("trees", "Response", "Tree-planting priority", "classes",
     list(zip([V.GREENS[1], V.GREENS[2], V.GREENS[4], V.GREENS[6]], ["low", "medium", "high", "very high"])),
     "Heat rank × population rank (×1.5 in informal-type areas) × canopy deficit, only where open grass, cropland or bare land exists on slopes under 30°. Wetlands are never plantable."),
    ("solar", "Response", "Rooftop solar potential", "ramp", ["0", "GWh/km²/yr", f"{solar_max:.0f}", "sequential"],
     "2.0 million Overture roofs, ERA5-Land irradiance 2015–2024, 21 % modules, 0.80 performance ratio, shading from neighbouring building heights."),
]

legend_ramps = {
    "diverging": "linear-gradient(90deg," + ",".join(V.BLUE[::-1][:6] + [V.NEUTRAL] + V.REDS[1:]) + ")",
    "sequential": "linear-gradient(90deg," + ",".join(V.ORANGES) + ")",
}
page = (Path(__file__).parent / "atlas_template.html").read_text()
css = (Path(__file__).parent / "vendor" / "leaflet-1.9.4.min.css").read_text()   # Leaflet's own CSS, BSD-2
data = dict(base=BASE, layers=layers, meta=[dict(id=i, group=gp, label=lb, kind=kd, legend=lg, caption=cp)
                                           for i, gp, lb, kd, lg, cp in LAYERS],
            ramps=legend_ramps, size=[H, W])
html = (page.replace("/*LEAFLET_CSS*/", css).replace("__DATA__", json.dumps(data))
        .replace("__CHARTS__", json.dumps(charts)).replace("__ANIMS__", json.dumps(anims)))
for k_, v in num.items():
    html = html.replace("{{" + k_ + "}}", v)
html = html.replace("{{y0}}", str(Y[0])).replace("{{y1}}", str(Y[1])).replace("{{y2}}", str(Y[2]))
(OUT / "kampala_heat_atlas.html").write_text(html)
print(f"wrote {OUT / 'kampala_heat_atlas.html'} ({len(html) / 1e6:.1f} MB)")
