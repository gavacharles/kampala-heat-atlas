"""Build the public GitHub Pages site in docs/.

    docs/index.html     2D atlas (scripts/14_atlas.py)
    docs/3d.html        deck.gl 3D viewer of the three model districts (sun shadows, breeze)
    docs/gallery.html   all figures and animations
    docs/data/*.json    district buildings, flow arrows and shade images for the 3D viewer
    docs/figures/*      copied from outputs/figures
"""
import base64
import io
import json
import shutil
import subprocess
import sys

import geopandas as gpd
import numpy as np
import pandas as pd
from PIL import Image
from pyproj import Transformer

from kampala import config
from kampala import model3d as M
from kampala import viz as V
from kampala.config import DATA, OUTPUTS, ROOT, FIGURES

DOCS = ROOT / "docs"; (DOCS / "data").mkdir(parents=True, exist_ok=True); (DOCS / "figures").mkdir(exist_ok=True)
M6, TAB, FIG = DATA / "processed" / "m6", OUTPUTS / "tables", FIGURES
crs = config.load()["study_area"]["crs"]
to_wgs = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
LABELS = {"cbd": "CBD", "informal": "Informal (Kawempe)", "frontier": "Growth frontier"}

PAGE_HEAD = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Kampala Heat Atlas · {title}</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wght@800&family=Inter:wght@400;500&display=swap">
<style>
:root{{--bg:#f6f6f3;--surface:#fff;--ink:#0b0b0b;--ink-2:#52514e;--line:#e1e0d9;--blue:#2a78d6}}
@media (prefers-color-scheme: dark){{:root{{--bg:#111312;--surface:#1a1c1b;--ink:#ecedea;--ink-2:#b6b9b2;--line:#2d302e;--blue:#6da7ec;color-scheme:dark}}}}
body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 Inter,Arial,sans-serif;padding:24px max(16px,env(safe-area-inset-left))}}
main{{max-width:1100px;margin:0 auto;display:grid;gap:20px}} h1{{font:800 34px/1.05 Archivo,Arial,sans-serif;margin:0}} h2{{font:800 24px/1.1 Archivo,Arial,sans-serif;margin:0}}
.lede{{color:var(--ink-2);max-width:70ch;margin:0}} nav{{display:flex;gap:8px;flex-wrap:wrap}}
nav a,.links a{{color:var(--ink);text-decoration:none;border:1px solid var(--line);background:var(--surface);padding:6px 12px;border-radius:999px;font-size:13px}}
section{{display:grid;gap:12px;border-top:1px solid var(--line);padding-top:16px}}
.media{{display:grid;grid-template-columns:minmax(0,540px) minmax(0,1fr);gap:16px;align-items:start}} video{{width:100%;border-radius:8px;background:#f6f6f3}}
.links{{display:flex;flex-direction:column;gap:8px;align-items:flex-start}} .cards{{display:flex;gap:8px;overflow-x:auto;padding-bottom:6px}} .cards img{{width:180px;border-radius:6px;border:1px solid var(--line)}}
.swipe{{position:relative;max-width:900px;user-select:none;touch-action:none;border-radius:8px;overflow:hidden;border:1px solid var(--line)}}
.swipe img{{display:block;width:100%;height:auto}} .swipe .top{{position:absolute;inset:0}} .swipe .bar{{position:absolute;top:0;bottom:0;width:3px;background:#fff;box-shadow:0 0 0 1px #0003;transform:translateX(-1px)}}
.tag{{position:absolute;top:10px;background:#fffd;color:#0b0b0b;font-size:12px;padding:3px 8px;border-radius:4px}} .tag.l{{left:10px}} .tag.r{{right:10px}}
input[type=range]{{max-width:900px;width:100%;accent-color:var(--blue)}}
@media (max-width:700px){{.media{{grid-template-columns:1fr}}}}
</style></head><body><main>
<nav><a href="index.html">Atlas</a><a href="3d.html">3D model</a><a href="stories.html">Stories</a><a href="daynight.html">Day vs night</a><a href="meteogram.html">Meteogram</a><a href="site-weather/">Site weather</a><a href="gallery.html">Figures</a><a href="https://github.com/gavacharles/kampala-heat-atlas">Code</a></nav>
"""

# ---------------------------------------------------------------- 2D atlas
subprocess.run([sys.executable, str(ROOT / "scripts" / "14_atlas.py"), str(DOCS)], check=True)
(DOCS / "kampala_heat_atlas.html").replace(DOCS / "index.html")

# ---------------------------------------------------------------- 3D viewer data
summary = pd.read_csv(TAB / "m6_district_summary.csv").set_index("district")
meta = json.loads((TAB / "m6_districts.json").read_text())
lookup = pd.read_csv(TAB / "m6_wall_sun_by_azimuth.csv", index_col=0)["kwh_m2_yr"].values
districts = {}
for name in ("cbd", "informal", "frontier"):
    npz = np.load(M6 / f"{name}.npz")
    x0, y0 = npz["origin"]
    b = gpd.read_file(M6 / f"buildings_{name}.gpkg")
    az, L, W = M.building_orientation(b)
    _, pen = M.wall_sun_penalty(az, L, W, lookup)
    b["pen"] = pen
    b = b[b.geometry.type == "Polygon"]
    b["geometry"] = b.geometry.simplify(0.3)
    bw = b.to_crs("EPSG:4326")
    blds = []
    for g, h, k, lo, pe in zip(bw.geometry, b["h"], b["roof_kwh_m2"], b["shading_loss_pct"], b["pen"]):
        ring = [[round(x, 6), round(y, 6)] for x, y in g.exterior.coords]
        blds.append({"p": ring, "h": round(float(h), 1),
                     "kwh": None if not np.isfinite(k) else round(float(k)),
                     "loss": None if not np.isfinite(lo) else round(float(lo), 1), "pen": round(float(pe), 1)})
    # flow arrows every 40 m (u, v are speed ratios; arrow length 18 m x speed)
    u, v, solid, spd = npz["u"], npz["v"], npz["solid"], npz["speed"]
    arrows = []
    for r in range(5, u.shape[0], 10):
        for c in range(5, u.shape[1], 10):
            if solid[r, c]:
                continue
            x, y = x0 + (c + 0.5) * 4, y0 - (r + 0.5) * 4
            x2, y2 = x + u[r, c] * 18, y + v[r, c] * 18
            (lx, ly), (lx2, ly2) = to_wgs.transform(x, y), to_wgs.transform(x2, y2)
            arrows.append([round(lx, 6), round(ly, 6), round(lx2, 6), round(ly2, 6), round(float(spd[r, c]), 2)])
    # ground shade hours image (0-6 h between 10:00 and 16:00)
    sh = npz["shade_hours"][::2, ::2]; ground = npz["bh"][::2, ::2] == 0
    cmap = V.seq(["#fbe7c6", "#86b6ef", "#1c5cab", "#0d366b"], "shade")
    rgba = cmap(np.clip(sh / 3.0, 0, 1)); rgba[~ground, 3] = 0
    buf = io.BytesIO(); Image.fromarray((rgba * 255).astype("uint8"), "RGBA").save(buf, "PNG", optimize=True)
    w, s = to_wgs.transform(x0, y0 - 2000); e, n = to_wgs.transform(x0 + 2000, y0)
    cx, cy = to_wgs.transform(x0 + 1000, y0 - 1000)
    out = {"buildings": blds, "arrows": arrows, "shade_png": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()}
    (DOCS / "data" / f"district_{name}.json").write_text(json.dumps(out, separators=(",", ":")))
    districts[name] = {"label": LABELS[name], "file": f"data/district_{name}.json", "centre": [cx, cy],
                       "bounds": [w, s, e, n], "ground": [[w, s], [e, s], [e, n], [w, n]],
                       "stats": {k: float(v) for k, v in summary.loc[name].items()}}
    print(name, len(blds), "buildings,", len(arrows), "arrows,",
          f"{(DOCS / 'data' / f'district_{name}.json').stat().st_size / 1e6:.1f} MB")
html = (ROOT / "scripts" / "site_3d_template.html").read_text().replace("__DISTRICTS__", json.dumps(districts))
(DOCS / "3d.html").write_text(html)

# ---------------------------------------------------------------- gallery
items = [
    ("fig7_walking_heat_maps.png", "Street trees take about 4 °C off the felt heat (UTCI) where they shade, 14:30"),
    ("fig7_trees_before_after_3d.png", "Informal district before and after 600 street trees, ground coloured by walking heat"),
    ("anim7_trees_shadows.gif", "An equinox day with the new street trees"),
    ("fig7_frontier_rotation.png", "The orientation trade-off: east–west turns cut wall sun but block the lake breeze"),
    ("anim7_solar_day.gif", "Solar over a day: CBD roofs and the city's output against average demand"),
    ("fig_m0_cube_cbd.png", "The Kampala cube: central Kampala terrain and buildings coloured by surface heat"),
    ("anim3d_cbd_shadows_m03.gif", "Equinox day in the CBD: shadows and the sun's path"),
    ("anim3d_informal_shadows_m03.gif", "Equinox day in the informal district (Kawempe): almost no shade at midday"),
    ("anim3d_informal_shadows_m06.gif", "21 June: the sun passes north of overhead"),
    ("anim3d_informal_shadows_m12.gif", "21 December: the sun passes south of overhead"),
    ("anim3d_frontier_shadows_m03.gif", "Equinox day at the growth frontier"),
    ("fig3d_cbd_sun_wind.png", "CBD in 3D: annual sunlight on every surface, sun paths, lake breeze"),
    ("fig3d_informal_sun_wind.png", "Informal district in 3D"),
    ("fig3d_frontier_sun_wind.png", "Growth frontier in 3D"),
    ("anim3d_cbd_orbit.gif", "Orbit of the CBD model"),
    ("anim3d_cbd_breeze.gif", "Lake breeze through the CBD street network (potential flow)"),
    ("anim3d_informal_breeze.gif", "Lake breeze through the informal district"),
    ("anim3d_frontier_breeze.gif", "Lake breeze through the growth frontier"),
    ("fig3d_orientation_rose.png", "Building orientation by settlement type, and sun on walls by facing direction"),
    ("fig3d_frontier_orientation_penalty.png", "Growth frontier: each building's orientation penalty"),
    ("fig3d_informal_orientation_penalty.png", "Informal district: each building's orientation penalty"),
    ("anim_m0_diurnal_wind.gif", "The daily wind cycle: the lake breeze switches on at 09:00"),
    ("anim_m0_cube_rotate.gif", "Rotating cube of central Kampala"),
    ("fig_m2_thermal_classes.png", "Thermal classes 2017, 2021, 2025"),
    ("anim_m2_thermal_classes.gif", "Thermal classes through time"),
    ("fig_m2_lst_change.png", "Change in relative surface heat 2017–2025"),
    ("fig_m2_lcz_dumbbell.png", "Surface heat by Local Climate Zone"),
    ("fig_m1_landcover_growth.png", "Land cover 2025 and where the city grew"),
    ("anim_m1_growth.gif", "Urban growth 2017–2025"),
    ("fig_m1_class_change.png", "Land-cover change by class"),
    ("fig_m3_informal_on_heat.png", "Packed low-rise morphology on 2025 surface heat"),
    ("fig_m3_sdi_georef_check.png", "Literature check: the 58 informal settlements mapped by SDI/NSDFU/ACTogether (2014, red), georeferenced from their map book, against the morphology layer (blue)"),
    ("fig_m3_trajectories.png", "Thermal trajectories by settlement type"),
    ("fig_m4_tree_priority.png", "Tree-planting priority"),
    ("fig_m4_priority_bars.png", "Residents and plantable land per priority class"),
    ("fig_m5_solar_density.png", "Rooftop solar potential"),
    ("fig_m5_scenarios.png", "Rooftop solar adoption scenarios"),
]
def publish(src, dst, max_mb=5.0, width=760):
    """Copy a figure; shrink GIFs over max_mb (fewer frames, narrower) so pages load quickly."""
    if src.suffix != ".gif" or src.stat().st_size <= max_mb * 1e6:
        shutil.copy(src, dst); return
    im = Image.open(src); frames, durs = [], []
    every = 2 if im.n_frames > 30 else 1
    for i in range(im.n_frames):
        im.seek(i)
        if i % every == 0:
            f = im.convert("RGB")
            frames.append(f.resize((width, round(f.height * width / f.width)), Image.LANCZOS).quantize(128))
            durs.append(im.info.get("duration", 80) * every)
    frames[0].save(dst, save_all=True, append_images=frames[1:], duration=durs, loop=0, optimize=True)


cards = []
for f, cap in items:
    if (FIG / f).exists():
        publish(FIG / f, DOCS / "figures" / f)
        cards.append(f'<figure><a href="figures/{f}"><img src="figures/{f}" alt="{cap}" loading="lazy"></a><figcaption>{cap}</figcaption></figure>')
gallery = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Kampala Heat Atlas · Gallery</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wght@800&family=Inter:wght@400;500&display=swap">
<style>
:root{{--bg:#f3f4f1;--surface:#fff;--ink:#141516;--ink-2:#4f524d;--line:#d9dbd4}}
@media (prefers-color-scheme: dark){{:root{{--bg:#111312;--surface:#1a1c1b;--ink:#ecedea;--ink-2:#b6b9b2;--line:#2d302e;color-scheme:dark}}}}
body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 Inter,Arial,sans-serif;padding:24px max(16px,env(safe-area-inset-left))}}
main{{max-width:1240px;margin:0 auto;display:grid;gap:20px}} h1{{font:800 34px/1.05 Archivo,Arial,sans-serif;margin:0}}
nav{{display:flex;gap:8px;flex-wrap:wrap}} nav a{{color:var(--ink);text-decoration:none;border:1px solid var(--line);background:var(--surface);padding:6px 12px;border-radius:999px;font-size:13px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(340px,1fr));gap:16px}}
figure{{margin:0;background:#fff;border:1px solid var(--line);border-radius:8px;padding:8px;display:grid;gap:6px;align-content:start}}
figure img{{width:100%;height:auto;display:block}} figcaption{{font-size:13px;color:#4f524d;padding:0 4px 4px}}
</style></head><body><main>
<nav><a href="index.html">Atlas</a><a href="3d.html">3D model</a><a href="stories.html">Stories</a><a href="daynight.html">Day vs night</a><a href="https://github.com/gavacharles/kampala-heat-atlas">Code</a></nav>
<h1>Figures &amp; animations</h1><p style="color:var(--ink-2);margin:0">Kampala Heat Atlas, first full run. Click any image for full size.</p>
<div class="grid">{''.join(cards)}</div></main></body></html>"""
(DOCS / "gallery.html").write_text(gallery)
# ---------------------------------------------------------------- day/night swipe
for nm in ("day", "night"):
    shutil.copy(FIGURES / f"modis_{nm}_2025.png", DOCS / "figures" / f"modis_{nm}_2025.png")
(DOCS / "daynight.html").write_text(PAGE_HEAD.format(title="Day vs night") + """
<h1>Day vs night</h1>
<p class="lede">Surface temperature over greater Kampala in the 2025 dry season, from the same sensor (MODIS Aqua, 1 km):
about 13:30 against about 01:30. Each map is shown relative to its own median, blue cooler and red warmer. Drag the handle.</p>
<div class="swipe" id="swipe" role="img" aria-label="Day and night surface temperature compared with a slider">
  <img src="figures/modis_night_2025.png" alt="Night-time surface temperature">
  <div class="top" id="top"><img src="figures/modis_day_2025.png" alt="Daytime surface temperature"></div>
  <div class="bar" id="bar"></div><span class="tag l">13:30 · day</span><span class="tag r">01:30 · night</span>
</div>
<input type="range" id="pos" min="0" max="100" value="50" aria-label="Swipe position">
<p class="lede">By day, the lake and wetlands are cool and the built-up core is hot. By night, the lake is the warmest surface and the dense
core stays warmer than the hills around it. The night heat is what keeps people in single-room houses from cooling down.</p>
<script>
const r = document.getElementById("pos"), top = document.getElementById("top"), bar = document.getElementById("bar");
const set = v => { top.style.clipPath = `inset(0 ${100 - v}% 0 0)`; bar.style.left = v + "%"; };
r.addEventListener("input", e => set(e.target.value)); set(50);
const sw = document.getElementById("swipe");
const drag = e => { const b = sw.getBoundingClientRect(); const x = ((e.touches ? e.touches[0].clientX : e.clientX) - b.left) / b.width * 100;
  r.value = Math.max(0, Math.min(100, x)); set(r.value); };
sw.addEventListener("pointermove", e => { if (e.buttons) drag(e); }); sw.addEventListener("pointerdown", drag);
</script></main></body></html>""")

# ---------------------------------------------------------------- stories
STORIES = [("heat", "Hotter by design", "Kampala's most tightly packed blocks are its hottest, and the ones that packed in fastest warmed most. A 3D model shows why."),
           ("trees", "Where the next 8,000 trees go", "Target planting to heat, people and open ground, and street trees take about 4 °C off the felt heat."),
           ("solar", "Roofs are not the constraint", "Kampala's roofs could make several times Uganda's electricity. The question is what the grid can take at noon.")]
blocks = []
for slug, title, intro in STORIES:
    src = FIGURES / f"social_{slug}"
    if not src.exists():
        continue
    dst = DOCS / "story" / slug; dst.mkdir(parents=True, exist_ok=True)
    for f in ("highlight.gif", "reel_square.mp4", "reel_vertical.mp4", f"film_{slug}.mp4"):
        shutil.copy(src / f, dst / f)
    cards = sorted((src / "carousel").glob("C*.png"))
    for c in cards:
        Image.open(c).convert("RGB").resize((540, 675), Image.LANCZOS).save(dst / c.name.replace(".png", ".jpg"), quality=85)
    shutil.copy(src / "carousel" / f"{slug}_carousel.pdf", dst / f"{slug}_carousel.pdf")
    thumbs = "".join(f'<img src="story/{slug}/{c.name.replace(".png", ".jpg")}" alt="Card {i + 1}" loading="lazy">' for i, c in enumerate(cards))
    blocks.append(f"""<section><h2>{title}</h2><p class="lede">{intro}</p>
<div class="media"><video src="story/{slug}/reel_square.mp4" controls muted loop playsinline poster="story/{slug}/{cards[1].name.replace('.png', '.jpg')}"></video>
<div class="links"><a href="story/{slug}/film_{slug}.mp4">Landscape film (1280×720)</a><a href="story/{slug}/reel_vertical.mp4">Vertical reel (1080×1920)</a>
<a href="story/{slug}/highlight.gif">Highlight GIF</a><a href="story/{slug}/{slug}_carousel.pdf">Carousel PDF</a></div></div>
<div class="cards">{thumbs}</div></section>""")
(DOCS / "stories.html").write_text(PAGE_HEAD.format(title="Stories") + """<h1>Stories</h1>
<p class="lede">Three short stories, one for each open research gap. Every dot is real data: a 90 m block, about 1,000 residents, or about 1,000 roofs.
Free to share with credit to Gava Charles.</p>""" + "".join(blocks) + "</main></body></html>")

# ---------------------------------------------------------------- meteogram
for f in ("meteogram_latest.png", "meteogram_latest.csv"):
    if (FIGURES / f).exists():
        shutil.copy(FIGURES / f, DOCS / "figures" / f)
(DOCS / "meteogram.html").write_text(PAGE_HEAD.format(title="Meteogram") + """<h1>Heat &amp; solar meteogram</h1>
<p class="lede">The next seven days in central Kampala, translated through the 3D model: felt heat (UTCI) for someone in an open street
and under a street tree, cloud and rain, the lake breeze, and rooftop solar output. Refreshed every morning at 06:00 from the
Open-Meteo forecast. Under a tree the felt heat is a few degrees lower by day and slightly higher at night, because the canopy
holds in the warmth that would otherwise radiate to the sky.</p>
<img src="figures/meteogram_latest.png" alt="Seven-day meteogram for Kampala: felt heat, cloud and rain, wind, rooftop solar" style="width:100%;height:auto;border:1px solid var(--line);border-radius:8px;background:#fff">
<p class="lede"><a href="figures/meteogram_latest.csv">Download the hourly numbers (CSV)</a></p></main></body></html>""")

(DOCS / ".nojekyll").write_text("")
print("site built:", sorted(p.name for p in DOCS.iterdir()))
