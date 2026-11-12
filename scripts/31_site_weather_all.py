"""Run the site weather service for every site in docs/site-weather/sites.json, then build the Uganda
landing page and a national card.

    python scripts/31_site_weather_all.py [docs/site-weather]

New sites are backfilled from ERA5 from 1 January of the current year on their first run, so their
monthly record starts with history. Writes <out>/index.html and <out>/uganda_week_4x5.png.
"""
import html
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "docs/site-weather")
HERE = Path(__file__).parent
SITES = json.loads((OUT / "sites.json").read_text())["sites"]
TH = json.loads((OUT / "thresholds.json").read_text())
W0, W1 = TH["working_hours"]
WORK = [h for h in range(W0, W1) if h not in TH.get("break_hours", [])]     # e.g. 8 working hours, lunch excluded
NW = len(WORK)
slug = lambda n: "-".join("".join(c if c.isalnum() else " " for c in n.lower()).split())

failed = []
for s in SITES:
    sd = OUT / slug(s["name"])
    args = [sys.executable, str(HERE / "29_site_weather.py"), str(OUT), "--name", s["name"], "--lat", str(s["lat"]),
            "--lon", str(s["lon"]), "--alt", str(s["alt"])]
    if not any((sd / "archive" / "observed").glob("*.csv")):
        args += ["--backfill", f"{pd.Timestamp.now():%Y}-01-01"]
    r = subprocess.run(args, capture_output=True, text=True)
    if r.returncode == 0:
        r = subprocess.run([sys.executable, str(HERE / "30_site_cards.py"), str(sd)], capture_output=True, text=True)
    print(s["name"], "ok" if r.returncode == 0 else "FAILED\n" + r.stderr[-800:], flush=True)
    if r.returncode:
        failed.append(s["name"])


# ---------------------------------------------------------------- national cards: one per activity, actual go hours
def F(sz, b=False):
    for c in ("/System/Library/Fonts/Supplemental/Arial Bold.ttf" if b else "/System/Library/Fonts/Supplemental/Arial.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if b else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(c, sz)
        except OSError:
            pass
    return ImageFont.load_default(sz)


BG, INK, INK2, MUT, LINE, BLUE = (246, 246, 243), (11, 11, 11), (82, 81, 78), (137, 135, 129), (225, 224, 217), (42, 120, 214)
GO, CAU, STOP = (47, 143, 58), (224, 165, 38), (194, 52, 52)
SITE_URL = "gavacharles.github.io/kampala-heat-atlas/site-weather"
ACTS = [("concrete", "Concrete pour", "a pour can go ahead"), ("asphalt", "Asphalt & surfacing", "laying or spraying can go ahead"),
        ("crane", "Crane lifts", "lifts can go ahead"), ("height", "Work at height", "roof and scaffold work can go ahead"),
        ("coatings", "Painting & coatings", "painting can go ahead"), ("earthworks", "Earthworks", "earthworks can go ahead"),
        ("labour", "Outdoor labour (heat)", "work can go on without heat work/rest cycles")]


def windows(st, hours):
    """Go windows from hourly status (0 = go) over working hours, split at gaps (lunch): ['08-13', '14-16']."""
    out, s0, prev = [], None, None
    for v, hh in zip(list(st) + [1], list(hours) + [None]):
        if s0 is not None and (v != 0 or hh != prev + 1):
            out.append(f"{s0:02d}-{prev + 1:02d}"); s0 = None
        if v == 0 and hh is not None and s0 is None:
            s0 = hh
        prev = hh
    return out


def day_windows(df, dd, k):
    w = df[(df.index.normalize() == dd) & df.index.hour.isin(WORK)]
    st = w[f"st_{k}"].values
    if len(w) and (st == 0).all():
        return "all day", len(w), len(w)
    win = windows(st, w.index.hour)
    return (", ".join(win) if win else "none"), int((st == 0).sum()), len(w)


def x_len(txt):                                              # X counts any link as 23 characters
    return len(txt.replace(SITE_URL, "x" * 23))


rows = []
for s in SITES:
    p = OUT / slug(s["name"]) / "site_forecast_latest.csv"
    if not p.exists():
        continue
    df = pd.read_csv(p, index_col=0); df.index = pd.to_datetime(df.index.str.slice(0, 19))
    rows.append((s, df))
now = pd.Timestamp.now(tz="Africa/Kampala").tz_localize(None)
days = sorted({d for _, df in rows for d in df.index.normalize().unique()})
days = [d for d in days if d > now.normalize() or now.hour < W1][:6]
for k, name, verb in ACTS:
    W, H = 1080, 1350
    im = Image.new("RGB", (W, H), BG); d = ImageDraw.Draw(im)
    d.text((64, 64), "SITE WEATHER · UGANDA", font=F(26, True), fill=BLUE)
    d.text((64, 104), f"{name.split(' (')[0]}: this week", font=F(54, True), fill=INK)
    d.text((64, 174), f"The hours {verb}, {W0:02d}-{W1:02d} h with lunch excluded, and daily rain", font=F(24), fill=INK2)
    x0, lab_w, y = 64, 230, 240
    cw = (W - 128 - lab_w) / max(len(days), 1)
    for j, dd in enumerate(days):
        d.text((x0 + lab_w + (j + 0.5) * cw, y), f"{dd:%a}", font=F(26, True), fill=INK, anchor="ma")
        d.text((x0 + lab_w + (j + 0.5) * cw, y + 34), f"{dd:%d %b}", font=F(18), fill=MUT, anchor="ma")
    y += 76
    region = None
    for s, df in rows:
        if s["region"] != region:
            region = s["region"]
            d.text((x0, y + 4), region.upper(), font=F(18, True), fill=BLUE); y += 28
        d.text((x0, y + 42), s["name"].replace("Central ", ""), font=F(28, True), fill=INK, anchor="lm")
        for j, dd in enumerate(days):
            lab, n, nw = day_windows(df, dd, k)
            share = n / max(nw, 1)
            col = GO if share >= 0.66 else CAU if share >= 0.33 else STOP
            cx = x0 + lab_w + j * cw
            d.rounded_rectangle((cx + 4, y + 4, cx + cw - 4, y + 82), 10, fill=col)
            parts = lab.split(", ")
            parts = parts[:2] if len(parts) <= 2 else [parts[0], parts[1] + "+"]
            for li, ptxt in enumerate(parts):
                yy = y + (30 if len(parts) == 1 else 20 + li * 24)
                d.text((cx + cw / 2, yy), ptxt, font=F(22 if len(ptxt) <= 6 else 19, True), fill=(255, 255, 255), anchor="mm")
            g = df[df.index.normalize() == dd]
            d.text((cx + cw / 2, y + 68), f"{g.precipitation.sum():.0f} mm", font=F(15, True), fill=(255, 255, 255), anchor="mm")
        y += 88
    y += 12
    lx = x0
    for col, lab in ((GO, "go most of the day"), (CAU, "part of the day"), (STOP, "little or none")):
        d.rounded_rectangle((lx, y, lx + 26, y + 26), 5, fill=col)
        d.text((lx + 36, y + 13), lab, font=F(20), fill=INK2, anchor="lm"); lx += 36 + d.textlength(lab, font=F(20)) + 36
    d.text((x0, y + 44), "Times are the go windows: 08-13 = from 08:00 until the lunch break.", font=F(19), fill=MUT)
    d.line((64, H - 140, W - 64, H - 140), fill=LINE, width=2)
    d.text((64, H - 118), "Screening guidance from a forecast. Your contract and site H&S plan come first.", font=F(20), fill=MUT)
    d.text((64, H - 86), SITE_URL, font=F(22, True), fill=INK2)
    d.text((64, H - 54), "Gava Charles · Kampala Heat Atlas · forecast data Open-Meteo", font=F(20), fill=MUT)
    im.save(OUT / f"uganda_{k}_4x5.png")
    # captions: long (WhatsApp, LinkedIn, Facebook) and X (280 characters, links count as 23)
    d0 = days[0]
    today = [(s["name"].replace("Central ", ""), day_windows(df, d0, k)[0]) for s, df in rows]
    short = name.split(" (")[0]
    long_ = ([f"Site weather, Uganda: {d0:%A %d %B}. {short}, go hours today:", ""] + [f"• {t}: {w_}" for t, w_ in today] +
             ["", f"Hour-by-hour for seven trades and your town's week ahead: {SITE_URL}", "Send us your site location to add it.", "",
              "Screening guidance from a forecast; your contract, crane limits and H&S plan come first.",
              "#Uganda #Construction #SiteSafety #Weather"])
    (OUT / f"uganda_{k}_caption.txt").write_text("\n".join(long_) + "\n")
    tags = " #Uganda #Construction"
    for m in range(len(today), 0, -1):
        xt = f"{short} go hours today ({d0:%a %d %b}): " + "; ".join(f"{t} {w_}" for t, w_ in today[:m]) + \
             ("…" if m < len(today) else "") + f"\nAll towns, all trades: {SITE_URL}" + tags
        if x_len(xt) <= 280:
            break
    (OUT / f"uganda_{k}_caption_x.txt").write_text(xt + "\n")
for src, dst in (("uganda_concrete_4x5.png", "uganda_week_4x5.png"), ("uganda_concrete_caption.txt", "uganda_caption.txt")):
    (OUT / dst).write_bytes((OUT / src).read_bytes())            # older links keep working

# ---------------------------------------------------------------- data for map.html and share.html
RANK = {"STOP": 0, "LIMITED": 1, "CAUTION": 2, "GO": 3}
towns = []
for s in SITES:
    sl = slug(s["name"]); bp = OUT / sl / "site_briefing_latest.json"
    if not bp.exists():
        continue
    bj = json.loads(bp.read_text()); b0 = bj["days"][0]
    worst = min((r["status"] for r in b0["rows"]), key=RANK.get)
    towns.append(dict(name=s["name"], region=s["region"], lat=s["lat"], lon=s["lon"], slug=sl, issued=bj["issued"], day=b0["day"],
                      headline=b0["headline"], worst=worst, facts=b0["facts"],
                      rows=[dict(trade=r["trade"], status=r["status"], label=r["label"]) for r in b0["rows"]],
                      outlook=[dict(day=b["day"], headline=b["headline"]) for b in bj["days"][1:4]]))
(OUT / "towns_today.json").write_text(json.dumps(dict(issued=f"{pd.Timestamp.now(tz='Africa/Kampala'):%Y-%m-%d %H:%M}", towns=towns,
                                                     activities=[dict(key=k, name=n) for k, n, _ in ACTS]), indent=1))

# ---------------------------------------------------------------- landing page
cards = ""
for reg in dict.fromkeys(s["region"] for s, _ in rows):
    cards += f"<h2>{reg}</h2><div class='row'>"
    for s, df in rows:
        if s["region"] == reg:
            sl = slug(s["name"])
            cards += (f"<a class='site' href='{sl}/'><img src='{sl}/social/today_4x5.png' alt='Today on site, {html.escape(s['name'])}'>"
                      f"<span>{html.escape(s['name'])}</span></a>")
    cards += "</div>"
(OUT / "index.html").write_text(f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Site Weather Uganda</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wght@800&family=Inter:wght@400;500&display=swap">
<style>
:root{{--bg:#f6f6f3;--surface:#fff;--ink:#0b0b0b;--ink-2:#52514e;--line:#e1e0d9;--blue:#2a78d6}}
@media (prefers-color-scheme: dark){{:root:not([data-theme="light"]){{--bg:#111312;--surface:#1a1c1b;--ink:#ecedea;--ink-2:#b6b9b2;--line:#2d302e;--blue:#6da7ec;color-scheme:dark}}}}
body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 Inter,Arial,sans-serif;padding:24px 16px}}
main{{max-width:1100px;margin:0 auto;display:grid;gap:16px}} h1{{font:800 34px/1.05 Archivo,Arial,sans-serif;margin:0}}
h2{{font:800 22px/1.1 Archivo,Arial,sans-serif;margin:12px 0 0}} .lede{{color:var(--ink-2);max-width:72ch;margin:0}} a{{color:var(--blue)}}
nav{{display:flex;gap:8px;flex-wrap:wrap}} nav a{{color:var(--ink);text-decoration:none;border:1px solid var(--line);background:var(--surface);padding:6px 12px;border-radius:999px;font-size:13px}}
.row{{display:flex;gap:12px;overflow-x:auto;padding-bottom:6px}} .site{{flex:none;width:260px;text-decoration:none;color:var(--ink);font-weight:500}}
.site img,.nat{{width:100%;height:auto;border:1px solid var(--line);border-radius:8px;background:#fff;display:block;margin-bottom:6px}} .nat{{max-width:520px}}
</style></head><body><main>
<nav><a href="../index.html">Atlas</a><a href="../3d.html">3D model</a><a href="../stories.html">Stories</a><a href="../daynight.html">Day vs night</a><a href="../meteogram.html">Meteogram</a><a href="./">Site weather</a><a href="map.html">Map</a><a href="share.html">Share today</a><a href="../gallery.html">Figures</a><a href="https://github.com/gavacharles/kampala-heat-atlas">Code</a></nav>
<h1>Site weather · Uganda</h1>
<p class="lede">For contractors and site managers: hour-by-hour workable time for seven trades, a daily site note and a
month-by-month weather record against each town's 1991-2020 climate. Refreshed every morning at 06:00. Pick the town nearest
your site; to add a site, send its coordinates.</p>
<p><a href="map.html"><strong>Open the map</strong></a> (click any town, or anywhere in Uganda, for a live site check) ·
<a href="share.html"><strong>Share today</strong></a> (cards and captions, one tap to WhatsApp)</p>
<img class="nat" src="uganda_week_4x5.png" alt="Concrete pour windows this week across Uganda">
{cards}
<p class="lede" style="font-size:13px">Screening guidance from a weather forecast, not an instruction. The contract specification,
crane manufacturer's limits and site health-and-safety plan take precedence. Data: Open-Meteo (CC BY 4.0).</p>
</main></body></html>""")
print("index and national card written;", "failed: " + ", ".join(failed) if failed else "all sites ok")
sys.exit(1 if failed else 0)
