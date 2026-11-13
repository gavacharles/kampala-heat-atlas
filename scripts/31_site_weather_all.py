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

import argparse  # noqa: E402
_ap = argparse.ArgumentParser()
_ap.add_argument("out", nargs="?", default="docs/site-weather")
_ap.add_argument("--group", help="build only this region (uganda, east-africa, gulf, west-africa, southern-africa)")
_ap.add_argument("--assemble", action="store_true", help="skip building; make the national cards, data files and landing page")
_A = _ap.parse_args()
OUT = Path(_A.out)
HERE = Path(__file__).parent
_SJ = json.loads((OUT / "sites.json").read_text())
SITES = _SJ["sites"]
GROUPS = _SJ.get("groups", {})
TH = json.loads((OUT / "thresholds.json").read_text())
W0, W1 = TH["working_hours"]
WORK = [h for h in range(W0, W1) if h not in TH.get("break_hours", [])]     # e.g. 8 working hours, lunch excluded
NW = len(WORK)
slug = lambda n: "-".join("".join(c if c.isalnum() else " " for c in n.lower()).split())

def one_site(s):
    """Forecast, note, record and meteograms (29), then the cards (30), for one site."""
    sd = OUT / slug(s["name"])
    args = [sys.executable, str(HERE / "29_site_weather.py"), str(OUT), "--name", s["name"], "--lat", str(s["lat"]),
            "--lon", str(s["lon"]), "--alt", str(s["alt"])]
    if not any((sd / "archive" / "observed").glob("*.csv")):
        args += ["--backfill", f"{pd.Timestamp.now():%Y}-01-01"]
    r = subprocess.run(args, capture_output=True, text=True)
    if r.returncode == 0:
        r = subprocess.run([sys.executable, str(HERE / "30_site_cards.py"), str(sd)], capture_output=True, text=True)
    return s["name"], r.returncode, r.stderr[-800:]


# sites are independent (each writes only its own folder), so build four at a time: mostly waiting on the network
from concurrent.futures import ThreadPoolExecutor  # noqa: E402
failed = []
todo = [] if _A.assemble else [s for s in SITES if not _A.group or s.get("group") == _A.group]
with ThreadPoolExecutor(max_workers=4) as pool:
    for name, code, err in pool.map(one_site, todo):
        print(name, "ok" if code == 0 else "FAILED\n" + err, flush=True)
        if code:
            failed.append(name)
if _A.group:                                                   # a regional job: list its folders for the artifact, then stop
    (OUT / f"_built_{_A.group}.txt").write_text("\n".join(slug(s["name"]) for s in todo) + "\n")
    for name in failed:
        print(f"::warning::site weather for {name} failed this run; its previous page stays up")
    sys.exit(1 if failed and len(failed) == len(todo) else 0)


# ---------------------------------------------------------------- national cards: one per activity, actual go hours
sys.path.insert(0, str(HERE))
from card_style import byline, BG, INK, INK2, MUT, SHARE_TINT, F, footer, header, spaced, text  # noqa: E402

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
    if s.get("country", "Uganda") != "Uganda":                   # the national cards are Uganda's
        continue
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
    header(d, "Site weather · Uganda", f"{name.split(' (')[0]}: this week", f"The hours {verb}, {W0:02d}-{W1:02d} h, lunch excluded",
           title_size=50 if len(name) < 22 else 44)
    x0, lab_w, y = 64, 210, 236
    cw = (W - 128 - lab_w) / max(len(days), 1)
    for j, dd in enumerate(days):
        text(d, (x0 + lab_w + (j + 0.5) * cw, y), f"{dd:%a}", F(24, 800), INK, "ma")
        text(d, (x0 + lab_w + (j + 0.5) * cw, y + 32), f"{dd:%d %b}", F(16, 600), MUT, "ma")
    y += 70
    region = None
    for s_, df in rows:
        if s_["region"] != region:
            region = s_["region"]
            spaced(d, (x0, y + 4), region, size=15); y += 30
        text(d, (x0, y + 40), s_["name"].replace("Central ", ""), F(25, 800), INK, "lm")
        for j, dd in enumerate(days):
            lab, n, nw = day_windows(df, dd, k)
            share = n / max(nw, 1)
            bg, fg = SHARE_TINT[0 if share >= 0.66 else 1 if share >= 0.33 else 2]
            cx = x0 + lab_w + j * cw
            d.rounded_rectangle((cx + 3, y + 4, cx + cw - 3, y + 80), 12, fill=bg)
            parts = lab.split(", ")
            parts = parts[:2] if len(parts) <= 2 else [parts[0], parts[1] + "+"]
            for li, ptxt in enumerate(parts):
                yy = y + (32 if len(parts) == 1 else 20 + li * 23)
                text(d, (cx + cw / 2, yy), ptxt, F(19 if len(ptxt) <= 6 else 17, 800), fg, "mm")
            g = df[df.index.normalize() == dd]
            text(d, (cx + cw / 2, y + 66), f"{g.precipitation.sum():.0f} mm", F(16, 700), INK2, "mm")
        y += 86
    y += 10
    lx = x0
    for s0, lab in ((0, "go most of the day"), (1, "part of the day"), (2, "little or none")):
        bg, fg = SHARE_TINT[s0]
        d.rounded_rectangle((lx, y, lx + 26, y + 26), 7, fill=bg); d.ellipse((lx + 9, y + 9, lx + 17, y + 17), fill=fg)
        text(d, (lx + 36, y + 13), lab, F(19, 600), INK2, "lm"); lx += 36 + d.textlength(lab, font=F(19, 600)) + 30
    text(d, (x0, y + 44), "Times are go windows: 08-13 = from 08:00 until the lunch break. Grey figure: daily rain.", F(19, 600), MUT)
    footer(d, W, H, SITE_URL)
    im.save(OUT / f"uganda_{k}_4x5.png")
    # 16:9 for X: today's go window per town, with the next two days
    from card_style import ACCENT, CARD, LINE, card, chips, marked_headline  # noqa: E402
    W2, H2 = 1600, 900
    im = Image.new("RGB", (W2, H2), BG); d = ImageDraw.Draw(im)
    spaced(d, (64, 64), f"Site weather · Uganda · {days[0]:%A %d %B}")
    yh = marked_headline(d, 64, 104, f"{name.split(' (')[0]}: go hours today across Uganda.", 540, size=44) + 20
    text(d, (64, yh), f"The hours {verb}, {W0:02d}-{W1:02d} h, lunch excluded.", F(20, 600), INK2)
    text(d, (64, yh + 32), "Green most of the day, amber part, red little or none.", F(20, 600), INK2)
    x0, rw, yy = 660, W2 - 660 - 64, 100
    rh = (H2 - 100 - 130 - 6 * 6) / max(len(rows), 1)
    for j, dd in enumerate(days[:3]):
        cx = x0 + 220 + (0 if j == 0 else 262 + (j - 1) * 162); cw_ = 250 if j == 0 else 150
        spaced(d, (cx + cw_ / 2 - (d.textlength(("Today" if j == 0 else f"{dd:%a %d}").upper(), font=F(14, 700)) * 1.15) / 2, 70),
               "Today" if j == 0 else f"{dd:%a %d}", size=14)
    for s_, df in rows:
        card(d, (x0, yy, x0 + rw, yy + rh), r=16)
        text(d, (x0 + 20, yy + rh / 2 - 11), s_["name"].replace("Central ", ""), F(21, 800), INK, "lm")
        spaced(d, (x0 + 20, yy + rh / 2 + 6), s_["region"], size=12)
        for j, dd in enumerate(days[:3]):
            lab, n, nw = day_windows(df, dd, k)
            share = n / max(nw, 1)
            bg, fg = SHARE_TINT[0 if share >= 0.66 else 1 if share >= 0.33 else 2]
            cw_ = 250 if j == 0 else 150
            cx = x0 + 220 + (0 if j == 0 else 262 + (j - 1) * 162)
            d.rounded_rectangle((cx, yy + 10, cx + cw_, yy + rh - 10), 12, fill=bg)
            lab2 = lab if len(lab) <= (22 if j == 0 else 12) else lab.split(", ")[0] + "+"
            text(d, (cx + cw_ / 2, yy + rh / 2), lab2, F(20 if j == 0 else 16, 800), fg, "mm")
            pass
        yy += rh + 6
    f2 = F(20, 800); lw = d.textlength(SITE_URL, font=f2) + 40
    d.rounded_rectangle((64, H2 - 96, 64 + lw, H2 - 56), 20, fill=ACCENT)
    text(d, (84, H2 - 76), SITE_URL, f2, INK, "lm")
    text(d, (64, H2 - 120), "Screening guidance from a forecast; your contract and H&S plan come first.", F(18, 600), MUT)
    byline(d, 0, H2 - 76, size=19, anchor_right=True, W=W2)
    im.save(OUT / f"uganda_{k}_x_16x9.png")
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
    towns.append(dict(name=s["name"], region=s["region"], country=s.get("country", "Uganda"), group=s.get("group", ""), lat=s["lat"], lon=s["lon"], slug=sl, issued=bj["issued"], day=b0["day"],
                      headline=b0["headline"], worst=worst, facts=b0["facts"],
                      rows=[dict(trade=r["trade"], status=r["status"], label=r["label"]) for r in b0["rows"]],
                      outlook=[dict(day=b["day"], headline=b["headline"]) for b in bj["days"][1:4]]))
(OUT / "towns_today.json").write_text(json.dumps(dict(issued=f"{pd.Timestamp.now(tz='Africa/Kampala'):%Y-%m-%d %H:%M}", towns=towns,
                                                     activities=[dict(key=k, name=n) for k, n, _ in ACTS]), indent=1))

# ---------------------------------------------------------------- landing page
cards = ""
allrows = [s for s in SITES if (OUT / slug(s["name"]) / "social" / "today_4x5.png").exists()]


def card_row(items):
    return "<div class='row'>" + "".join(
        f"<a class='site' href='{slug(x['name'])}/'><img src='{slug(x['name'])}/social/today_4x5.png' alt='Today on site, {html.escape(x['name'])}' loading='lazy'>"
        f"<span>{html.escape(x['name'])}</span></a>" for x in items) + "</div>"


for gk, gname in GROUPS.items():                               # region, then country (Uganda: by region of Uganda)
    gs = [x for x in allrows if x.get("group") == gk]
    if not gs:
        continue
    cards += f"<h2>{html.escape(gname)}</h2>"
    key = "region" if gk == "uganda" else "country"
    for sub_ in dict.fromkeys(x.get(key, "Uganda") for x in gs):
        cards += f"<p class='lede' style='margin:6px 0 0;font-weight:700'>{html.escape(sub_)}</p>"
        cards += card_row([x for x in gs if x.get(key, "Uganda") == sub_])
(OUT / "index.html").write_text(f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Site Weather</title>
<meta name="description" content="Daily construction site weather for East, West and Southern Africa and the Gulf: go windows for seven trades, the week ahead and a monthly record. By Gava Charles.">
<meta property="og:type" content="website"><meta property="og:site_name" content="Kampala Heat Atlas · Site weather">
<meta property="og:title" content="Site weather: today's go windows for construction">
<meta property="og:description" content="Concrete, asphalt, cranes, work at height, painting, earthworks and heat, hour by hour, for 34 towns and cities in Africa and the Gulf.">
<meta property="og:url" content="https://gavacharles.github.io/kampala-heat-atlas/site-weather/">
<meta property="og:image" content="https://gavacharles.github.io/kampala-heat-atlas/site-weather/uganda_concrete_x_16x9.png?v={pd.Timestamp.now(tz='Africa/Kampala'):%Y%m%d}">
<meta property="og:image:width" content="1600"><meta property="og:image:height" content="900">
<meta name="twitter:card" content="summary_large_image">
<meta name="twitter:image" content="https://gavacharles.github.io/kampala-heat-atlas/site-weather/uganda_concrete_x_16x9.png?v={pd.Timestamp.now(tz='Africa/Kampala'):%Y%m%d}">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wght@800&family=Inter:wght@400;500&display=swap">
<style>
:root{{--bg:#f6f6f3;--surface:#fff;--ink:#0b0b0b;--ink-2:#3a3a3e;--line:#e1e0d9;--blue:#2a78d6}}
@media (prefers-color-scheme: dark){{:root:not([data-theme="light"]){{--bg:#111312;--surface:#1a1c1b;--ink:#ecedea;--ink-2:#b6b9b2;--line:#2d302e;--blue:#6da7ec;color-scheme:dark}}}}
body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 Inter,Arial,sans-serif;padding:24px 16px}}
main{{max-width:1100px;margin:0 auto;display:grid;gap:16px}} h1{{font:800 34px/1.05 Archivo,Arial,sans-serif;margin:0}}
h2{{font:800 22px/1.1 Archivo,Arial,sans-serif;margin:12px 0 0}} .lede{{color:var(--ink-2);max-width:72ch;margin:0}} a{{color:var(--blue)}}
nav{{display:flex;gap:8px;flex-wrap:wrap}} nav a{{color:var(--ink);text-decoration:none;border:1px solid var(--line);background:var(--surface);padding:6px 12px;border-radius:999px;font-size:13px}}
.row{{display:flex;gap:12px;overflow-x:auto;padding-bottom:6px}} .site{{flex:none;width:260px;text-decoration:none;color:var(--ink);font-weight:500}}
.site img,.nat{{width:100%;height:auto;border:1px solid var(--line);border-radius:8px;background:#fff;display:block;margin-bottom:6px}} .nat{{max-width:520px}}
</style></head><body><main>
<nav><a href="../index.html">Atlas</a><a href="../3d.html">3D model</a><a href="../stories.html">Stories</a><a href="../daynight.html">Day vs night</a><a href="../meteogram.html">Meteogram</a><a href="./">Site weather</a><a href="map.html">Map</a><a href="share.html">Share today</a><a href="../gallery.html">Figures</a><a href="https://github.com/gavacharles/kampala-heat-atlas">Code</a></nav>
<h1>Site weather</h1>
<p class="lede">For contractors and site managers: hour-by-hour workable time for seven trades, a daily site note and a
month-by-month weather record against each town's 1991-2020 climate, for 34 towns and cities across East, West and Southern Africa and the Gulf. Refreshed every morning. Pick the town nearest
your site; to add a site, send its coordinates.</p>
<p><a href="market/"><strong>Site and market briefs</strong></a> (weather time and construction costs, per country, weekly) ·
<a href="map.html"><strong>Open the map</strong></a> (click any town, or anywhere in Uganda, for a live site check) ·
<a href="share.html"><strong>Share today</strong></a> (cards and captions, one tap to WhatsApp)</p>
<img class="nat" src="uganda_week_4x5.png" alt="Concrete pour windows this week across Uganda">
{cards}
<p class="lede" style="font-size:13px">Screening guidance from a weather forecast, not an instruction. The contract specification,
crane manufacturer's limits and site health-and-safety plan take precedence. Data: Open-Meteo (CC BY 4.0).</p>
<p style="margin:0;font-weight:700">By Gava Charles · Kampala Heat Atlas</p>
</main></body></html>""")
print("index and national card written;", "failed: " + ", ".join(failed) if failed else "all sites ok")
# one town's forecast failing (a timeout) must not fail the run: publish the others and warn; fail only if all failed
for name in failed:
    print(f"::warning::site weather for {name} failed this run; its previous page stays up")
sys.exit(1 if failed and len(failed) == len(todo) and todo else 0)
