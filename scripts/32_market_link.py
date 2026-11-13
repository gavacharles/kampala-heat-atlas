"""Market intelligence: site weather (time) joined with CiO construction costs (money), per country.

    python scripts/32_market_link.py [docs/site-weather]

Run in the daily assemble step after scripts/31_site_weather_all.py.
1. Weather, out: <out>/market/index.json, one entry per country: for each trade, days with a go window and go
   hours per working day over the next 7 days, and hours lost to weather in the last 7 days, averaged over the
   country's towns. CiO reads this weekly (cio-construction-costs/src/ops/market_link.py) for its country pages.
2. Costs, in: CiO's cost signal from its public repository (expected change in construction input costs with its
   range, oil and exchange-rate drivers, materials moving most), saved as <out>/market/costs.json.
3. The weekly site and market brief for every country with both: brief_<country>_4x5.png (card),
   brief_<country>_x_16x9.png (for X), brief_<country>.md (note), and <out>/market/index.html.
Advice lines are rules on these numbers (see what_to_do); screening guidance, not instructions.
"""
import html
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent))
from card_style import (ACCENT, BG, CARD, EYEBROW, INK, INK2, LINE, MUT, TINTED, F, byline, card, chips,  # noqa: E402
                        marked_headline, spaced, text, wrap)

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "docs/site-weather")
MK = OUT / "market"; MK.mkdir(exist_ok=True)
BASE = "https://gavacharles.github.io/kampala-heat-atlas/site-weather"
CIO_SIGNAL = "https://raw.githubusercontent.com/gavacharles/cio-construction-costs/main/public/data/market_link.json"
CIO_REPO = "github.com/gavacharles/cio-construction-costs"
SITES = json.loads((OUT / "sites.json").read_text())["sites"]
TRADES = [("concrete", "Concrete pour"), ("asphalt", "Asphalt & surfacing"), ("crane", "Crane lifts"), ("height", "Work at height"),
          ("coatings", "Painting & coatings"), ("earthworks", "Earthworks"), ("labour", "Outdoor labour (heat)")]
slug = lambda n: "-".join("".join(c if c.isalnum() else " " for c in n.lower()).split())          # noqa: E731
ckey = lambda c: "_".join("".join(ch if ch.isalnum() else " " for ch in c.lower().replace("ô", "o")).split())  # noqa: E731


def site_numbers(s):
    sd = OUT / slug(s["name"])
    p = sd / "site_forecast_latest.csv"
    if not p.exists():
        return None
    th = json.loads(((sd / "thresholds_site.json") if (sd / "thresholds_site.json").exists() else (OUT / "thresholds.json")).read_text())
    w0, w1 = th["working_hours"]; work = [h for h in range(w0, w1) if h not in th.get("break_hours", [])]
    df = pd.read_csv(p, index_col=0); df.index = pd.to_datetime(df.index.str.slice(0, 19))
    days = sorted(df.index.normalize().unique())[:7]
    ahead = {}
    for k, _ in TRADES:
        go_h, go_d = [], 0
        for d in days:
            w = df[(df.index.normalize() == d) & df.index.hour.isin(work)]
            n = int((w[f"st_{k}"] == 0).sum()); go_h.append(n); go_d += n >= 2
        ahead[k] = dict(days=go_d, hours=float(np.mean(go_h)))
    lost = {k: 0.0 for k, _ in TRADES}; rain7 = 0.0
    obs = sorted((sd / "archive" / "observed").glob("*.csv"))
    if obs:
        o = pd.concat([pd.read_csv(f, dtype={"date": str}) for f in obs[-2:]]).drop_duplicates("date").sort_values("date").tail(7)
        for k, _ in TRADES:
            if f"go_hours_{k}" in o:
                lost[k] = float((o.working_hours - o[f"go_hours_{k}"]).sum())
        rain7 = float(o.rain_mm.sum())
    month = None
    mi = sd / "archive" / "monthly_index.csv"
    if mi.exists():
        m = pd.read_csv(mi).tail(1).iloc[0]
        month = dict(month=m["month"], rain_mm=float(m["rain_mm"]), pct=None if pd.isna(m["rain_pct_of_normal_pro_rata"]) else float(m["rain_pct_of_normal_pro_rata"]),
                     heavy=int(m["heavy_rain_days"]), exceptional=int(m["exceptional_rain_days"]))
    return dict(name=s["name"].replace("Central ", ""), url=f"{BASE}/{slug(s['name'])}/", hours_per_day=len(work), ahead=ahead, lost=lost,
                rain7=rain7, month=month, days=len(days))


# ---------------------------------------------------------------- 1. weather per country
countries = {}
for s in SITES:
    n = site_numbers(s)
    if n:
        countries.setdefault(s.get("country", "Uganda"), []).append(n)
now = pd.Timestamp.now(tz="Africa/Kampala")
W = {}
for c, ns in countries.items():
    k = ckey(c)
    trades = [dict(key=t, trade=lab, go_days_ahead=int(round(np.mean([x["ahead"][t]["days"] for x in ns]))),
                   go_hours_per_day_ahead=round(float(np.mean([x["ahead"][t]["hours"] for x in ns])), 1),
                   lost_hours_last_7=round(float(np.mean([x["lost"][t] for x in ns])), 1)) for t, lab in TRADES]
    conc = trades[0]
    pcts = [x["month"]["pct"] for x in ns if x["month"] and x["month"]["pct"] is not None]
    W[k] = dict(country=c, generated=f"{now:%Y-%m-%d}", days_ahead=ns[0]["days"], hours_per_day=int(round(np.mean([x["hours_per_day"] for x in ns]))),
                trades=trades, towns=[dict(name=x["name"], url=x["url"]) for x in ns],
                rain_last_7_mm=round(float(np.mean([x["rain7"] for x in ns])), 1),
                rain_month_pct_of_normal=round(float(np.mean(pcts))) if pcts else None,
                headline=(f"Next {ns[0]['days']} days in {c}: concrete can go on {conc['go_days_ahead']} days "
                          f"({conc['go_hours_per_day_ahead']:.1f} of {int(round(np.mean([x['hours_per_day'] for x in ns])))} working hours a day on average); "
                          f"weather took {conc['lost_hours_last_7']:.0f} concrete hours in the last 7 days."),
                page=f"{BASE}/", brief=f"{BASE}/market/#{k}")
(MK / "index.json").write_text(json.dumps(dict(source="Site weather service (gavacharles/kampala-heat-atlas)", generated=f"{now:%Y-%m-%d %H:%M} EAT",
                                               countries=W), indent=1, ensure_ascii=False))

# ---------------------------------------------------------------- 2. costs from CiO
costs = {}
try:
    sig = json.loads(urllib.request.urlopen(CIO_SIGNAL, timeout=60).read())
    costs = sig.get("countries", {})
    (MK / "costs.json").write_text(json.dumps(sig, indent=1, ensure_ascii=False))
except Exception as e:  # noqa: BLE001
    if (MK / "costs.json").exists():                       # keep last week's signal if CiO's file is not reachable
        sig = json.loads((MK / "costs.json").read_text()); costs = sig.get("countries", {})
    print("CiO cost signal not reachable:", str(e)[:100])


# ---------------------------------------------------------------- 3. the brief
def what_to_do(w, c):
    t = {x["key"]: x for x in w["trades"]}; out = []
    cd = t["concrete"]["go_days_ahead"]
    if cd <= 2:
        out.append(f"Few pour windows ({cd} of {w['days_ahead']} days): bring pours to the clearest days on the town cards; have covers and curing ready.")
    elif t["concrete"]["go_hours_per_day_ahead"] < 3.5:
        out.append(f"Pour windows are short ({t['concrete']['go_hours_per_day_ahead']:.1f} of {w['hours_per_day']} working hours a day): "
                   "plan smaller pours or extra crews for the clear hours on the town cards.")
    elif cd >= 5:
        out.append(f"Good week for pours and surfacing ({cd} of {w['days_ahead']} days, {t['concrete']['go_hours_per_day_ahead']:.1f} h a day): "
                   "use it to recover programme.")
    for fl in (c or {}).get("flags", []):                     # CiO's shock watch: observed prices off the forecast
        if fl.startswith("Observed"):
            out.append(fl.split(" (")[0] + ": costs may already be higher than the forecast shows.")
    if c and abs(c.get("oil_4w_pct") or 0) >= 10:
        out.append(f"Oil has moved {c['oil_4w_pct']:+.0f}% in 4 weeks: cost forecasts set before the move may be "
                   f"{'too high' if c['oil_4w_pct'] < 0 else 'too low'}; recheck fuel-heavy items before committing.")
    if c:
        m = c["near"]["median"]
        if m >= 3:
            mv = ", ".join(x["name"].lower() for x in c.get("movers", [])[:2])
            out.append(f"Input costs expected up {m:.1f}% by {c['near']['period']}: fix prices or buy ahead" + (f" ({mv} moving most)." if mv else "."))
        elif m <= 0.5:
            out.append(f"Input costs about flat ({m:+.1f}% by {c['near']['period']}): no need to buy ahead.")
        if (c.get("fx_13w_pct") or 0) >= 5:
            out.append(f"The {c['currency']} has weakened {c['fx_13w_pct']:.1f}% in 13 weeks: imported steel, fuel and fittings follow; "
                       "check that price adjustment (FIDIC 13.7/13.8) is in the contract.")
        if (c.get("oil_13w_pct") or 0) >= 10:
            out.append(f"Oil up {c['oil_13w_pct']:.0f}% in 13 weeks: diesel, asphalt and haulage costs follow within weeks.")
    if t["concrete"]["lost_hours_last_7"] >= 8:
        out.append(f"Weather took {t['concrete']['lost_hours_last_7']:.0f} concrete hours last week: record it in the site diary for time "
                   "claims (FIDIC 8.5); the monthly weather record is on each town page.")
    return out[:5] or ["No weather or cost signal needs action this week; keep to programme."]


def brief_card(k, w, c, W_, H_, path):
    im = Image.new("RGB", (W_, H_), BG); d = ImageDraw.Draw(im)
    wide = W_ > H_
    spaced(d, (64, 60), f"Site and market · {w['country']} · week of {now:%d %b %Y}")
    cd = w["trades"][0]["go_days_ahead"]
    head = (f"Concrete can go on {cd} of {w['days_ahead']} days; input costs "
            + (f"{c['near']['median']:+.1f}% by {c['near']['period']}." if c else "not tracked here yet."))
    colw = (W_ - 128 - 20) // 2 if not wide else 450
    y = marked_headline(d, 64, 100, head, W_ - 128, size=40 if not wide else 38) + 20
    # TIME card
    tx, ty = 64, y
    tw = colw
    th_ = 430 if not wide else (H_ - 150 - y)
    if wide:
        tx, ty = 64, y
    card(d, (tx, ty, tx + tw, ty + th_))
    spaced(d, (tx + 22, ty + 22), "Time · site weather", size=15)
    yy = ty + 60
    for t in w["trades"][:6]:
        share = t["go_hours_per_day_ahead"] / max(w["hours_per_day"], 1)          # tint by go hours, not just days
        bg, fg = TINTED["GO"] if share >= 0.66 else TINTED["LIMITED"] if share >= 0.33 else TINTED["STOP"]
        text(d, (tx + 22, yy + 16), t["trade"], F(19, 700), INK, "lm")
        lab = f"{t['go_days_ahead']} days · {t['go_hours_per_day_ahead']:.1f} h/day"
        lw = d.textlength(lab, font=F(17, 800)) + 24
        d.rounded_rectangle((tx + tw - 22 - lw, yy + 2, tx + tw - 22, yy + 30), 14, fill=bg)
        text(d, (tx + tw - 22 - lw / 2, yy + 16), lab, F(17, 800), fg, "mm")
        yy += 42
    yy += 8
    for ln in wrap(d, f"Last 7 days: {w['trades'][0]['lost_hours_last_7']:.0f} concrete hours lost; rain {w['rain_last_7_mm']:.0f} mm"
                   + (f"; month to date {w['rain_month_pct_of_normal']:.0f}% of normal rain." if w["rain_month_pct_of_normal"] is not None else "."),
                   F(17, 600), tw - 44)[:3]:
        text(d, (tx + 22, yy), ln, F(17, 600), INK2); yy += 24
    # COST card
    cx, cy = (tx + tw + 20, ty) if True else (64, ty + th_ + 20)
    card(d, (cx, cy, cx + tw, cy + th_))
    spaced(d, (cx + 22, cy + 22), "Money · CiO construction costs", size=15)
    if c:
        text(d, (cx + 22, cy + 62), f"{c['near']['median']:+.1f}%", F(64, 800), INK)
        text(d, (cx + 22, cy + 140), f"{c['series']} by {c['near']['period']}", F(17, 700), INK2)
        text(d, (cx + 22, cy + 166), f"range {c['near']['lo']:+.1f}% to {c['near']['hi']:+.1f}% · base {c['base']}", F(16, 600), MUT)
        yy = cy + 206
        if c.get("next"):
            text(d, (cx + 22, yy), f"then {c['next']['median']:+.1f}% by {c['next']['period']}", F(18, 700), INK); yy += 34
        if c.get("oil_13w_pct") is not None:
            o4 = f" · 4 weeks {c['oil_4w_pct']:+.1f}%" if c.get("oil_4w_pct") is not None else ""
            text(d, (cx + 22, yy), f"Oil, 13 weeks: {c['oil_13w_pct']:+.1f}%{o4}", F(17, 600), INK2); yy += 28
        if c.get("fx_13w_pct") is not None:
            txt = (f"{c['currency']} {'weaker' if c['fx_13w_pct'] > 0 else 'stronger'} by {abs(c['fx_13w_pct']):.1f}% (13 weeks)")
            text(d, (cx + 22, yy), txt, F(17, 600), INK2); yy += 28
        if c.get("movers"):
            yy += 6
            spaced(d, (cx + 22, yy), "Moving most, 12 months", size=13); yy += 26
            for m in c["movers"][:3]:
                text(d, (cx + 22, yy), f"{m['name']}  {m['change_year']:+.1f}%", F(17, 700), INK); yy += 26
    else:
        for i, ln in enumerate(wrap(d, "CiO does not track this country yet. Weather time only.", F(18, 600), tw - 44)):
            text(d, (cx + 22, cy + 70 + i * 26), ln, F(18, 600), INK2)
    # what to do: under the two cards (square), or as a third column (X)
    ax, y, aw = (64, ty + th_ + 22, W_ - 128) if not wide else (cx + tw + 30, ty, W_ - 64 - (cx + tw + 30))
    fs = 20 if not wide else 18
    spaced(d, (ax, y), "What to do this week", size=16 if not wide else 15); y += 34
    for a in what_to_do(w, c):
        for i, ln in enumerate(wrap(d, a, F(fs, 600), aw - 26)):
            if y > H_ - 160:
                break
            if i == 0:
                d.ellipse((ax, y + 7, ax + 10, y + 17), fill=ACCENT)
            text(d, (ax + 26, y), ln, F(fs, 600), INK); y += fs + 8
        y += 6
    # footer
    d.line((64, H_ - 128, W_ - 64, H_ - 128), fill=LINE, width=2)
    f = F(19, 800); link = BASE.replace("https://", "") + "/market"
    lw = d.textlength(link, font=f) + 36
    d.rounded_rectangle((64, H_ - 92, 64 + lw, H_ - 54), 19, fill=ACCENT); text(d, (82, H_ - 73), link, f, INK, "lm")
    text(d, (64, H_ - 112), f"Weather: Open-Meteo, ECMWF · Costs: CiO Construction Cost Platform ({CIO_REPO})", F(15, 600), MUT)
    byline(d, 64, H_ - 28, size=16)
    im.save(path)


md_all, cards_html = [], ""
for k, w in sorted(W.items(), key=lambda kv: (kv[0] != "uganda", kv[0])):
    c = costs.get(k)
    brief_card(k, w, c, 1080, 1350, MK / f"brief_{k}_4x5.png")
    brief_card(k, w, c, 1600, 900, MK / f"brief_{k}_x_16x9.png")
    todo = what_to_do(w, c)
    md = [f"# Site and market · {w['country']} · week of {now:%d %B %Y}", "", f"**{w['headline']}**", ""]
    if c:
        md.append(f"**Costs (CiO):** {c['series']} {c['near']['median']:+.1f}% by {c['near']['period']} (range {c['near']['lo']:+.1f}% to "
                  f"{c['near']['hi']:+.1f}%, base {c['base']})" + (f", then {c['next']['median']:+.1f}% by {c['next']['period']}" if c.get("next") else "")
                  + (f". Oil {c['oil_13w_pct']:+.1f}% over 13 weeks" + (f" ({c['oil_4w_pct']:+.1f}% over 4)" if c.get("oil_4w_pct") is not None else "")
                     + f"; {c['currency']} {c['fx_13w_pct']:+.1f}% over 13 weeks." if c.get("fx_13w_pct") is not None else "."))
    md += ["", "| Trade | Days with a go window, next 7 | Go hours per working day | Hours lost, last 7 days |", "|---|---|---|---|"]
    md += [f"| {t['trade']} | {t['go_days_ahead']} | {t['go_hours_per_day_ahead']:.1f} of {w['hours_per_day']} | {t['lost_hours_last_7']:.0f} |" for t in w["trades"]]
    md += ["", "**What to do this week**", ""] + [f"- {a}" for a in todo] + ["", "Towns: " + ", ".join(f"[{x['name']}]({x['url']})" for x in w["towns"]), "",
           "*Weather: site weather service (Open-Meteo, ECMWF). Costs: CiO Construction Cost Platform. Screening guidance, not instructions. By Gava Charles.*"]
    (MK / f"brief_{k}.md").write_text("\n".join(md) + "\n")
    md_all.append(k)
    cards_html += (f"<section id='{k}'><h2>{html.escape(w['country'])}</h2><img src='brief_{k}_4x5.png' alt='Site and market brief for {html.escape(w['country'])}' loading='lazy'>"
                   f"<p class='lede'><strong>{html.escape(w['headline'])}</strong></p><ul>" + "".join(f"<li>{html.escape(a)}</li>" for a in todo) + "</ul>"
                   f"<p class='lede'><a href='brief_{k}_4x5.png'>Card</a> · <a href='brief_{k}_x_16x9.png'>Card for X</a> · <a href='brief_{k}.md'>Note</a> · "
                   + " · ".join(f"<a href='{x['url']}'>{html.escape(x['name'])}</a>" for x in w["towns"]) + "</p></section>")
(MK / "index.html").write_text(f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Site and Market</title>
<meta name="description" content="Weekly site and market briefs: weather working time and construction input costs, per country. By Gava Charles.">
<meta property="og:title" content="Site and market: weather time and construction costs, week of {now:%d %b %Y}">
<meta property="og:image" content="{BASE}/market/brief_uganda_x_16x9.png?v={now:%Y%m%d}"><meta name="twitter:card" content="summary_large_image">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Manrope:wght@400;600;700;800&display=swap">
<style>
:root{{--bg:#fff;--surface:#fafaf9;--ink:#0a0a0b;--ink-2:#3e3e42;--line:#e1e1e0;--eyebrow:#755200;--link:#2a6fc4}}
@media (prefers-color-scheme: dark){{:root:not([data-theme="light"]){{--bg:#0d0d0e;--surface:#17171a;--ink:#f2f2ef;--ink-2:#b8b8b4;--line:#2a2a2e;--eyebrow:#e3b341;--link:#7fb0ef;color-scheme:dark}}}}
body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.6 Manrope,Arial,sans-serif;padding:28px 16px 48px}}
main{{max-width:1100px;margin:0 auto;display:grid;grid-template-columns:minmax(0,1fr);gap:28px}} h1{{font:800 clamp(30px,5vw,44px)/1.05 Manrope,Arial,sans-serif;margin:0}}
h2{{font:800 24px/1.15 Manrope,Arial,sans-serif;margin:0}} a{{color:var(--link)}} .lede{{color:var(--ink-2);max-width:72ch;margin:0}}
nav{{display:flex;gap:8px;flex-wrap:wrap}} nav a{{color:var(--ink);text-decoration:none;border:1px solid var(--line);padding:7px 14px;border-radius:999px;font-size:13px;font-weight:600}}
section{{display:grid;gap:10px;border-top:1px solid var(--line);padding-top:22px}} section img{{width:100%;max-width:520px;height:auto;border:1px solid var(--line);border-radius:16px}}
.eyebrow{{font-size:12px;font-weight:700;letter-spacing:.14em;text-transform:uppercase;color:var(--eyebrow);margin:0}}
</style></head><body><main>
<nav><a href="../">Site weather</a><a href="../map.html">Map</a><a href="../share.html">Share today</a><a href="./">Site and market</a></nav>
<p class="eyebrow">Market intelligence · week of {now:%d %B %Y}</p>
<h1>Site and market</h1>
<p class="lede">Each week, per country: how much working time the weather allows and took (site weather), and where construction
input costs are heading (CiO Construction Cost Platform). Weather sets the time; costs set the money. Together they inform programme,
contingency and when to buy. Screening guidance, not instructions.</p>
{cards_html}
<p class="lede" style="font-size:13px">Weather: Open-Meteo (CC BY 4.0), ECMWF. Costs: CiO ({CIO_REPO}). Updated daily; costs weekly.</p>
<p style="margin:0;font-weight:800">By Gava Charles · Kampala Heat Atlas</p>
</main></body></html>""")
print(f"market: weather for {len(W)} countries, costs for {sum(1 for k in W if k in costs)}, briefs written")
