"""Social cards for the site weather service, drawn from the same numbers as the site note.

    python scripts/30_site_cards.py docs/site-weather/central-kampala

Writes <site>/social/:
  today_4x5.png     1080 x 1350  the site briefing: label, headline with the key time marked, fact chips, a card per
                                 trade with limits (status pill, go window, hour chips, reason and action), clear trades
  today_9x16.png    1080 x 1920  the same for WhatsApp Status / Stories, with the next three days
  today_hours_4x5.png            the full hour-by-hour grid for all trades
  briefing_A4.pdf/.png           printable morning briefing (toolbox talk) with a sign-off box
  week_4x5.png      1080 x 1350  seven days: the go windows per trade, rain, storm risk
  howto_4x5.png     1080 x 1350  how to read the cards (pin this once)
  carousel.pdf      today + week + how-to, for LinkedIn document posts
  caption.txt       a ready-to-post caption with the link (WhatsApp, LinkedIn, Facebook)
  caption_x.txt     the same within X's 280 characters
Look: scripts/card_style.py (same as the site note on the town pages). Standalone so the daily Action can run it.
"""
import json
import math
import sys
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent))
from card_style import (ACCENT, BG, CARD, CAU, EYEBROW, GO, INK, INK2, LINE, MUT, SHARE_TINT, STOP, TINTED, F,  # noqa: E402
                        card, chips, footer, header, marked_headline, pill, spaced, text, wrap)

SITE = Path(sys.argv[1] if len(sys.argv) > 1 else "docs/site-weather/central-kampala")
OUTD = SITE / "social"; OUTD.mkdir(parents=True, exist_ok=True)
TH = json.loads((SITE.parent / "thresholds.json").read_text())
W0, W1 = TH["working_hours"]
WORK = [h for h in range(W0, W1) if h not in TH.get("break_hours", [])]     # e.g. 8 working hours, lunch excluded
NW = len(WORK)
NAME = (SITE / "site_note_latest.md").read_text().splitlines()[0].split("· ", 1)[1].strip()
TOWN = NAME.replace("Central ", "")
LINK = f"gavacharles.github.io/kampala-heat-atlas/site-weather/{SITE.name}"
COL = {0: GO, 1: CAU, 2: STOP}
TRADES = [("labour", "Outdoor labour"), ("concrete", "Concrete pour"), ("asphalt", "Asphalt & surfacing"), ("crane", "Crane lifts"),
          ("height", "Work at height"), ("coatings", "Painting & coatings"), ("earthworks", "Earthworks")]

df = pd.read_csv(SITE / "site_forecast_latest.csv", index_col=0)
df.index = pd.to_datetime(df.index.str.slice(0, 19))
issued = pd.Timestamp((SITE / "site_note_latest.md").read_text().split("Issued ")[1].split(" (")[0].replace(",", ""))
days = [(d, g) for d, g in df.groupby(df.index.normalize())]
if issued.hour >= W1:
    days = days[1:]
day, g = days[0]
w = g[g.index.hour.isin(WORK)]
BJ = json.loads((SITE / "site_briefing_latest.json").read_text())
B = BJ["days"][0]


def runs(mask, idx):
    out, s = [], None
    for k, ok in enumerate(list(mask) + [False]):
        if ok and s is not None and idx[k].hour != idx[k - 1].hour + 1:       # gap (lunch): close the run
            out.append((idx[s].hour, idx[k - 1].hour + 1)); s = None
        if ok and s is None:
            s = k
        if not ok and s is not None:
            out.append((idx[s].hour, idx[k - 1].hour + 1)); s = None
    return out


def go_windows(st, idx):
    if len(st) and (st == 0).all():
        return ["all day"]
    return [f"{a:02d}-{b:02d}" for a, b in runs(st == 0, idx)] or ["none"]


def fact_items(b=B):
    f = b["facts"]
    return [f"Rain {f['rain_mm']:.0f} mm · {f['rain_chance']}%", "Storms " + f["storm"] + (f" from {f['storm_from']}:00" if f["storm_from"] else ""),
            f"Heat {f['wbgt_peak']:.0f} °C WBGT", f"Gusts {f['gust_hook_max']:.0f} m/s"]


def hour_chips(d, x, y, width, hours, h=34, size=15):
    """One chip per working hour, coloured by status, hour written in; a wider gap at lunch."""
    n = len(hours); gaps = sum(1 for i in range(1, n) if hours[i][0] != hours[i - 1][0] + 1)
    gap, lunch = 4, 10
    cw = (width - gap * (n - 1) - lunch * gaps) / max(n, 1)
    xx = x
    for i, (hh, v) in enumerate(hours):
        if i and hh != hours[i - 1][0] + 1:
            xx += lunch
        d.rounded_rectangle((xx, y, xx + cw, y + h), 8, fill=COL[int(v)])
        text(d, (xx + cw / 2, y + h / 2 + 1), f"{hh:02d}", F(size, 800), (255, 255, 255), "mm")
        xx += cw + gap
    return y + h


def trade_title(r):
    if r["status"] == "STOP":
        return "No window today"
    if r["key"] == "labour":
        return "Work/rest cycles"
    return f"Go {r['windows']}" if r["windows"] else "Go with care"


def trade_body(r):
    why = ("Limited by " + ", ".join(r["causes"]) + ". ") if r["causes"] and r["key"] != "labour" and r["status"] != "GO" else ""
    return why + r["action"]


def trade_cards(d, x, y, W, ymax, rows, cols=2, gap=16):
    """A card per trade: spaced trade label, tinted pill + go window, hour chips, reason and action."""
    rows = list(rows)
    if not rows:
        return y
    nrow = math.ceil(len(rows) / cols)
    cw = (W - 2 * x - gap * (cols - 1)) / cols
    ch = min(262, (ymax - y - gap * (nrow - 1)) / nrow)
    body_lines = max(1, int((ch - 24 - 26 - 16 - 36 - 14 - 34 - 20) // 26))
    for i, r in enumerate(rows):
        cx = x + (i % cols) * (cw + gap); cy = y + (i // cols) * (ch + gap)
        card(d, (cx, cy, cx + cw, cy + ch))
        spaced(d, (cx + 22, cy + 22), r["trade"], size=15)
        px = pill(d, cx + 22, cy + 52, r["status"], size=16, h=32)
        title = trade_title(r)
        tf = F(24 if d.textlength(title, font=F(24, 800)) < cw - (px - cx) - 36 else 20, 800)
        text(d, (px + 12, cy + 68), title, tf, INK, "lm")
        yy = hour_chips(d, cx + 22, cy + 100, cw - 44, r.get("hours", []))
        for k, ln in enumerate(wrap(d, trade_body(r), F(19, 400), cw - 44)[:body_lines]):
            text(d, (cx + 22, yy + 14 + k * 26), ln, F(19, 400), INK2)
    return y + nrow * ch + (nrow - 1) * gap


def clear_card(d, x, y, W, names):
    """One card for the trades with no weather limits."""
    if not names:
        return y
    lines = wrap(d, ", ".join(names) + ".", F(22, 800), W - 2 * x - 44)
    h = 70 + 30 * len(lines)
    card(d, (x, y, W - x, y + h))
    px = pill(d, x + 22, y + 18, "GO", size=16, h=32)
    spaced(d, (px + 14, y + 26), "No weather limits", size=15)
    for k, ln in enumerate(lines):
        text(d, (x + 22, y + 62 + k * 30), ln, F(22, 800))
    return y + h


def briefing_card(W, H, path):
    im = Image.new("RGB", (W, H), BG); d = ImageDraw.Draw(im)
    top = 110 if H >= 1600 else 0
    spaced(d, (64, 60 + top), f"Site weather · {TOWN} · {B['day']}")
    y = marked_headline(d, 64, 100 + top, B["headline"], W - 128, size=44 if H >= 1600 else 40) + 10
    y = chips(d, 64, y, fact_items(), W - 128) + 22
    limited = [r for r in B["rows"] if r["status"] != "GO"]
    clear = [r["trade"] for r in B["rows"] if r["status"] == "GO"]
    bottom = H - 170 - (380 if H >= 1600 else 0) - (90 if clear else 0)
    y = trade_cards(d, 64, y, W, bottom, limited) + 16
    y = clear_card(d, 64, y, W, clear) + 12
    if H >= 1600:
        y = max(y + 10, H - 170 - 360)
        spaced(d, (64, y), "Next days"); y += 36
        for b in BJ["days"][1:4]:
            card(d, (64, y, W - 64, y + 96), fill=BG)
            spaced(d, (86, y + 18), b["day"], size=14)
            text(d, (86, y + 46), wrap(d, b["headline"], F(22, 700), W - 172)[0], F(22, 700))
            y += 108
    footer(d, W, H, LINK)
    im.save(path)
    return im


def a4_sheet(path_pdf, path_png):
    W, H = 1240, 1754                                          # A4 at 150 dpi
    im = Image.new("RGB", (W, H), BG); d = ImageDraw.Draw(im)
    spaced(d, (64, 60), f"Morning site briefing · {TOWN} · {B['day']} · issued {BJ['issued'][-5:]}")
    y = marked_headline(d, 64, 100, B["headline"], W - 128, size=40) + 10
    y = chips(d, 64, y, fact_items(), W - 128) + 22
    box = H - 372                                              # briefing record sits above the footer
    y = trade_cards(d, 64, y, W, box - 260, B["rows"]) + 16
    for e in B["extras"]:
        text(d, (64, y), e, F(19, 400), INK2); y += 28
    y += 12
    spaced(d, (64, y), "The week"); y += 34
    for b in BJ["days"][1:6]:
        if y + 28 > box - 12:
            break
        text(d, (64, y), b["day"], F(19, 800))
        text(d, (330, y), wrap(d, b["headline"], F(19, 400), W - 394)[0], F(19, 400), INK2)
        y += 28
    y = box
    card(d, (64, y, W - 64, y + 172))
    spaced(d, (88, y + 20), "Briefing record", size=15)
    for k, lab in enumerate(["Briefing given by", "Date and time", "Workers present (number)", "Signature"]):
        cx, cy = 88 + (k % 2) * 560, y + 64 + (k // 2) * 54
        text(d, (cx, cy), lab, F(19, 600), INK2)
        d.line((cx + d.textlength(lab, font=F(19, 600)) + 12, cy + 22, cx + 520, cy + 22), fill=MUT, width=1)
    footer(d, W, H, LINK)
    im.save(path_png)
    im.save(path_pdf, resolution=150)
    return im


def hours_card(path):
    """Every trade, hour by hour, today."""
    W, H = 1080, 1350
    im = Image.new("RGB", (W, H), BG); d = ImageDraw.Draw(im)
    header(d, f"Site weather · {TOWN}", f"{day:%A %d %B}", "Hour by hour: what can go ahead on site")
    y = chips(d, 64, 250, fact_items(), W - 128) + 30
    for k, name in TRADES:
        r = next(x for x in B["rows"] if x["key"] == k)
        card(d, (64, y, W - 64, y + 104))
        spaced(d, (88, y + 20), name, size=15)
        pill(d, 88, y + 52, r["status"], size=15, h=30)
        hour_chips(d, 400, y + 34, W - 64 - 24 - 400, r.get("hours", []), h=40, size=16)
        y += 116
    lx = 64
    for col, lab in ((GO, "Go"), (CAU, "Care / limited"), (STOP, "Stop")):
        d.rounded_rectangle((lx, y + 12, lx + 24, y + 36), 6, fill=col)
        text(d, (lx + 34, y + 24), lab, F(19, 600), INK2, "lm"); lx += 34 + d.textlength(lab, font=F(19, 600)) + 30
    text(d, (W - 64, y + 24), f"{W0:02d}-{W1:02d} h, lunch excluded", F(18, 400), MUT, "rm")
    footer(d, W, H, LINK)
    im.save(path)
    return im


def week_card(path):
    W, H = 1080, 1350
    im = Image.new("RGB", (W, H), BG); d = ImageDraw.Draw(im)
    header(d, f"Site weather · {TOWN}", "The week on site", f"When each trade can go ahead, {W0:02d}-{W1:02d} h, lunch excluded")
    x0, y0, lab_w = 64, 250, 250
    cw = (W - 128 - lab_w) / len(days)
    for j, (dd, gg) in enumerate(days):
        cx = x0 + lab_w + (j + 0.5) * cw
        text(d, (cx, y0), f"{dd:%a}", F(24, 800), INK, "ma")
        text(d, (cx, y0 + 32), f"{dd:%d}", F(18, 600), MUT, "ma")
    y = y0 + 74
    for k, name in TRADES:
        text(d, (x0, y + 38), name, F(22, 800), INK, "lm")
        for j, (dd, gg) in enumerate(days):
            ww = gg[gg.index.hour.isin(WORK)]
            st = ww[f"st_{k}"].values
            share = (st == 0).sum() / max(len(ww), 1)
            bg, fg = SHARE_TINT[0 if share >= 0.66 else 1 if share >= 0.33 else 2]
            cx = x0 + lab_w + j * cw
            d.rounded_rectangle((cx + 3, y + 4, cx + cw - 3, y + 72), 12, fill=bg)
            win = go_windows(st, ww.index)
            win = win[:2] if len(win) <= 2 else [win[0], win[1] + "+"]
            for li, wtxt in enumerate(win):
                text(d, (cx + cw / 2, y + (38 if len(win) == 1 else 24 + li * 26)), wtxt, F(19 if len(wtxt) <= 6 else 17, 800), fg, "mm")
        y += 78
    y += 8
    text(d, (x0, y + 26), "Rain", F(22, 800), INK, "lm")
    text(d, (x0, y + 82), "Storms", F(22, 800), INK, "lm")
    for j, (dd, gg) in enumerate(days):
        cx = x0 + lab_w + (j + 0.5) * cw
        text(d, (cx, y + 22), f"{gg.precipitation.sum():.0f} mm", F(21, 800), INK, "mm")
        text(d, (cx, y + 46), f"{gg.precipitation_probability.max():.0f}%", F(16, 600), MUT, "mm")
        lvl = int(gg.thunder.max())
        bg, fg = [TINTED["GO"], TINTED["LIMITED"], TINTED["STOP"]][lvl]
        word = ["low", "possible", "likely"][lvl]
        tw = d.textlength(word, font=F(15, 800)) + 20
        d.rounded_rectangle((cx - tw / 2, y + 68, cx + tw / 2, y + 96), 14, fill=bg)
        text(d, (cx, y + 82), word, F(15, 800), fg, "mm")
    y += 124
    lx = x0
    for s, lab in ((0, "go most of the day"), (1, "part of the day"), (2, "little or none")):
        bg, fg = SHARE_TINT[s]
        d.rounded_rectangle((lx, y, lx + 26, y + 26), 7, fill=bg)
        d.ellipse((lx + 9, y + 9, lx + 17, y + 17), fill=fg)
        text(d, (lx + 36, y + 13), lab, F(19, 600), INK2, "lm"); lx += 36 + d.textlength(lab, font=F(19, 600)) + 30
    text(d, (x0, y + 54), "Times are go windows: 08-13 = from 08:00 until the lunch break; + = more windows that day.", F(18, 400), MUT)
    footer(d, W, H, LINK)
    im.save(path)
    return im


def howto_card(path):
    W, H = 1080, 1350
    im = Image.new("RGB", (W, H), BG); d = ImageDraw.Draw(im)
    header(d, "Site weather · how it works", "How to read the cards", "A daily weather check for contractors and site managers")
    items = [("The headline", "The one thing to plan the day around. The key time is marked in yellow."),
             ("A card per trade", "Problems first: Stop, Limited (go only in the hours shown), Care. Trades with no limits share one card."),
             ("Hour chips", f"{NW} working hours, {W0:02d}:00-{W1:02d}:00, lunch excluded. Green go, amber care, red stop."),
             ("Heat uses WBGT", "The standard for heat stress at work; it sets work/rest cycles."),
             ("Concrete and asphalt", "Rain now or within three hours, wet surfaces, and drying fast enough to crack fresh concrete (ACI 305)."),
             ("Cranes and painting", "Gusts at 30 m hook height and storms; painting needs the surface 3 °C above dew point (ISO 8502-4)."),
             ("Every day on record", "Month by month against the 1991-2020 climate, for programme reviews and delay claims.")]
    y = 250; cw = (W - 128 - 16) / 2
    for i, (h, s) in enumerate(items):
        cx = 64 + (i % 2) * (cw + 16)
        if i % 2 == 0 and i:
            y += 186
        if i == len(items) - 1 and i % 2 == 0:
            cw2 = W - 128
        else:
            cw2 = cw
        card(d, (cx, y, cx + cw2, y + 172))
        spaced(d, (cx + 22, y + 20), f"Step {i + 1:02d}", size=14)
        d.rounded_rectangle((cx + cw2 - 58, y + 16, cx + cw2 - 22, y + 52), 9, fill=(237, 224, 182))
        text(d, (cx + cw2 - 40, y + 34), str(i + 1), F(17, 800), EYEBROW, "mm")
        text(d, (cx + 22, y + 52), h, F(23, 800))
        for k, ln in enumerate(wrap(d, s, F(18, 400), cw2 - 44)[:3]):
            text(d, (cx + 22, y + 90 + k * 24), ln, F(18, 400), INK2)
    footer(d, W, H, LINK)
    im.save(path)
    return im


a = briefing_card(1080, 1350, OUTD / "today_4x5.png")
briefing_card(1080, 1920, OUTD / "today_9x16.png")
hours_card(OUTD / "today_hours_4x5.png")                       # the full hour-by-hour grid, for the detail-minded
a4_sheet(OUTD / "briefing_A4.pdf", OUTD / "briefing_A4.png")
b = week_card(OUTD / "week_4x5.png")
c = howto_card(OUTD / "howto_4x5.png")
a.save(OUTD / "carousel.pdf", save_all=True, append_images=[b, c], resolution=150)
cap = [f"Site weather, {NAME}: {B['day']}.", "", B["headline"], ""] + [
    f"• {r['trade']}: {r['label']}." for r in B["rows"] if r["status"] != "GO"] + [
    f"• No limits: {', '.join(r['trade'] for r in B['rows'] if r['status'] == 'GO')}." if any(r["status"] == "GO" for r in B["rows"]) else ""] + [
    "", f"Hour-by-hour for seven trades, the week ahead and the monthly record: {LINK}", "",
    "Screening guidance from a forecast; your contract, crane limits and H&S plan come first.",
    "#Kampala #Construction #SiteSafety #HeatStress #Uganda #Weather"]
(OUTD / "caption.txt").write_text("\n".join(cap) + "\n")
# X: 280 characters, a link counts as 23
short = {"labour": "Labour", "concrete": "Concrete", "asphalt": "Asphalt", "crane": "Cranes", "height": "Height", "coatings": "Painting", "earthworks": "Earthworks"}
go = [f"{short[r['key']]} {('all day' if r['status'] == 'GO' else 'none' if r['status'] == 'STOP' else (r['windows'].replace(' h', '') or 'with care'))}" for r in B["rows"]]
tag = " #" + NAME.replace("Central ", "").replace(" ", "") + " #Construction"
for m in range(len(go), 0, -1):
    xt = (f"{NAME.replace('Central ', '')} site weather, {day:%a %d %b}: {B['headline']}\nGo today: " + "; ".join(go[:m]) + ("…" if m < len(go) else "")
          + f"\n{LINK}" + tag)
    if len(xt.replace(LINK, "x" * 23)) <= 280:
        break
(OUTD / "caption_x.txt").write_text(xt + "\n")
print("wrote", OUTD, "for", f"{day:%a %d %b}")
