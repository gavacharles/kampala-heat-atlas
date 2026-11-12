"""Social cards for the site weather service, drawn from the same numbers as the site note.

    python scripts/30_site_cards.py docs/site-weather/central-kampala

Writes <site>/social/:
  today_4x5.png     1080 x 1350  the site briefing: headline, problem trades first (status, window, reason,
                                 action, hour strip), clear trades on one line
  today_9x16.png    1080 x 1920  the same for WhatsApp Status / Stories, with the next three days
  today_hours_4x5.png            the full hour-by-hour grid for all trades
  briefing_A4.pdf/.png           printable morning briefing (toolbox talk) with a sign-off box
  week_4x5.png      1080 x 1350  seven days: workable hours per trade, rain, storm risk
  howto_4x5.png     1080 x 1350  how to read the card (pin this once)
  carousel.pdf      today + week + how-to, for LinkedIn document posts
  caption.txt       a ready-to-post caption with the link
House style as src/kampala/story.py (light background, Arial, one hook / one number), standalone so the
daily GitHub Action can run it (falls back to DejaVu Sans where Arial is missing).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

SITE = Path(sys.argv[1] if len(sys.argv) > 1 else "docs/site-weather/central-kampala")
OUTD = SITE / "social"; OUTD.mkdir(parents=True, exist_ok=True)
TH = json.loads((SITE.parent / "thresholds.json").read_text())
W0, W1 = TH["working_hours"]
WORK = [h for h in range(W0, W1) if h not in TH.get("break_hours", [])]     # e.g. 8 working hours, lunch excluded
NW = len(WORK)
NAME = (SITE / "site_note_latest.md").read_text().splitlines()[0].split("· ", 1)[1].strip()
LINK = f"gavacharles.github.io/kampala-heat-atlas/site-weather/{SITE.name}"


def _font(cands, s):
    for c in cands:
        try:
            return ImageFont.truetype(c, s)
        except OSError:
            continue
    return ImageFont.load_default(s)


def F(s, b=False):
    return _font(["/System/Library/Fonts/Supplemental/Arial Bold.ttf" if b else "/System/Library/Fonts/Supplemental/Arial.ttf",
                  "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if b else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                  "DejaVuSans-Bold.ttf" if b else "DejaVuSans.ttf"], s)


BG, CARD, INK, INK2, MUT, LINE = (246, 246, 243), (255, 255, 255), (11, 11, 11), (82, 81, 78), (137, 135, 129), (225, 224, 217)
GO, CAU, STOP, OFF, BLUE = (47, 143, 58), (224, 165, 38), (194, 52, 52), (236, 235, 230), (42, 120, 214)
COL = {0: GO, 1: CAU, 2: STOP}
WORD = {0: "GO", 1: "CAUTION", 2: "STOP"}
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


def best(mask, idx):
    r = runs(mask, idx)
    return max(r, key=lambda x: x[1] - x[0]) if r else None


def text(d, xy, s, f, fill=INK, anchor="la"):
    d.text(xy, s, font=f, fill=fill, anchor=anchor)


def wrap(d, s, f, width):
    lines, cur = [], ""
    for wd in s.split():
        t = (cur + " " + wd).strip()
        if d.textlength(t, font=f) <= width:
            cur = t
        else:
            lines.append(cur); cur = wd
    return lines + [cur]


def header(d, W, title, sub, top=0):
    text(d, (64, 64 + top), "SITE WEATHER · " + NAME.upper(), F(26, True), BLUE)
    text(d, (64, 104 + top), title, F(60, True))
    text(d, (64, 178 + top), sub, F(28), INK2)


def footer(d, W, H):
    d.line((64, H - 140, W - 64, H - 140), fill=LINE, width=2)
    text(d, (64, H - 118), "Screening guidance from a forecast. Your contract, crane limits and site H&S plan come first.", F(20), MUT)
    text(d, (64, H - 86), LINK, F(22, True), INK2)
    text(d, (64, H - 54), "Gava Charles · Kampala Heat Atlas · forecast data Open-Meteo", F(20), MUT)


def day_facts():
    rain, pm = g.precipitation.sum(), g.precipitation_probability.max()
    storm = runs(g.thunder.values >= 1, g.index)
    pour = best(w.st_concrete.values == 0, w.index)
    return [("RAIN", f"{rain:.0f} mm", f"chance up to {pm:.0f}%"),
            ("STORM RISK", f"{storm[0][0]:02d}-{storm[-1][1]:02d} h" if storm else "low", "stop exposed work at first thunder" if storm else "keep an eye on the sky"),
            ("HEAT (WBGT, SUN)", f"{g.wbgt_sun.max():.1f} °C", "continuous work OK" if w.st_labour.max() == 0 else "use work/rest cycles"),
            ("BEST POUR", f"{pour[0]:02d}-{pour[1]:02d} h" if pour else "none", "longest clear concrete window")]


def advice():
    out = []
    storm = runs(w.thunder.values >= 1, w.index)
    pour = best(w.st_concrete.values == 0, w.index)
    if w.st_labour.max() >= 1:
        hot = runs(w.st_labour.values >= 1, w.index)
        out.append(f"Heat stress {hot[0][0]:02d}-{hot[-1][1]:02d} h: rest breaks in shade, water every 20 minutes, heavy tasks early.")
    if storm:
        out.append(f"Thunderstorms possible {storm[0][0]:02d}-{storm[-1][1]:02d} h: plan lifts and roof work before then.")
    if pour:
        out.append(f"Pour concrete {pour[0]:02d}-{pour[1]:02d} h; have curing cover ready.")
    else:
        out.append("No clear concrete window today: postpone pours or protect against rain.")
    asp = best(w.st_asphalt.values == 0, w.index)
    out.append(f"Asphalt: dry window {asp[0]:02d}-{asp[1]:02d} h." if asp else "Asphalt: no dry window; don't lay on a wet surface.")
    if w.rain_prev24.max() >= TH["earthworks"]["rain_prev24h_caution_mm"] if "rain_prev24" in w else False:
        out.append("Ground is wet from recent rain: protect formation and stockpiles.")
    ok = runs(w.st_coatings.values == 0, w.index)
    out.append(f"Painting window {ok[0][0]:02d}-{ok[-1][1]:02d} h." if ok else "Too humid for painting and coatings.")
    return out[:4]


def facts_block(d, x, y, W):
    cw = (W - 2 * x - 3 * 16) // 4
    for k, (lab, big, sub) in enumerate(day_facts()):
        cx = x + k * (cw + 16)
        d.rounded_rectangle((cx, y, cx + cw, y + 170), 14, fill=CARD, outline=LINE, width=2)
        text(d, (cx + 18, y + 18), lab, F(18, True), MUT)
        text(d, (cx + 18, y + 50), big, F(44, True))
        for i, ln in enumerate(wrap(d, sub, F(18), cw - 36)[:2]):
            text(d, (cx + 18, y + 112 + 24 * i), ln, F(18), INK2)
    return y + 170


def trade_grid(d, x, y, W, row_h=78):
    hrs = WORK
    lab_w = 300; gx = x + lab_w; gw = W - x - gx
    cw = gw / len(hrs)
    for i, hh in enumerate(hrs):
        if hh % 2 == 1:
            text(d, (gx + (i + 0.5) * cw, y), f"{hh:02d}", F(20), MUT, "ma")
    y += 34
    for name_k, name in TRADES:
        st = w[f"st_{name_k}"].values
        ngo = int((st == 0).sum())
        text(d, (x, y + row_h / 2 - 14), name, F(28, True), INK, "lm")
        text(d, (x, y + row_h / 2 + 18), f"{ngo} of {len(hrs)} h go", F(20), INK2, "lm")
        for i, s in enumerate(st):
            d.rounded_rectangle((gx + i * cw + 2, y + 10, gx + (i + 1) * cw - 2, y + row_h - 10), 6, fill=COL[int(s)])
        y += row_h
    # legend with words, so colour is never the only cue
    lx = x
    for s in (0, 1, 2):
        d.rounded_rectangle((lx, y + 14, lx + 26, y + 40), 5, fill=COL[s])
        text(d, (lx + 36, y + 27), WORD[s], F(20, True), INK2, "lm")
        lx += 36 + d.textlength(WORD[s], font=F(20, True)) + 40
    text(d, (W - 64, y + 27), f"working hours {W0:02d}-{W1:02d}, lunch excluded", F(20), MUT, "rm")
    return y + 54


def today_card(W, H, path):
    im = Image.new("RGB", (W, H), BG); d = ImageDraw.Draw(im)
    top = 110 if H >= 1600 else 0                            # clear the Stories profile bar
    header(d, W, f"{day:%A %d %B}", f"Hour-by-hour: what can go ahead on site · issued {issued:%H:%M}", top)
    y = facts_block(d, 64, 240 + top, W) + 44
    y = trade_grid(d, 64, y, W, row_h=62 if H < 1600 else 88) + 28
    text(d, (64, y), "ON SITE TODAY", F(22, True), BLUE); y += 40
    for a in advice():
        if y + 36 > H - 150:
            break
        for i, ln in enumerate(wrap(d, a, F(26), W - 128 - 30)):
            if i == 0:
                d.ellipse((64, y + 10, 76, y + 22), fill=INK)
            text(d, (94, y), ln, F(26), INK); y += 36
        y += 8
        if y > H - 150:
            break
    if H >= 1600:                                            # vertical: a short outlook fills the lower third
        y += 30
        text(d, (64, y), "NEXT DAYS", F(22, True), BLUE); y += 44
        for dd, gg in days[1:4]:
            ww = gg[gg.index.hour.isin(WORK)]
            lvl = int(gg.thunder.max())
            text(d, (64, y), f"{dd:%a %d}", F(28, True))
            text(d, (230, y), f"rain {gg.precipitation.sum():.0f} mm", F(26), BLUE)
            text(d, (450, y), ["storms low", "storms possible", "storms likely"][lvl], F(26), [INK2, (150, 100, 0), STOP][lvl])
            text(d, (W - 64, y), f"concrete {int((ww.st_concrete == 0).sum())} h go", F(26), INK2, "ra")
            y += 50
    footer(d, W, H)
    im.save(path)
    return im


def week_card(path):
    W, H = 1080, 1350
    im = Image.new("RGB", (W, H), BG); d = ImageDraw.Draw(im)
    header(d, W, "The week on site", f"Hours each trade can go ahead, out of the {NW}-hour working day")
    x0, y0, lab_w = 64, 260, 290
    cw = (W - 128 - lab_w) / len(days)
    for j, (dd, gg) in enumerate(days):
        cx = x0 + lab_w + (j + 0.5) * cw
        text(d, (cx, y0), f"{dd:%a}", F(26, True), INK, "ma")
        text(d, (cx, y0 + 34), f"{dd:%d}", F(20), MUT, "ma")
    y = y0 + 80
    for k, name in TRADES:
        text(d, (x0, y + 40), name, F(26, True), INK, "lm")
        for j, (dd, gg) in enumerate(days):
            ww = gg[gg.index.hour.isin(WORK)]
            n = int((ww[f"st_{k}"] == 0).sum()); share = n / max(len(ww), 1)
            col = GO if share >= 0.66 else CAU if share >= 0.33 else STOP
            cx = x0 + lab_w + j * cw
            d.rounded_rectangle((cx + 5, y + 6, cx + cw - 5, y + 74), 10, fill=col)
            text(d, (cx + cw / 2, y + 32), f"{n}h", F(30, True), (255, 255, 255), "mm")
            text(d, (cx + cw / 2, y + 60), "go", F(16, True), (255, 255, 255), "mm")
        y += 80
    y += 10
    text(d, (x0, y + 30), "Rain", F(26, True), INK, "lm")
    text(d, (x0, y + 90), "Storm risk", F(26, True), INK, "lm")
    for j, (dd, gg) in enumerate(days):
        cx = x0 + lab_w + (j + 0.5) * cw
        text(d, (cx, y + 30), f"{gg.precipitation.sum():.0f} mm", F(24, True), BLUE, "mm")
        text(d, (cx, y + 58), f"{gg.precipitation_probability.max():.0f}%", F(18), MUT, "mm")
        lvl = int(gg.thunder.max())
        text(d, (cx, y + 96), ["low", "possible", "likely"][lvl], F(22, True), [INK2, (150, 100, 0), STOP][lvl], "mm")
    y += 140
    lx = x0
    for col, lab in ((GO, "most of the day (8h+)"), (CAU, "part of the day (4-7h)"), (STOP, "little or none (0-3h)")):
        d.rounded_rectangle((lx, y, lx + 26, y + 26), 5, fill=col)
        text(d, (lx + 36, y + 13), lab, F(20), INK2, "lm"); lx += 36 + d.textlength(lab, font=F(20)) + 36
    text(d, (x0, y + 60), f"Example: 3h = only 3 of the {NW} hours ({W0:02d}-{W1:02d} h) are clear of rain, storm, wind and heat limits for that trade.", F(20), MUT)
    footer(d, W, H)
    im.save(path)
    return im


def howto_card(path):
    W, H = 1080, 1350
    im = Image.new("RGB", (W, H), BG); d = ImageDraw.Draw(im)
    header(d, W, "How to read the card", "A daily weather check for contractors and site managers")
    y = 270
    items = [("The headline is the one thing to plan around", "Then each trade, problems first: STOP, LIMITED (go only in the hours shown), CARE, and GO."),
             ("Each small block is an hour", f"{NW} working hours, {W0:02d}:00 to {W1:02d}:00 with lunch excluded. Green: go. Amber: go with care. Red: stop or reschedule."),
             ("Heat uses WBGT", "Wet-bulb globe temperature, the standard for heat stress at work. It sets work/rest cycles."),
             ("Concrete looks at rain and drying", "Rain in the next hours, and evaporation fast enough to crack fresh concrete (ACI 305)."),
             ("Cranes use gusts at hook height", "Gusts scaled to 30 m, against typical tower-crane limits, plus thunderstorm risk."),
             ("Painting needs dry air", "Surface 3 °C above dew point and humidity under 85 % (ISO 8502-4)."),
             ("Every day is kept on record", "Month by month against the 1991-2020 climate, for programme reviews and delay claims.")]
    for h, s in items:
        text(d, (64, y), h, F(30, True)); y += 42
        for ln in wrap(d, s, F(24), W - 128):
            text(d, (64, y), ln, F(24), INK2); y += 32
        y += 22
    footer(d, W, H)
    im.save(path)
    return im


# ---------------------------------------------------------------- briefing card and A4 sheet (from site_briefing_latest.json)
BJ = json.loads((SITE / "site_briefing_latest.json").read_text())
B = BJ["days"][0]
PILL = {"STOP": (STOP, "STOP"), "LIMITED": (CAU, "LIMITED"), "CAUTION": (CAU, "CARE"), "GO": (GO, "GO")}
WORST = min(B["rows"], key=lambda r: ["STOP", "LIMITED", "CAUTION", "GO"].index(r["status"]))["status"]


def pill(d, x, y, status, w=128, h=38, size=20):
    col, word = PILL[status]
    d.rounded_rectangle((x, y, x + w, y + h), h // 2, fill=col)
    text(d, (x + w / 2, y + h / 2), word, F(size, True), (255, 255, 255), "mm")


CELL, GAP, LUNCH = 31, 3, 12


def cell_x(x, i):
    """x of hour cell i; a wider gap marks the lunch break."""
    brk = sum(1 for j in range(1, i + 1) if WORK[j] != WORK[j - 1] + 1)
    return x + i * (CELL + GAP) + brk * LUNCH


def strip_width():
    return cell_x(0, len(WORK) - 1) + CELL


def strip(d, x, y, key, h=30):
    """One cell per working hour with the hour written in it, so the go hours can be read directly."""
    st = w[f"st_{key}"].values
    for i, (v, hh) in enumerate(zip(st, WORK)):
        cx = cell_x(x, i)
        d.rounded_rectangle((cx, y, cx + CELL, y + h), 5, fill=COL[int(v)])
        text(d, (cx + CELL / 2, y + h / 2 + 1), f"{hh:02d}", F(15, True), (255, 255, 255), "mm")
    return x + strip_width()


def strip_axis(d, x, y):
    lunch = [i for i in range(1, len(WORK)) if WORK[i] != WORK[i - 1] + 1]
    text(d, (x + strip_width(), y), "hour of day" + ("  ·  gap = lunch" if lunch else ""), F(16), MUT, "ra")


def headline_box(d, x, y, W, size=34):
    lines = wrap(d, B["headline"], F(size, True), W - 2 * x - 60)
    hgt = 36 + len(lines) * (size + 10)
    d.rounded_rectangle((x, y, W - x, y + hgt), 14, fill=CARD, outline=LINE, width=2)
    d.rounded_rectangle((x, y, x + 12, y + hgt), 6, fill=PILL[WORST][0])
    for k, ln in enumerate(lines):
        text(d, (x + 36, y + 18 + k * (size + 10)), ln, F(size, True))
    return y + hgt


def facts_line(d, x, y):
    f = B["facts"]
    s = (f"Rain {f['rain_mm']:.0f} mm ({f['rain_chance']}%)  ·  Storms {f['storm']}" + (f" from {f['storm_from']}:00" if f["storm_from"] else "")
         + f"  ·  Heat (WBGT) {f['wbgt_peak']:.0f} °C  ·  Gusts {f['gust_hook_max']:.0f} m/s")
    text(d, (x, y), s, F(24), INK2)
    return y + 40


def briefing_rows(d, x, y, W, ymax, size=28, small=22, show_go=False):
    """Problem trades first: pill, trade, hour strip; then label, reason and action. Clear trades on one line."""
    sx = W - x - strip_width()
    strip_axis(d, sx, y)
    y += 26
    for r in B["rows"]:
        if r["status"] == "GO" and not show_go:
            continue
        sub = r["label"][0].upper() + r["label"][1:] + (" · limited by " + ", ".join(r["causes"]) if r["causes"] and r["key"] != "labour" else "")
        if r["status"] == "STOP":
            sub = ("Stopped by " + ", ".join(r["causes"])) if r["causes"] else "No window"
        body = r["action"] if r["status"] == "GO" else sub + ". " + r["action"]
        lines = wrap(d, body, F(small), sx - x - 158 - 20)[:3]
        hgt = 46 + len(lines) * (small + 8) + 14
        if y + hgt > ymax:
            break
        pill(d, x, y + 2, r["status"])
        text(d, (x + 150, y), r["trade"], F(size, True))
        strip(d, sx, y + 2, r["key"])
        for k, ln in enumerate(lines):
            text(d, (x + 150, y + 42 + k * (small + 8)), ln, F(small), INK2)
        y += hgt
        d.line((x, y - 7, W - x, y - 7), fill=LINE, width=1)
    clear = [r["trade"] for r in B["rows"] if r["status"] == "GO"]
    if clear and not show_go and y + 50 < ymax:
        pill(d, x, y + 4, "GO")
        for k, ln in enumerate(wrap(d, "No weather limits: " + ", ".join(clear) + ".", F(size - 2, True), W - 2 * x - 150)[:2]):
            text(d, (x + 150, y + 6 + k * (size + 4)), ln, F(size - 2, True))
        y += 56 + (size + 4) * (len(wrap(d, "No weather limits: " + ", ".join(clear) + ".", F(size - 2, True), W - 2 * x - 150)[:2]) - 1)
    return y


def briefing_card(W, H, path):
    im = Image.new("RGB", (W, H), BG); d = ImageDraw.Draw(im)
    top = 110 if H >= 1600 else 0
    header(d, W, B["day"], f"Site briefing · issued {BJ['issued'][-5:]} · {NW}-hour day, lunch excluded", top)
    y = headline_box(d, 64, 236 + top, W) + 18
    y = facts_line(d, 64, y) + 10
    y = briefing_rows(d, 64, y, W, H - 160 - (330 if H >= 1600 else 0))
    for e in B["extras"]:
        lines = wrap(d, e, F(22), W - 128)
        if y + 30 * len(lines) > H - 160 - (330 if H >= 1600 else 0):
            break
        for ln in lines:
            text(d, (64, y), ln, F(22), MUT); y += 30
        y += 6
    if H >= 1600:
        y = H - 160 - 300
        text(d, (64, y), "NEXT DAYS", F(22, True), BLUE); y += 44
        for b in BJ["days"][1:4]:
            text(d, (64, y), b["day"].split()[0][:3] + " " + b["day"].split()[1], F(26, True))
            for k, ln in enumerate(wrap(d, b["headline"], F(23), W - 128 - 140)[:2]):
                text(d, (200, y + 2 + k * 30), ln, F(23), INK2)
            y += 76
    footer(d, W, H)
    im.save(path)
    return im


def a4_sheet(path_pdf, path_png):
    W, H = 1240, 1754                                         # A4 at 150 dpi
    im = Image.new("RGB", (W, H), (255, 255, 255)); d = ImageDraw.Draw(im)
    header(d, W, B["day"], f"Morning site briefing · {NAME} · issued {BJ['issued']}")
    y = headline_box(d, 64, 236, W) + 18
    y = facts_line(d, 64, y) + 8
    y = briefing_rows(d, 64, y, W, H - 560, show_go=True)
    for e in B["extras"]:
        text(d, (64, y), e, F(21), INK2); y += 30
    y += 14
    text(d, (64, y), "THE WEEK", F(20, True), BLUE); y += 34
    for b in BJ["days"][1:6]:
        text(d, (64, y), b["day"], F(21, True))
        text(d, (330, y), wrap(d, b["headline"], F(21), W - 394)[0], F(21), INK2)
        y += 32
    y = max(y + 20, H - 330)
    d.rounded_rectangle((64, y, W - 64, y + 170), 10, outline=INK2, width=2)
    text(d, (84, y + 16), "BRIEFING RECORD", F(18, True), MUT)
    for k, lab in enumerate(["Briefing given by", "Date and time", "Workers present (number)", "Signature"]):
        cx, cy = 84 + (k % 2) * 560, y + 58 + (k // 2) * 56
        text(d, (cx, cy), lab, F(20), INK2)
        d.line((cx + d.textlength(lab, font=F(20)) + 12, cy + 22, cx + 520, cy + 22), fill=MUT, width=1)
    footer(d, W, H)
    im.save(path_png)
    im.save(path_pdf, resolution=150)
    return im


a = briefing_card(1080, 1350, OUTD / "today_4x5.png")
briefing_card(1080, 1920, OUTD / "today_9x16.png")
today_card(1080, 1350, OUTD / "today_hours_4x5.png")          # the full hour-by-hour grid, for the detail-minded
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
print("wrote", OUTD, "for", f"{day:%a %d %b}")
