"""Shared look for the site weather cards (scripts/30_site_cards.py, scripts/31_site_weather_all.py), matching the
site note on the town pages: Manrope (SIL OFL, scripts/fonts), white page, near-black ink, one yellow accent used as
a marker behind the key time and for the link button, amber spaced-capital labels, tinted status pills, thin-bordered
rounded cards."""
import re
from pathlib import Path

from PIL import ImageFont

FONTS = Path(__file__).parent / "fonts"
BG, CARD, INK, INK2, MUT, LINE = (255, 255, 255), (250, 250, 249), (10, 10, 11), (96, 96, 97), (138, 138, 136), (225, 225, 224)
ACCENT, EYEBROW, TINT = (251, 202, 12), (138, 97, 0), (253, 246, 220)
GO, CAU, STOP = (47, 143, 58), (224, 165, 38), (194, 52, 52)                 # solid, for hour cells
TINTED = {"STOP": ((249, 234, 234), (180, 35, 24)), "LIMITED": ((253, 243, 215), (138, 97, 0)),
          "CAUTION": ((253, 243, 215), (138, 97, 0)), "GO": ((232, 244, 235), (31, 122, 53))}
WORDS = {"STOP": "Stop", "LIMITED": "Limited", "CAUTION": "Care", "GO": "Go"}
SHARE_TINT = {0: TINTED["GO"], 1: TINTED["LIMITED"], 2: TINTED["STOP"]}       # 0 most of the day, 1 part, 2 little
_cache = {}


def F(size, weight=400):
    """Manrope at 400/600/700/800 (True = 700, False = 400)."""
    weight = 700 if weight is True else 400 if weight is False else weight
    key = (size, weight)
    if key not in _cache:
        for path in (FONTS / f"Manrope-{weight}.ttf", "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if weight >= 600
                     else "/System/Library/Fonts/Supplemental/Arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
            try:
                _cache[key] = ImageFont.truetype(str(path), size); break
            except OSError:
                continue
        else:
            _cache[key] = ImageFont.load_default(size)
    return _cache[key]


def text(d, xy, s, f, fill=INK, anchor="la"):
    d.text(xy, s, font=f, fill=fill, anchor=anchor)


def spaced(d, xy, s, size=18, fill=EYEBROW, track=0.14, weight=700):
    """Spaced capitals (the amber section labels). Returns the x where the text ends."""
    x, y = xy; f = F(size, weight)
    for ch in s.upper():
        d.text((x, y), ch, font=f, fill=fill)
        x += d.textlength(ch, font=f) + size * track
    return x


def wrap(d, s, f, width):
    lines, cur = [], ""
    for wd in s.split():
        t = (cur + " " + wd).strip()
        if d.textlength(t, font=f) <= width or not cur:
            cur = t
        else:
            lines.append(cur); cur = wd
    return lines + ([cur] if cur else [])


HL = re.compile(r"(\d{2}(?::\d{2})?-\d{2} h|from \d{2}:00|\d+ mm)")


def marked_headline(d, x, y, s, width, size=40, gap=10):
    """Big headline; the first time window (or rain amount) gets the yellow marker. Returns the bottom y."""
    f = F(size, 800)
    m = HL.search(s); mark = set()
    words = s.split()
    if m:                                                        # indices of the words inside the marked span
        pos = 0
        for i, wd in enumerate(words):
            start = s.index(wd, pos); end = start + len(wd); pos = end
            if start < m.end() and end > m.start():
                mark.add(i)
    lines, cur, idx = [], [], []
    for i, wd in enumerate(words):
        t = " ".join([words[j] for j in cur] + [wd])
        if cur and d.textlength(t, font=f) > width:
            lines.append(cur); cur = []
        cur.append(i)
    lines.append(cur)
    sp = d.textlength(" ", font=f)
    for li, ids in enumerate(lines):
        yy, xx = y + li * (size + gap), x
        for j in ids:
            wd = words[j]; wl = d.textlength(wd, font=f)
            if j in mark:                                        # marker over the time, not the punctuation around it
                lead = len(wd) - len(wd.lstrip("("))
                core = wd.strip("();:,.")
                x0m = xx + d.textlength(wd[:lead], font=f)
                cl = d.textlength(core, font=f)
                ext = sp if (j + 1) in mark and not wd.endswith((")", ";", ":", ",")) else 0
                d.rounded_rectangle((x0m - 6, yy - 2, x0m + cl + ext + 6, yy + size + 6), 8, fill=ACCENT)
            text(d, (xx, yy), wd, f)
            xx += wl + sp
    return y + len(lines) * (size + gap)


def chips(d, x, y, items, width, size=20, h=40):
    """Rounded fact chips with a thin border; wraps to a new row. Returns the bottom y."""
    f = F(size, 600); xx = x
    for s in items:
        w = d.textlength(s, font=f) + 32
        if xx + w > x + width:
            xx = x; y += h + 10
        d.rounded_rectangle((xx, y, xx + w, y + h), h // 2, fill=CARD, outline=LINE, width=2)
        text(d, (xx + w / 2, y + h / 2), s, f, INK2, "mm")
        xx += w + 10
    return y + h


def pill(d, x, y, status, size=18, h=34):
    """Tinted status pill with a dot: Stop / Limited / Care / Go. Returns its right x."""
    bg, fg = TINTED[status]; word = WORDS[status]; f = F(size, 800)
    w = d.textlength(word, font=f) + 46
    d.rounded_rectangle((x, y, x + w, y + h), h // 2, fill=bg)
    d.ellipse((x + 14, y + h / 2 - 4, x + 22, y + h / 2 + 4), fill=fg)
    text(d, (x + 30, y + h / 2 + 1), word, f, fg, "lm")
    return x + w


def card(d, box, r=22, fill=CARD):
    d.rounded_rectangle(box, r, fill=fill, outline=LINE, width=2)


def header(d, eyebrow, title, sub, top=0, x=64, title_size=58):
    spaced(d, (x, 64 + top), eyebrow)
    text(d, (x, 98 + top), title, F(title_size, 800))
    text(d, (x, 98 + top + title_size + 14), sub, F(24, 400), INK2)


def footer(d, W, H, link, credit="Gava Charles · Kampala Heat Atlas · forecast data Open-Meteo"):
    d.line((64, H - 160, W - 64, H - 160), fill=LINE, width=2)
    text(d, (64, H - 140), "Screening guidance from a forecast. Your contract, crane limits and site H&S plan come first.", F(19, 400), MUT)
    f = F(20, 800); w = d.textlength(link, font=f) + 40
    d.rounded_rectangle((64, H - 106, 64 + w, H - 66), 20, fill=ACCENT)
    text(d, (84, H - 86), link, f, INK, "lm")
    text(d, (64, H - 46), credit, F(16, 600), MUT)
