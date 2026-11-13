"""Construction site weather: 7-day meteogram, a short daily site note, and a growing monthly record.

    python scripts/29_site_weather.py docs/site-weather                       # central Kampala
    python scripts/29_site_weather.py docs/site-weather --lat 0.35 --lon 32.62 --name "Ntinda site"

Standalone (Open-Meteo, free, no key; only the site coordinates are sent) so a daily GitHub Action can
run it. Thresholds are read from <out>/thresholds.json; they are typical screening values, and the
contract specification, crane manufacturer's limits and site health-and-safety plan take precedence.

Per site (<out>/<slug>/):
  site_meteogram_latest.png   WBGT work-rest bands, rain + thunderstorm risk, wind/gust at hook height,
                              concrete evaporation rate, coating dew-point margin, per-trade workable strip
  site_note_latest.md / .html the daily site note generated from the same numbers
  site_forecast_latest.csv    hourly values and per-trade status (0 go, 1 caution, 2 stop)
  archive/forecasts/YYYY-MM/YYYY-MM-DD.csv   each day's forecast (next 72 h), as issued
  archive/observed/YYYY-MM.csv               one row per day: what happened (model analysis, replaced
                                             by ERA5 once available), workable hours lost, flags vs the
                                             1991-2020 climate of the site, and yesterday's forecast
  archive/monthly/YYYY-MM.md                 month summary vs normal, for programme and claim records
  index.html                                 the page

Methods (screening level, documented so a reviewer can check them):
  WBGT     0.7 Tnw + 0.2 Tg + 0.1 Ta; Tg from mean radiant temperature (radiant budget as in
           scripts/25_meteogram.py) via the ISO 7726 globe equation (D 0.15 m); Tnw from the Stull (2011)
           wet bulb plus 0.2 (Tg - Ta) for sun. A WBGT meter on site is the reference.
  Concrete evaporation (kg/m2/h) by Uno (1998): 5[(Tc+18)^2.5 - r(Ta+18)^2.5](V+4)1e-6, V km/h near the slab.
  Coatings surface minus dew point (surface = air by day, 1-2 C below at night), and RH.
  Cranes   10 m gust scaled to hook height by the 1/7 power law.
  Thunderstorms CAPE with rain probability (no lightning observations are used).
"""
import argparse
import datetime as dt
import html
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pvlib
from matplotlib.colors import ListedColormap

ap = argparse.ArgumentParser()
ap.add_argument("out", nargs="?", default="docs/site-weather")
ap.add_argument("--lat", type=float, default=0.3136)
ap.add_argument("--lon", type=float, default=32.5811)
ap.add_argument("--alt", type=float, default=1190)
ap.add_argument("--name", default="Central Kampala")
ap.add_argument("--backfill", help="YYYY-MM-DD: fill the record from ERA5 from this date (one-off)")
A = ap.parse_args()
OUT = Path(A.out); OUT.mkdir(parents=True, exist_ok=True)
SLUG = re.sub(r"[^a-z0-9]+", "-", A.name.lower()).strip("-")
SITE = OUT / SLUG
for d in ("archive/forecasts", "archive/observed", "archive/monthly"):
    (SITE / d).mkdir(parents=True, exist_ok=True)
TH = json.loads((OUT / "thresholds.json").read_text())


def _merge(a, b):
    for k, v in b.items():
        a[k] = _merge(a.get(k, {}), v) if isinstance(v, dict) and isinstance(a.get(k), dict) else v
    return a


_SITES = json.loads((OUT / "sites.json").read_text())["sites"] if (OUT / "sites.json").exists() else []
_ME = next((s for s in _SITES if s["name"] == A.name), {})
TH = _merge(TH, _ME.get("thresholds", {}))                    # per-site working hours and limits
_sb = TH.get("seasonal_break")
if _sb:                                                        # e.g. Saudi midday outdoor-work ban, 15 Jun - 15 Sep
    _md = dt.date.today().strftime("%m-%d")
    if _sb["from"] <= _md <= _sb["to"]:
        TH["break_hours"] = sorted(set(TH.get("break_hours", [])) | set(_sb["hours"]))
        TH["_seasonal_break_active"] = True
COUNTRY = _ME.get("country", "Uganda")
(SITE / "thresholds_site.json").write_text(json.dumps(TH, indent=1))
W0, W1 = TH["working_hours"]
WORK = [h for h in range(W0, W1) if h not in TH.get("break_hours", [])]     # e.g. 8 working hours, lunch excluded
NW = len(WORK)
SIGMA = 5.67e-8
TRADES = ["Outdoor labour (heat)", "Concrete pour", "Asphalt & surfacing", "Crane lifts", "Work at height / roofing", "Painting & coatings", "Earthworks"]
KEYS = ["labour", "concrete", "asphalt", "crane", "height", "coatings", "earthworks"]
INK, INK2, MUTED, GRID = "#0b0b0b", "#3a3a3e", "#55555a", "#e1e0d9"
BLUE, ORANGE, RED, GREEN, AMBER, OFF = "#2a78d6", "#eb6834", "#c23434", "#2f8f3a", "#e0a526", "#ecebe6"
plt.rcParams.update({"font.family": ["Helvetica Neue", "Arial", "DejaVu Sans"], "font.size": 9, "axes.edgecolor": "#c3c2b7",
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.labelcolor": INK2})
BASE_VARS = ("temperature_2m,relative_humidity_2m,dew_point_2m,precipitation,cloud_cover,wind_speed_10m,wind_direction_10m,"
             "wind_gusts_10m,shortwave_radiation,direct_normal_irradiance,diffuse_radiation")
FC_VARS = BASE_VARS + ",precipitation_probability,cape,uv_index,visibility,cloud_cover_low,cloud_cover_mid,cloud_cover_high,pressure_msl"


def get(url, tries=6):
    for k in range(tries):
        try:
            return json.loads(urllib.request.urlopen(url, timeout=90).read())
        except Exception as e:  # noqa: BLE001
            if k == tries - 1:
                raise
            # 429 = the free API's per-minute limit: wait for the next minute; otherwise back off briefly
            time.sleep(62 if getattr(e, "code", None) == 429 else 10 * (k + 1))


def hourly_frame(js):
    df = pd.DataFrame(js["hourly"])
    df["time"] = pd.to_datetime(df["time"]).dt.tz_localize(js["timezone"], ambiguous="NaT", nonexistent="NaT")
    return df.dropna(subset=["time"]).set_index("time")


def climatology():
    p = OUT / f"climatology_{A.lat:.3f}_{A.lon:.3f}.json"
    if p.exists():
        return json.loads(p.read_text())["months"]
    js = get(f"https://archive-api.open-meteo.com/v1/archive?latitude={A.lat}&longitude={A.lon}&start_date=1991-01-01"
             "&end_date=2020-12-31&daily=precipitation_sum,temperature_2m_max,wind_gusts_10m_max&timezone=auto&wind_speed_unit=ms&models=era5")
    d = pd.DataFrame(js["daily"]); d["time"] = pd.to_datetime(d["time"])
    out = {}
    for m, g in d.groupby(d.time.dt.month):
        r = g.precipitation_sum
        out[str(m)] = dict(rain_mean_mm_month=float(r.groupby(g.time.dt.year).sum().mean()),
                           rain_day_p90=float(np.percentile(r, 90)), rain_day_p99=float(np.percentile(r, 99)),
                           rain_days_mean=float((r >= 1).groupby(g.time.dt.year).sum().mean()),
                           tmax_p90=float(np.percentile(g.temperature_2m_max, 90)),
                           gust_p90=float(np.nanpercentile(g.wind_gusts_10m_max, 90)))
    p.write_text(json.dumps({"source": f"Open-Meteo historical API (ERA5), {A.lat}, {A.lon}, 1991-2020", "months": out}, indent=1))
    return out


# ---------------------------------------------------------------- physics
def tmrt(ta, dni, dhi, ghi, elev, shaded, svf):
    e = np.radians(elev)
    fp = 0.308 * np.cos(e * (1 - elev ** 2 / 48402))
    S = 0.7 * (fp * dni * (~shaded) * (elev > 0) + 0.5 * dhi * svf + 0.5 * 0.2 * ghi)
    ta_k = ta + 273.15
    tg = ta_k + np.where(shaded | (elev <= 0), 2.0, 2.0 + 13.0 * np.clip(ghi / 900, 0, 1))
    L = 0.97 * (0.5 * (0.85 * SIGMA * ta_k ** 4 * svf + SIGMA * (ta_k + 2) ** 4 * (1 - svf)) + 0.5 * 0.95 * SIGMA * tg ** 4)
    return ((S + L) / (0.97 * SIGMA)) ** 0.25 - 273.15


def globe_temp(tr, ta, v, D=0.15, eps=0.95):
    k = 1.1e8 * v ** 0.6 / (eps * D ** 0.4)
    lo, hi = np.minimum(ta, tr) - 5, np.maximum(ta, tr) + 5
    for _ in range(50):
        mid = (lo + hi) / 2
        f = (mid + 273.15) ** 4 + k * (mid - ta) - (tr + 273.15) ** 4
        hi, lo = np.where(f > 0, mid, hi), np.where(f > 0, lo, mid)
    return (lo + hi) / 2


def stull_wetbulb(t, rh):
    return (t * np.arctan(0.151977 * np.sqrt(rh + 8.313659)) + np.arctan(t + rh) - np.arctan(rh - 1.676331)
            + 0.00391838 * rh ** 1.5 * np.arctan(0.023101 * rh) - 4.686035)


def derive(df):
    """Add WBGT, evaporation, margins, lightning level and per-trade status (0 go, 1 caution, 2 stop)."""
    df = df.copy()
    for c in ("precipitation_probability", "cape", "uv_index"):
        if c not in df:
            df[c] = 0.0
    df = df.fillna({"precipitation_probability": 0, "cape": 0, "uv_index": 0, "wind_gusts_10m": df.wind_speed_10m * 1.6})
    sp = pvlib.location.Location(A.lat, A.lon, altitude=A.alt).get_solarposition(df.index - pd.Timedelta(minutes=30))
    elev = np.clip(sp["apparent_elevation"].values, 0, 90)
    ta, rh = df.temperature_2m.values, df.relative_humidity_2m.values
    v = np.clip(df.wind_speed_10m.values * 0.6, 0.5, 17)                    # m/s, near the ground
    dni, dhi, ghi = (df[c].values for c in ("direct_normal_irradiance", "diffuse_radiation", "shortwave_radiation"))
    tw = stull_wetbulb(ta, rh)
    for lab, shaded, svf in (("sun", np.zeros(len(df), bool), 0.8), ("shade", np.ones(len(df), bool), 0.3)):
        tg = globe_temp(tmrt(ta, dni, dhi, ghi, elev, shaded, svf), ta, v)
        tnw = tw + 0.2 * np.maximum(tg - ta, 0)
        df[f"wbgt_{lab}"] = 0.7 * tnw + 0.2 * tg + 0.1 * ta
    c = TH["concrete"]
    tc, V = ta + c["concrete_minus_air_C"], v * 3.6
    df["evap_kg_m2_h"] = np.clip(5 * ((tc + 18) ** 2.5 - rh / 100 * (ta + 18) ** 2.5) * (V + 4) * 1e-6, 0, None)
    surf = ta - np.where(elev > 0, 0.0, np.where(df.cloud_cover.values < 50, 2.0, 1.0))
    df["dewpoint_margin"] = surf - df.dew_point_2m.values
    cr = TH["crane"]
    df["gust_hook"] = df.wind_gusts_10m * (cr["hook_height_m"] / 10) ** (1 / 7)
    lt = TH["lightning"]
    cape, prob = df.cape.values, df.precipitation_probability.values
    df["thunder"] = np.where((cape >= lt["cape_stop"]) & (prob >= lt["prob_stop_pct"]) & (df.precipitation.values >= lt["rain_stop_mm_h"]), 2,
                             np.where((cape >= lt["cape_caution"]) & (prob >= lt["prob_caution_pct"]), 1, 0))
    df["rain_prev24"] = df.precipitation.rolling(24, min_periods=1).sum()
    df["rain_prev3"] = df.precipitation.shift(1, fill_value=0).rolling(3, min_periods=1).sum()
    df["prob_next3"] = df.precipitation_probability[::-1].rolling(3, min_periods=1).max()[::-1]
    df["solar_elev"] = elev

    def lvl(stop, caution):
        return np.where(stop, 2, np.where(caution, 1, 0))
    h = TH["heat_wbgt_moderate_work_acclimatised_C"]; ws = df.wbgt_sun.values; thn = df.thunder.values
    pr, g10, gh = df.precipitation.values, df.wind_gusts_10m.values, df.gust_hook.values
    legal = TH.get("legal_wbgt_stop_C", 99)                      # e.g. Qatar: all outdoor work stops above WBGT 32.1 C
    df["st_labour"] = lvl((ws > h["work25_rest75"]) | (ws > legal), ws > h["continuous"])
    cold_s, cold_c = ta < c.get("cold_stop_C", -99), ta < c.get("cold_caution_C", -99)
    df["st_concrete"] = lvl((pr >= c["rain_stop_mm_h"]) | (df.prob_next3 >= c["rain_prob_stop_pct"]) | (df.evap_kg_m2_h >= c["evap_stop_kg_m2_h"]) | (thn == 2) | cold_s,
                            (df.prob_next3 >= c["rain_prob_caution_pct"]) | (df.evap_kg_m2_h >= c["evap_caution_kg_m2_h"]) | (thn == 1) | cold_c)
    vis = (df["visibility"].values / 1000) if "visibility" in df else np.full(len(df), np.inf)
    vis = np.where(np.isnan(vis), np.inf, vis)
    df["st_crane"] = lvl((gh >= cr["gust_stop_ms"]) | (thn == 2) | (vis < cr.get("vis_stop_km", 0)),
                         (gh >= cr["gust_caution_ms"]) | (thn == 1) | (vis < cr.get("vis_caution_km", 0)))
    wh = TH["work_at_height"]
    df["st_height"] = lvl((g10 >= wh["gust_stop_ms"]) | (pr >= wh["rain_stop_mm_h"]) | (thn == 2) | (vis < wh.get("vis_stop_km", 0)),
                          (g10 >= wh["gust_caution_ms"]) | (thn == 1) | (vis < wh.get("vis_caution_km", 0)))
    co = TH["coatings"]; m, r = df.dewpoint_margin.values, rh
    df["st_coatings"] = lvl((m < co["dewpoint_margin_stop_C"]) | (r > co["rh_stop_pct"]) | (pr > 0.1),
                            (m < co["dewpoint_margin_caution_C"]) | (r > co["rh_caution_pct"]) | (df.precipitation_probability >= 50))
    asp = TH["asphalt"]; p3 = df.rain_prev3.values
    df["st_asphalt"] = lvl((pr >= asp["rain_stop_mm_h"]) | (p3 >= asp["wet_surface_prev3h_mm"]) | (df.prob_next3 >= asp["rain_prob_next3_stop_pct"])
                           | (thn == 2) | (ta < asp["min_air_C"]),
                           (p3 > 0) | (df.prob_next3 >= asp["rain_prob_next3_caution_pct"]) | (thn == 1) | (df.wind_speed_10m.values >= asp["wind_caution_ms"]))
    ew = TH["earthworks"]; p24 = df.rain_prev24.values
    df["st_earthworks"] = lvl((p24 >= ew["rain_prev24h_stop_mm"]) | (pr >= ew["rain_stop_mm_h"]), (p24 >= ew["rain_prev24h_caution_mm"]) | (pr > 0.2))
    df["work_hour"] = df.index.hour.isin(WORK)
    return df


def work_rest(w):
    h = TH["heat_wbgt_moderate_work_acclimatised_C"]
    if w <= h["continuous"]:
        return "continuous work"
    for k, lab in (("work75_rest25", "45 min work / 15 min rest each hour"), ("work50_rest50", "30 min work / 30 min rest"),
                   ("work25_rest75", "15 min work / 45 min rest")):
        if w <= h[k]:
            return lab
    return "stop heavy and moderate work in the sun"


def runs(mask, idx):
    """Contiguous True runs -> ['07-11 h', ...] (hour labels are the start of each hour)."""
    out, start = [], None
    for k, ok in enumerate(list(mask) + [False]):
        if ok and start is not None and idx[k].hour != idx[k - 1].hour + 1:   # gap (lunch): close the run
            out.append(f"{idx[start]:%H}-{(idx[k - 1] + pd.Timedelta(hours=1)):%H} h"); start = None
        if ok and start is None:
            start = k
        if not ok and start is not None:
            out.append(f"{idx[start]:%H}-{(idx[k - 1] + pd.Timedelta(hours=1)):%H} h"); start = None
    return out


def longest(mask, idx):
    best, cur, bs = 0, 0, None
    for k, ok in enumerate(mask):
        cur = cur + 1 if ok and (cur == 0 or idx[k].hour == idx[k - 1].hour + 1) else (1 if ok else 0)
        if cur > best:
            best, bs = cur, k - cur + 1
    return (f"{idx[bs]:%H}-{(idx[bs + best - 1] + pd.Timedelta(hours=1)):%H} h", best) if best else ("none", 0)


# ---------------------------------------------------------------- forecast
fjs = get(f"https://api.open-meteo.com/v1/forecast?latitude={A.lat}&longitude={A.lon}&hourly={FC_VARS}"
          "&timezone=auto&past_days=1&forecast_days=7&wind_speed_unit=ms")
TZ = fjs["timezone"]
raw = hourly_frame(fjs)
full = derive(raw)
OBS_DROP = ["precipitation_probability", "cape", "uv_index", "visibility", "cloud_cover_low", "cloud_cover_mid", "cloud_cover_high", "pressure_msl"]   # records use what happened only, so preliminary and ERA5 days compare
obs_full = derive(raw.drop(columns=OBS_DROP))
now = pd.Timestamp.now(tz=TZ)
today = now.normalize()
fc = full[full.index >= today]
aq = aqf = None
try:
    ajs = get(f"https://air-quality-api.open-meteo.com/v1/air-quality?latitude={A.lat}&longitude={A.lon}&hourly=pm2_5,pm10,dust&timezone=auto&forecast_days=5")
    aqf = hourly_frame(ajs); aq = aqf.pm2_5
except Exception:  # noqa: BLE001
    pass
CLIM = climatology()

# ---------------------------------------------------------------- figures (scripts/meteogram_site.py)
sys.path.insert(0, str(Path(__file__).parent))
import meteogram_site  # noqa: E402

_sites = json.loads((OUT / "sites.json").read_text())["sites"] if (OUT / "sites.json").exists() else []
meteogram_site.build(fc, dict(name=A.name, short=A.name.replace("Central ", ""), lat=A.lat, lon=A.lon, alt=A.alt,
                              elev=fjs.get("elevation", A.alt), TH=TH, WORK=WORK, sites=_sites, now=now,
                              tz_abbr=fjs.get("timezone_abbreviation", "")), SITE)

# ---------------------------------------------------------------- the briefing (shared by note, card and A4 sheet)
NAMES = dict(zip(KEYS, ["Outdoor labour", "Concrete pour", "Asphalt & surfacing", "Crane lifts", "Work at height", "Painting & coatings", "Earthworks"]))
STATUS_ORDER = {"STOP": 0, "LIMITED": 1, "CAUTION": 2, "GO": 3}


def causes(k, w):
    """Why the non-go hours of trade k are not go, most frequent first."""
    bad = w[w[f"st_{k}"] >= 1]
    if not len(bad):
        return []
    c, co, asp = TH["concrete"], TH["coatings"], TH["asphalt"]
    test = {"labour": {"heat": bad.st_labour >= 1},
            "concrete": {"rain": (bad.precipitation >= c["rain_stop_mm_h"]) | (bad.prob_next3 >= c["rain_prob_caution_pct"]),
                         "storms": bad.thunder >= 1, "fast drying": bad.evap_kg_m2_h >= c["evap_caution_kg_m2_h"],
                         "cold": bad.temperature_2m < c.get("cold_caution_C", -99)},
            "asphalt": {"rain or wet surface": (bad.precipitation >= asp["rain_stop_mm_h"]) | (bad.rain_prev3 > 0) | (bad.prob_next3 >= asp["rain_prob_next3_caution_pct"]),
                        "storms": bad.thunder >= 1, "wind": bad.wind_speed_10m >= asp["wind_caution_ms"], "cold": bad.temperature_2m < asp["min_air_C"]},
            "crane": {"wind": bad.gust_hook >= TH["crane"]["gust_caution_ms"], "storms": bad.thunder >= 1,
                      "poor visibility": (bad["visibility"] / 1000 < TH["crane"].get("vis_caution_km", 0)) if "visibility" in bad else bad.thunder < -1},
            "height": {"wind": bad.wind_gusts_10m >= TH["work_at_height"]["gust_caution_ms"], "rain": bad.precipitation >= TH["work_at_height"]["rain_stop_mm_h"],
                       "storms": bad.thunder >= 1,
                       "poor visibility": (bad["visibility"] / 1000 < TH["work_at_height"].get("vis_caution_km", 0)) if "visibility" in bad else bad.thunder < -1},
            "coatings": {"humidity or dew": (bad.dewpoint_margin < co["dewpoint_margin_caution_C"]) | (bad.relative_humidity_2m > co["rh_caution_pct"]),
                         "rain": (bad.precipitation > 0.1) | (bad.precipitation_probability >= 50)},
            "earthworks": {"wet ground": bad.rain_prev24 >= TH["earthworks"]["rain_prev24h_caution_mm"], "rain": bad.precipitation > 0.2}}[k]
    n = {lab: int(m.sum()) for lab, m in test.items() if m.sum()}
    return sorted(n, key=n.get, reverse=True)


def trade_row(k, w):
    st = w[f"st_{k}"].values
    go = runs(st == 0, w.index); n = int((st == 0).sum())
    status = "GO" if n == len(w) else "CAUTION" if st.max() == 1 and n == 0 else "STOP" if n == 0 else "LIMITED"
    gw = ", ".join(go)
    why = causes(k, w)
    if k == "labour":
        hot = w[w.st_labour >= 1]
        act = ("No heat limits; keep water and shade available." if status == "GO" else
               f"Work/rest cycles {', '.join(runs(st >= 1, w.index))}: at the peak, {work_rest(hot.wbgt_sun.max())}. Water every 20 min, shade for breaks.")
        gw = gw if status != "GO" else ""
    else:
        act = {"concrete": {"LIMITED": "Curing cover and windbreaks ready." if "fast drying" in why else "Keep covers ready outside this window.",
                            "STOP": "No pour window: postpone, or pour under cover.", "CAUTION": "Pour with care: covers and curing ready."},
               "asphalt": {"LIMITED": "Never lay or spray on a wet surface.", "STOP": "No dry window: do not lay or spray.",
                           "CAUTION": "Lay with care: check the surface is dry and rain is not imminent."},
               "crane": {"LIMITED": "Check the anemometer before each lift.", "STOP": "No safe lift window: leave the crane in free slew.",
                         "CAUTION": "Lift with care: check the anemometer before each lift."},
               "height": {"LIMITED": "Clear roofs and scaffolds before storms.", "STOP": "Keep off roofs and scaffolds.",
                          "CAUTION": "Work at height with care: watch gusts and the sky."},
               "coatings": {"LIMITED": "Check the surface is 3 °C above dew point.", "STOP": "Too humid or wet for painting and coatings.",
                            "CAUTION": "Paint with care: check dew point and humidity first."},
               "earthworks": {"LIMITED": "Protect formation and stockpiles.", "STOP": "Ground too wet: protect formation, keep plant off soft ground.",
                              "CAUTION": "Ground soft: protect formation and stockpiles."}}[k].get(status, "No weather limits.")
        if k == "earthworks" and w.rain_prev24.max() >= TH["earthworks"]["rain_prev24h_caution_mm"]:
            act += f" Ground wet after {w.rain_prev24.max():.0f} mm in 24 h."
    label = {"GO": "no limits", "STOP": "STOP", "CAUTION": "go with care", "LIMITED": f"go {gw} only"}[status]
    if k == "labour" and status != "GO":
        label = "work/rest needed" if status != "STOP" else "STOP moderate work in sun"
    return dict(key=k, trade=NAMES[k], status=status, label=label, go_hours=n, windows=gw, causes=why, action=act,
                hours=[[int(h_), int(v)] for h_, v in zip(w.index.hour, st)])


def briefing(d, g):
    w = g[g.work_hour]
    rows = sorted([trade_row(k, w) for k in KEYS], key=lambda r: STATUS_ORDER[r["status"]])
    stl = runs(w.thunder.values == 2, w.index); stp = runs(w.thunder.values >= 1, w.index)
    rain = float(g.precipitation.sum()); r = {x["key"]: x for x in rows}
    pour = longest(w.st_concrete.values == 0, w.index)
    if stl:
        head = f"Storms likely from {stl[0][:2]}:00: finish lifts, roof and asphalt work before then."
    elif (w.st_labour == 2).any():
        head = f"Heat stress: stop moderate work in the sun {runs(w.st_labour.values == 2, w.index)[0]}."
    elif r["concrete"]["status"] == "STOP" and r["asphalt"]["status"] == "STOP":
        head = f"Wet day ({rain:.0f} mm): no concrete or asphalt window."
    elif stp:
        head = f"Storms possible from {stp[0][:2]}:00: do exposed work in the morning."
    elif r["concrete"]["status"] == "LIMITED":
        head = f"Pour concrete {pour[0]}; {(r['concrete']['causes'] or ['weather'])[0]} limits the rest of the day."
    elif all(x["status"] == "GO" for x in rows):
        head = "Good working day: no weather limits on any trade."
    else:
        head = "Some limits today: check your trade below."
    extras = []
    if g.uv_index.max() >= 8:
        extras.append(f"UV index up to {g.uv_index.max():.0f} (very high): hats with neck cover, long sleeves, sunscreen.")
    if aq is not None:
        a = aq[aq.index.normalize() == d]
        if len(a) and a.max() >= 35:
            extras.append(f"Dust: PM2.5 up to {a.max():.0f} µg/m³; damp down haul roads and cutting work; masks for dusty tasks.")
    if aqf is not None and "dust" in aqf:
        dd_ = aqf[aqf.index.normalize() == d]
        dw = TH.get("dust", {})
        if len(dd_) and (dd_.dust.max() >= dw.get("dust_warn_ug_m3", 1e9) or dd_.pm10.max() >= dw.get("pm10_warn_ug_m3", 1e9)):
            extras.append(f"Dust storm risk (dust up to {dd_.dust.max():.0f} µg/m³): goggles and masks, secure sheeting and loose materials, "
                          "stop lifts if the operator loses sight of the load.")
    if TH.get("legal_wbgt_stop_C") and (g.wbgt_sun > TH["legal_wbgt_stop_C"]).any():
        hrs = runs((g.wbgt_sun > TH["legal_wbgt_stop_C"]).values, g.index)
        extras.append(f"Legal heat limit: WBGT above {TH['legal_wbgt_stop_C']} °C at {', '.join(hrs)}; outdoor work must stop by law in those hours.")
    if TH.get("_seasonal_break_active"):
        extras.append("Midday outdoor-work ban in force (12:00-15:00): those hours are excluded from the working day.")
    storm = "likely" if stl else "possible" if stp else "low"
    return dict(date=f"{d:%Y-%m-%d}", day=f"{d:%A %d %B}", headline=head, rows=rows, extras=extras,
                facts=dict(rain_mm=round(rain, 1), rain_chance=int(g.precipitation_probability.max()), storm=storm,
                           storm_from=(stl or stp or [""])[0][:2], wbgt_peak=round(float(g.wbgt_sun.max()), 1),
                           gust_hook_max=round(float(g.gust_hook.max()), 1), go_hours={x["key"]: x["go_hours"] for x in rows}))


days = [(d, g) for d, g in fc.groupby(fc.index.normalize())]
if now.hour >= W1:                                          # evening run: the note starts with tomorrow
    days = days[1:]
BR = [briefing(d, g) for d, g in days]
(SITE / "site_briefing_latest.json").write_text(json.dumps(dict(site=A.name, lat=A.lat, lon=A.lon, issued=f"{now:%Y-%m-%d %H:%M}",
                                                                 working=f"{W0:02d}:00-{W1:02d}:00, lunch excluded ({NW} h)", days=BR), indent=1))
md = [f"# Site note · {A.name}", "", f"Issued {now:%A %d %B %Y, %H:%M} ({TZ}). Site {A.lat:.4f}, {A.lon:.4f}. "
      f"Working hours {W0:02d}:00-{W1:02d}:00, lunch break excluded ({NW} h).", ""]
for b in BR[:3]:
    f = b["facts"]
    md += [f"## {b['day']}", "", f"**{b['headline']}**", "",
           f"Rain {f['rain_mm']:.0f} mm (chance up to {f['rain_chance']} %) · storms {f['storm']} · WBGT peak {f['wbgt_peak']} °C · "
           f"gusts {f['gust_hook_max']:.0f} m/s at hook height", ""]
    for x in b["rows"]:
        if x["status"] != "GO":
            md.append(f"- **{x['trade']} — {x['label']}.** "
                      f"{'Limited by ' + ', '.join(x['causes']) + '. ' if x['causes'] and x['key'] != 'labour' else ''}{x['action']}")
    clear = [x["trade"] for x in b["rows"] if x["status"] == "GO"]
    if clear:
        md.append(f"- **No weather limits:** {', '.join(clear)}.")
    md += [f"- {e}" for e in b["extras"]] + [""]
md += ["## Outlook", ""]
for b in BR[3:]:
    f = b["facts"]
    md.append(f"- **{b['day']}**: {b['headline']} Rain {f['rain_mm']:.0f} mm ({f['rain_chance']} %); concrete {f['go_hours']['concrete']} h go.")
md += ["", "*Screening guidance from a weather forecast, not an instruction. The contract specification, the crane manufacturer's "
       "limits and the site health-and-safety plan take precedence; check conditions on site (WBGT meter, anemometer, "
       "the 30-30 rule for lightning) before each activity. Thresholds used: thresholds.json.*"]
(SITE / "site_note_latest.md").write_text("\n".join(md) + "\n")


def md_to_html(lines):
    out, in_ul = [], False
    for ln in lines:
        s = html.escape(ln)
        s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s); s = re.sub(r"\*(.+?)\*", r"<em>\1</em>", s)
        if ln.startswith("- "):
            if not in_ul:
                out.append("<ul>"); in_ul = True
            out.append(f"<li>{s[2:]}</li>"); continue
        if in_ul:
            out.append("</ul>"); in_ul = False
        if ln.startswith("# "):
            continue
        if ln.startswith("## "):
            out.append(f"<h3>{s[3:]}</h3>")
        elif ln:
            out.append(f"<p>{s}</p>")
    return "\n".join(out + (["</ul>"] if in_ul else []))


note_html = md_to_html(md)


def note_cards(BR):
    """The site note as a web page section: label, headline with the key time marked, fact chips, a card per trade,
    then the next days. Style: white space, Manrope, one yellow accent, tinted status pills."""
    e = html.escape
    b = BR[0]; f = b["facts"]
    head = re.sub(r"(\d{2}(?::\d{2})?-\d{2} h|from \d{2}:00|\d+ mm)", r"<mark>\1</mark>", e(b["headline"]), count=1)
    chips = [f"Rain {f['rain_mm']:.0f} mm · {f['rain_chance']}%", "Storms " + f["storm"] + (f" from {f['storm_from']}:00" if f["storm_from"] else ""),
             f"Heat (WBGT) {f['wbgt_peak']:.0f} °C", f"Gusts {f['gust_hook_max']:.0f} m/s at hook height"]
    WORDS = {"STOP": ("stop", "Stop"), "LIMITED": ("lim", "Limited"), "CAUTION": ("lim", "Go with care"), "GO": ("go", "Go")}
    cards = []
    for r in b["rows"]:
        if r["status"] == "GO":
            continue
        cls, word = WORDS[r["status"]]
        title = ("No window today" if r["status"] == "STOP" else
                 "Work/rest cycles" if r["key"] == "labour" else
                 f"Go {r['windows']}" if r["windows"] else "Go with care")
        why = ("Limited by " + ", ".join(r["causes"]) + ". ") if r["causes"] and r["key"] != "labour" else ""
        hl = r.get("hours", [])
        hrs = "".join(f"<span class='h s{v}{' gap' if i and h_ != hl[i - 1][0] + 1 else ''}' title='{h_:02d}:00'>{h_:02d}</span>" for i, (h_, v) in enumerate(hl))
        cards.append(f"<article class='tc'><div class='tc-top'><span class='eyebrow'>{e(r['trade'])}</span>"
                     f"<span class='pill {cls}'><i></i>{word}</span></div><h3>{e(title)}</h3><div class='hrs'>{hrs}</div>"
                     f"<p>{e(why + r['action'])}</p></article>")
    clear = [r["trade"] for r in b["rows"] if r["status"] == "GO"]
    if clear:
        cards.append(f"<article class='tc ok'><div class='tc-top'><span class='eyebrow'>No weather limits</span><span class='pill go'><i></i>Go</span></div>"
                     f"<h3>{e(', '.join(clear))}</h3><p>Work as planned; keep an eye on the sky.</p></article>")
    extras = "".join(f"<li>{e(x)}</li>" for x in b["extras"])
    nxt = "".join(f"<article class='step'><span class='eyebrow'>{e(d['day'])}</span><p>{e(d['headline'])}</p></article>" for d in BR[1:4])
    return (f"<div class='sn'><p class='eyebrow'>Site note · {e(b['day'])} · issued {now:%H:%M}</p><h2 class='hl'>{head}</h2>"
            f"<div class='chips'>{''.join(f'<span>{e(c)}</span>' for c in chips)}</div>"
            f"<div class='tgrid'>{''.join(cards)}</div>" + (f"<ul class='extras'>{extras}</ul>" if extras else "") +
            f"<p class='eyebrow' style='margin-top:8px'>Next days</p><div class='steps'>{nxt}</div>"
            f"<p class='fine'>Hours are go hours, {W0:02d}:00-{W1:02d}:00 with lunch excluded; 08-13 means from 08:00 until the lunch break. "
            "Screening guidance from a forecast: the contract, crane manufacturer's limits and site H&amp;S plan come first.</p></div>")
(SITE / "site_note_latest.html").write_text(note_html)
cols = ["temperature_2m", "relative_humidity_2m", "dew_point_2m", "precipitation", "precipitation_probability", "cape", "wind_speed_10m",
        "wind_gusts_10m", "gust_hook", "uv_index", "wbgt_sun", "wbgt_shade", "evap_kg_m2_h", "dewpoint_margin", "rain_prev24", "rain_prev3", "thunder"] + [f"st_{k}" for k in KEYS]
fc[cols].round(2).to_csv(SITE / "site_forecast_latest.csv")

# ---------------------------------------------------------------- archive: forecast as issued
fdir = SITE / "archive" / "forecasts" / f"{today:%Y-%m}"; fdir.mkdir(parents=True, exist_ok=True)
fc[cols][fc.index < today + pd.Timedelta(hours=72)].round(2).to_csv(fdir / f"{today:%Y-%m-%d}.csv")


# ---------------------------------------------------------------- archive: what happened
def day_record(d, g, source):
    m = CLIM[str(d.month)]
    w = g[g.work_hour]
    rain = float(g.precipitation.sum())
    flag = "exceptional (above 1-in-100-day)" if rain >= m["rain_day_p99"] else "heavy (above 1-in-10-day)" if rain >= m["rain_day_p90"] else ""
    rec = dict(date=f"{d:%Y-%m-%d}", source=source, rain_mm=round(rain, 1), rain_max_mm_h=round(float(g.precipitation.max()), 1),
               tmax_C=round(float(g.temperature_2m.max()), 1), gust_max_ms=round(float(g.wind_gusts_10m.max()), 1),
               wbgt_sun_max_C=round(float(g.wbgt_sun.max()), 1), rain_flag=flag,
               gust_flag="above 1-in-10-day" if g.wind_gusts_10m.max() >= m["gust_p90"] else "",
               heat_flag="above 1-in-10-day" if g.temperature_2m.max() >= m["tmax_p90"] else "")
    rec.update({f"go_hours_{k}": int((w[f"st_{k}"] == 0).sum()) for k in KEYS})
    rec["working_hours"] = int(len(w))
    fp = SITE / "archive" / "forecasts" / f"{d - pd.Timedelta(days=1):%Y-%m}" / f"{d - pd.Timedelta(days=1):%Y-%m-%d}.csv"
    if fp.exists():                                         # what yesterday's forecast said about this day
        f1 = pd.read_csv(fp, index_col=0, parse_dates=True)
        f1 = f1[f1.index.strftime("%Y-%m-%d") == f"{d:%Y-%m-%d}"]
        rec["forecast_rain_mm_day_before"] = round(float(f1.precipitation.sum()), 1) if len(f1) else None
    return rec


def upsert(recs):
    by_month = {}
    for r in recs:
        by_month.setdefault(r["date"][:7], []).append(r)
    for mon, rs in by_month.items():
        p = SITE / "archive" / "observed" / f"{mon}.csv"
        old = pd.read_csv(p, dtype={"date": str}) if p.exists() else pd.DataFrame(columns=["date", "source"])
        new = pd.DataFrame(rs)
        if len(old):
            keep_final = old[old.source.str.startswith("ERA5")].date
            new = new[~(new.date.isin(keep_final) & ~new.source.str.startswith("ERA5"))]
            if "forecast_rain_mm_day_before" in old:            # keep the forecast comparison once recorded
                fmap = old.set_index("date").forecast_rain_mm_day_before.dropna()
                if "forecast_rain_mm_day_before" not in new:
                    new["forecast_rain_mm_day_before"] = np.nan
                new["forecast_rain_mm_day_before"] = new.forecast_rain_mm_day_before.fillna(new.date.map(fmap))
            old = old[~old.date.isin(new.date)]
        pd.concat([old, new]).sort_values("date").to_csv(p, index=False)
    return sorted(by_month)


touched = []
yday = today - pd.Timedelta(days=1)
yg = obs_full[obs_full.index.normalize() == yday]
if len(yg) >= 20:
    touched += upsert([day_record(yday, yg, "model analysis (preliminary)")])
if A.backfill:
    end = (today - pd.Timedelta(days=8)).strftime("%Y-%m-%d")
    bjs = get(f"https://archive-api.open-meteo.com/v1/archive?latitude={A.lat}&longitude={A.lon}&start_date={A.backfill}"
              f"&end_date={end}&hourly={BASE_VARS}&timezone=auto&wind_speed_unit=ms&models=era5")
    b = derive(hourly_frame(bjs))
    touched += upsert([day_record(d, g, "ERA5 (final)") for d, g in b.groupby(b.index.normalize()) if g.temperature_2m.notna().sum() >= 20])
# replace preliminary days with ERA5 once it is published (about 5-7 days behind)
prelim = []
for p in sorted((SITE / "archive" / "observed").glob("*.csv"))[-3:]:
    o = pd.read_csv(p, dtype={"date": str})
    prelim += list(o[~o.source.str.startswith("ERA5") & (pd.to_datetime(o.date) <= (today - pd.Timedelta(days=8)).tz_localize(None))].date)
if prelim:
    try:
        ejs = get(f"https://archive-api.open-meteo.com/v1/archive?latitude={A.lat}&longitude={A.lon}&start_date={min(prelim)}"
                  f"&end_date={max(prelim)}&hourly={BASE_VARS}&timezone=auto&wind_speed_unit=ms&models=era5")
        e = hourly_frame(ejs)
        e = derive(e)
        recs = [day_record(d, g, "ERA5 (final)") for d, g in e.groupby(e.index.normalize())
                if f"{d:%Y-%m-%d}" in prelim and g.temperature_2m.notna().sum() >= 20]
        touched += upsert(recs)
    except Exception as ex:  # noqa: BLE001
        print("ERA5 finalisation skipped:", str(ex)[:120])

# ---------------------------------------------------------------- monthly records
LAB = dict(zip(KEYS, [x.split(" (")[0].split(" /")[0] for x in TRADES]))
rows = []
for p in sorted((SITE / "archive" / "observed").glob("*.csv")):
    o = pd.read_csv(p, dtype={"date": str, "rain_flag": str, "gust_flag": str, "heat_flag": str}).fillna({"rain_flag": "", "gust_flag": "", "heat_flag": ""})
    mon = p.stem; m = CLIM[str(int(mon[5:]))]
    nd = len(o); final = int(o.source.str.startswith("ERA5").sum())
    lost = {k: int((o.working_hours - o[f"go_hours_{k}"]).sum()) for k in KEYS}
    rain = o.rain_mm.sum()
    days_in_month = pd.Period(mon).days_in_month
    rows.append(dict(month=mon, days_recorded=nd, days_final=final, rain_mm=round(rain, 1),
                     rain_normal_mm_month=round(m["rain_mean_mm_month"], 0),
                     rain_pct_of_normal_pro_rata=round(100 * rain / (m["rain_mean_mm_month"] * nd / days_in_month), 0)
                     if nd and m["rain_mean_mm_month"] * nd / days_in_month >= 1 else None,
                     rain_days=int((o.rain_mm >= 1).sum()), rain_days_normal=round(m["rain_days_mean"] * nd / days_in_month, 1),
                     heavy_rain_days=int(o.rain_flag.str.startswith("heavy").sum()), exceptional_rain_days=int(o.rain_flag.str.startswith("exceptional").sum()),
                     high_gust_days=int((o.gust_flag != "").sum()), hot_days=int((o.heat_flag != "").sum()),
                     **{f"lost_hours_{k}": v for k, v in lost.items()}))
    if mon in touched or not (SITE / "archive" / "monthly" / f"{mon}.md").exists():
        r = rows[-1]
        tbl = ["| date | source | rain mm | max mm/h | Tmax °C | gust m/s | WBGT sun °C | flags | " + " | ".join(f"go h {LAB[k]}" for k in KEYS) + " |",
               "|" + "---|" * (8 + len(KEYS))]
        for _, x in o.iterrows():
            fl = "; ".join(f for f in (x.rain_flag and f"rain {x.rain_flag}", x.gust_flag and f"gust {x.gust_flag}", x.heat_flag and f"heat {x.heat_flag}") if f)
            tbl.append(f"| {x.date} | {x.source} | {x.rain_mm} | {x.rain_max_mm_h} | {x.tmax_C} | {x.gust_max_ms} | {x.wbgt_sun_max_C} | {fl} | "
                       + " | ".join(str(int(x[f'go_hours_{k}'])) for k in KEYS) + " |")
        txt = [f"# Site weather record · {A.name} · {pd.Period(mon):%B %Y}", "",
               f"Site {A.lat:.4f}, {A.lon:.4f}. {nd} of {days_in_month} days recorded, {final} finalised from ERA5 (the rest are "
               "preliminary model analysis and are replaced automatically once ERA5 is published).", "",
               f"- Rain {r['rain_mm']} mm against a pro-rata normal of {m['rain_mean_mm_month'] * nd / days_in_month:.0f} mm "
               + ("(" + (f"{r['rain_pct_of_normal_pro_rata']:.0f} % of normal; " if r['rain_pct_of_normal_pro_rata'] is not None else "normal under 1 mm, so no percentage; ")
                + f"full-month normal {m['rain_mean_mm_month']:.0f} mm, 1991-2020)."),
               f"- Rain days (1 mm or more): {r['rain_days']} against a pro-rata normal of {r['rain_days_normal']}.",
               f"- Heavy rain days (above the site's 1-in-10-day amount, {m['rain_day_p90']:.0f} mm): {r['heavy_rain_days']}; "
               f"exceptional (above 1-in-100-day, {m['rain_day_p99']:.0f} mm): {r['exceptional_rain_days']}.",
               f"- Days with gusts above the 1-in-10-day value ({m['gust_p90']:.0f} m/s): {r['high_gust_days']}; days hotter than the "
               f"1-in-10-day maximum ({m['tmax_p90']:.1f} °C): {r['hot_days']}.",
               "- Working hours lost to weather (caution or stop): " + ", ".join(f"{LAB[k].lower()} {v} h" for k, v in lost.items()) + ".",
               "", *tbl, "",
               "*How to use this record: under FIDIC 2017 Sub-Clause 8.5 (1999 Sub-Clause 8.4) a contractor may claim time for "
               "exceptionally adverse climatic conditions. This record shows how the month compared with the site's own 1991-2020 "
               "climate. It is a weather record for the site location, not a site measurement; a site rain gauge and daily diary "
               "remain the primary evidence, and the contract's own definition of 'exceptional' governs.*"]
        (SITE / "archive" / "monthly" / f"{mon}.md").write_text("\n".join(txt) + "\n")
idx = pd.DataFrame(rows)
idx.to_csv(SITE / "archive" / "monthly_index.csv", index=False)

# ---------------------------------------------------------------- page
def cost_line():
    """Construction input costs for this country from CiO (scripts/32_market_link.py saves them), with a link to the brief."""
    f = OUT / "market" / "costs.json"
    if not f.exists():
        return ""
    k = "_".join("".join(ch if ch.isalnum() else " " for ch in COUNTRY.lower().replace("ô", "o")).split())
    c = json.loads(f.read_text()).get("countries", {}).get(k)
    if not c:
        return ""
    fx = (f"; the {c['currency']} {'weakened' if c['fx_13w_pct'] > 0 else 'strengthened'} {abs(c['fx_13w_pct']):.1f}% in 13 weeks"
          if c.get("fx_13w_pct") is not None else "")
    return (f"<section><p class='eyebrow'>Costs in {html.escape(COUNTRY)}</p><p class='lede'><strong>{html.escape(c['series'])}: "
            f"{c['near']['median']:+.1f}% by {html.escape(c['near']['period'])}</strong> (range {c['near']['lo']:+.1f}% to {c['near']['hi']:+.1f}%, "
            f"base {html.escape(c['base'])}){fx}. From the CiO Construction Cost Platform. Weather sets the time, costs set the money: "
            f"<a href='../market/#{k}'>this week's site and market brief</a>.</p></section>")


months_html = "".join(f"<tr><td><a href='archive/monthly/{r['month']}.md'>{r['month']}</a></td><td>{r['days_recorded']}</td>"
                      f"<td>{r['rain_mm']}</td><td>{'–' if r['rain_pct_of_normal_pro_rata'] is None else str(int(r['rain_pct_of_normal_pro_rata'])) + ' %'}</td><td>{r['heavy_rain_days']}</td>"
                      f"<td>{r['exceptional_rain_days']}</td><td>{r['lost_hours_concrete']}</td><td>{r['lost_hours_labour']}</td>"
                      f"<td><a href='archive/observed/{r['month']}.csv'>CSV</a></td></tr>"
                      for r in reversed(rows)) or "<tr><td colspan='9'>The record starts tomorrow.</td></tr>"
page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Site Weather · {html.escape(A.name)}</title>
<meta name="description" content="{html.escape(BR[0]['headline'])} Hour-by-hour go windows for seven trades, the week ahead and the monthly weather record. By Gava Charles.">
<meta property="og:type" content="website"><meta property="og:site_name" content="Kampala Heat Atlas · Site weather">
<meta property="og:title" content="Site weather · {html.escape(A.name)} · {html.escape(BR[0]['day'])}">
<meta property="og:description" content="{html.escape(BR[0]['headline'])} Go windows for concrete, asphalt, cranes, work at height, painting, earthworks and heat.">
<meta property="og:url" content="https://gavacharles.github.io/kampala-heat-atlas/site-weather/{SLUG}/">
<meta property="og:image" content="https://gavacharles.github.io/kampala-heat-atlas/site-weather/{SLUG}/social/x_16x9.png?v={now:%Y%m%d}">
<meta property="og:image:width" content="1600"><meta property="og:image:height" content="900">
<meta property="og:image:alt" content="{html.escape(BR[0]['headline'])} Site weather card for {html.escape(A.name)}.">
<meta name="twitter:card" content="summary_large_image"><meta name="twitter:title" content="Site weather · {html.escape(A.name)}">
<meta name="twitter:description" content="{html.escape(BR[0]['headline'])}">
<meta name="twitter:image" content="https://gavacharles.github.io/kampala-heat-atlas/site-weather/{SLUG}/social/x_16x9.png?v={now:%Y%m%d}">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap">
<style>
:root{{--bg:#ffffff;--surface:#fafaf9;--ink:#0a0a0b;--ink-2:#3e3e42;--line:#e1e1e0;--accent:#fbca0c;--eyebrow:#755200;--tint:#fdf6dc;--link:#2a6fc4;
--go-bg:#e8f4eb;--go:#1f7a35;--lim-bg:#fdf3d7;--lim:#8a6100;--stop-bg:#f9eaea;--stop:#b42318;--s0:#2f8f3a;--s1:#e0a526;--s2:#c23434}}
@media (prefers-color-scheme: dark){{:root:not([data-theme="light"]){{--bg:#0d0d0e;--surface:#17171a;--ink:#f2f2ef;--ink-2:#a8a8a4;--line:#2a2a2e;--eyebrow:#e3b341;
--tint:#2a2410;--link:#7fb0ef;--go-bg:#11281a;--go:#7fd394;--lim-bg:#2b2410;--lim:#f0c75e;--stop-bg:#2e1414;--stop:#f08a80;color-scheme:dark}}}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.6 Manrope,Arial,sans-serif;padding:28px 16px 48px}}
main{{max-width:1100px;margin:0 auto;display:grid;grid-template-columns:minmax(0,1fr);gap:28px}}
h1{{font:800 clamp(30px,5vw,44px)/1.05 Manrope,Arial,sans-serif;letter-spacing:-.02em;margin:0}}
h2{{font:800 24px/1.15 Manrope,Arial,sans-serif;letter-spacing:-.01em;margin:0}} a{{color:var(--link)}}
.lede{{color:var(--ink-2);max-width:72ch;margin:0}} nav{{display:flex;gap:8px;flex-wrap:wrap}}
nav a{{color:var(--ink);text-decoration:none;border:1px solid var(--line);background:var(--bg);padding:7px 14px;border-radius:999px;font-size:13px;font-weight:600}}
nav a:hover{{background:var(--accent);border-color:var(--accent);color:#0a0a0b}}
section{{display:grid;gap:12px;border-top:1px solid var(--line);padding-top:24px}}
.eyebrow{{font-size:12px;font-weight:700;letter-spacing:.14em;text-transform:uppercase;color:var(--eyebrow);margin:0}}
.sn{{display:grid;gap:16px}} .hl{{font:800 clamp(26px,4.2vw,40px)/1.12 Manrope,Arial,sans-serif;letter-spacing:-.02em;max-width:24ch}}
mark{{background:var(--accent);color:#0a0a0b;padding:0 .18em;border-radius:6px}}
.chips{{display:flex;gap:8px;flex-wrap:wrap}} .chips span{{border:1px solid var(--line);background:var(--surface);border-radius:999px;padding:6px 12px;font-size:13px;font-weight:600;color:var(--ink-2)}}
.tgrid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(250px,1fr));gap:12px}}
.tc{{background:var(--surface);border:1px solid var(--line);border-radius:16px;padding:18px;display:grid;gap:10px;align-content:start}}
.tc h3{{margin:0;font-size:19px;font-weight:800;letter-spacing:-.01em}} .tc p{{margin:0;color:var(--ink-2);font-size:14px}}
.tc-top{{display:flex;flex-direction:column;gap:8px;align-items:flex-start}}
.pill{{display:inline-flex;align-items:center;gap:6px;border-radius:999px;padding:4px 10px;font-size:12px;font-weight:800}}
.pill i{{width:7px;height:7px;border-radius:50%;background:currentColor}}
.pill.stop{{background:var(--stop-bg);color:var(--stop)}} .pill.lim{{background:var(--lim-bg);color:var(--lim)}} .pill.go{{background:var(--go-bg);color:var(--go)}}
.hrs{{display:flex;gap:3px}} .h.gap{{margin-left:7px}} .h{{flex:1 1 0;min-width:0;text-align:center;border-radius:7px;padding:3px 0;font-size:12px;font-weight:800;color:#fff}}
.h.s0{{background:var(--s0)}} .h.s1{{background:var(--s1)}} .h.s2{{background:var(--s2)}}
.extras{{margin:0;padding-left:18px;color:var(--ink-2)}} .steps{{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:12px}}
.step{{border:1px solid var(--line);border-radius:16px;padding:16px;display:grid;gap:6px;background:var(--bg)}} .step p{{margin:0;font-weight:600}}
.fine{{font-size:12px;color:var(--ink-2);margin:0}}
.wrap{{overflow-x:auto}} table{{border-collapse:collapse;font-size:13px;min-width:640px}}
th,td{{border-bottom:1px solid var(--line);padding:8px 10px;text-align:right}} th:first-child,td:first-child{{text-align:left}}
img{{width:100%;height:auto;border:1px solid var(--line);border-radius:16px;background:#fff}}
</style></head><body><main>
<nav><a href="../../index.html">Atlas</a><a href="../../3d.html">3D model</a><a href="../../stories.html">Stories</a><a href="../../daynight.html">Day vs night</a><a href="../../meteogram.html">Meteogram</a><a href="../">Site weather</a><a href="../map.html">Map</a><a href="../share.html">Share today</a><a href="../../gallery.html">Figures</a><a href="https://github.com/gavacharles/kampala-heat-atlas">Code</a></nav>
<h1>Site weather · {html.escape(A.name)}</h1>
<p class="lede">For contractors and site managers: the next seven days as workable hours per trade, with a short site note,
refreshed every morning at 06:00. Every day is also archived, so the record of what the weather did, month by month and
compared with the site's 1991-2020 climate, builds up on its own.</p>
<section>{note_cards(BR)}</section>
{cost_line()}
<section><h2>Seven-day site meteogram</h2>
<img src="site_meteogram_latest.png" alt="Seven-day site meteogram: heat, rain and storms, wind, sun and cloud, concrete and painting, go hours per trade, daily summary and regional maps">
<p class="lede">Trade panels, the same forecast with only what each trade needs:
<a href="site_meteogram_concrete.png">concrete, asphalt, painting and earthworks</a> ·
<a href="site_meteogram_lifting.png">cranes and work at height</a> · <a href="site_meteogram_heat.png">heat and outdoor labour</a></p>
<p class="lede"><a href="site_forecast_latest.csv">Hourly numbers and per-trade status (CSV)</a> · <a href="../thresholds.json">Thresholds used</a> ·
<a href="archive/forecasts/">Forecasts as issued</a></p></section>
<section><h2>Cards to share</h2>
<p class="lede">The same forecast as images for WhatsApp, LinkedIn, X, Instagram and Facebook, refreshed each morning.
<a href="social/today_4x5.png">Today's briefing</a> · <a href="social/today_9x16.png">Today (Status / Stories)</a> ·
<a href="social/today_hours_4x5.png">Hour-by-hour grid</a> · <a href="social/briefing_A4.pdf"><strong>Printable A4 briefing (PDF)</strong></a> ·
<a href="social/week_4x5.png">The week</a> · <a href="social/carousel.pdf">LinkedIn carousel (PDF)</a> · <a href="social/caption.txt">Caption</a></p>
<div style="display:flex;gap:12px;overflow-x:auto"><img src="social/today_4x5.png" alt="Today's site card" style="width:300px;flex:none">
<img src="social/week_4x5.png" alt="The week on site card" style="width:300px;flex:none"><img src="social/howto_4x5.png" alt="How to read the card" style="width:300px;flex:none"></div></section>
<section><h2>Monthly record</h2>
<p class="lede">Rain against the 1991-2020 normal, heavy and exceptional days, and working hours lost. Days start as preliminary
model analysis and are replaced by ERA5 about a week later. Useful as supporting evidence for extension-of-time claims
(FIDIC 2017 Sub-Clause 8.5); a site rain gauge and the daily diary remain the primary record.</p>
<div class="wrap"><table><tr><th>month</th><th>days</th><th>rain mm</th><th>of normal</th><th>heavy days</th><th>exceptional days</th>
<th>concrete h lost</th><th>labour (heat) h lost</th><th>daily</th></tr>{months_html}</table></div></section>
<p class="lede" style="font-size:13px">Screening guidance from a weather forecast, not an instruction. The contract specification,
crane manufacturer's limits and site health-and-safety plan take precedence. Forecast and ERA5 data: Open-Meteo (open-meteo.com, CC BY 4.0).</p>
<p style="margin:0;font-weight:800">By Gava Charles · <span style="font-weight:600;color:var(--ink-2)">Kampala Heat Atlas</span></p>
</main></body></html>"""
(SITE / "index.html").write_text(page)
sites = sorted(p.parent.name for p in OUT.glob("*/index.html"))
(OUT / "index.html").write_text("<!doctype html><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'>"
                                "<title>Site Weather</title><meta http-equiv='refresh' content='0; url=" + (sites[0] if len(sites) == 1 else SLUG) + "/'>"
                                "<body style='font-family:Arial;padding:24px'>" + "".join(f"<p><a href='{s}/'>{s}</a></p>" for s in sites))
print("wrote", SITE, "| days:", [f"{d:%a}" for d, _ in days], "| months recorded:", [r["month"] for r in rows])
