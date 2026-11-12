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
W0, W1 = TH["working_hours"]
WORK = [h for h in range(W0, W1) if h not in TH.get("break_hours", [])]     # e.g. 8 working hours, lunch excluded
NW = len(WORK)
SIGMA = 5.67e-8
TRADES = ["Outdoor labour (heat)", "Concrete pour", "Asphalt & surfacing", "Crane lifts", "Work at height / roofing", "Painting & coatings", "Earthworks"]
KEYS = ["labour", "concrete", "asphalt", "crane", "height", "coatings", "earthworks"]
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
BLUE, ORANGE, RED, GREEN, AMBER, OFF = "#2a78d6", "#eb6834", "#c23434", "#2f8f3a", "#e0a526", "#ecebe6"
plt.rcParams.update({"font.family": ["Helvetica Neue", "Arial", "DejaVu Sans"], "font.size": 9, "axes.edgecolor": "#c3c2b7",
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.labelcolor": INK2})
BASE_VARS = ("temperature_2m,relative_humidity_2m,dew_point_2m,precipitation,cloud_cover,wind_speed_10m,wind_direction_10m,"
             "wind_gusts_10m,shortwave_radiation,direct_normal_irradiance,diffuse_radiation")
FC_VARS = BASE_VARS + ",precipitation_probability,cape,uv_index"


def get(url, tries=4):
    for k in range(tries):
        try:
            return json.loads(urllib.request.urlopen(url, timeout=90).read())
        except Exception:  # noqa: BLE001
            if k == tries - 1:
                raise
            time.sleep(5 * (k + 1))


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
    df["st_labour"] = lvl(ws > h["work25_rest75"], ws > h["continuous"])
    df["st_concrete"] = lvl((pr >= c["rain_stop_mm_h"]) | (df.prob_next3 >= c["rain_prob_stop_pct"]) | (df.evap_kg_m2_h >= c["evap_stop_kg_m2_h"]) | (thn == 2),
                            (df.prob_next3 >= c["rain_prob_caution_pct"]) | (df.evap_kg_m2_h >= c["evap_caution_kg_m2_h"]) | (thn == 1))
    df["st_crane"] = lvl((gh >= cr["gust_stop_ms"]) | (thn == 2), (gh >= cr["gust_caution_ms"]) | (thn == 1))
    wh = TH["work_at_height"]
    df["st_height"] = lvl((g10 >= wh["gust_stop_ms"]) | (pr >= wh["rain_stop_mm_h"]) | (thn == 2), (g10 >= wh["gust_caution_ms"]) | (thn == 1))
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
OBS_DROP = ["precipitation_probability", "cape", "uv_index"]   # records use what happened only, so preliminary and ERA5 days compare
obs_full = derive(raw.drop(columns=OBS_DROP))
now = pd.Timestamp.now(tz=TZ)
today = now.normalize()
fc = full[full.index >= today]
aq = None
try:
    ajs = get(f"https://air-quality-api.open-meteo.com/v1/air-quality?latitude={A.lat}&longitude={A.lon}&hourly=pm2_5&timezone=auto&forecast_days=5")
    aq = hourly_frame(ajs).pm2_5
except Exception:  # noqa: BLE001
    pass
CLIM = climatology()

# ---------------------------------------------------------------- figure
t = fc.index
fig, ax = plt.subplots(6, 1, figsize=(11.5, 14), sharex=True, constrained_layout=True,
                       gridspec_kw=dict(height_ratios=[2.6, 1.2, 1.3, 1.1, 1.1, 1.8]))
h = TH["heat_wbgt_moderate_work_acclimatised_C"]
bands = [(-50, h["continuous"], "#ffffff", "continuous work"), (h["continuous"], h["work75_rest25"], "#fde0d2", "45/15 work/rest"),
         (h["work75_rest25"], h["work50_rest50"], "#f8c4ad", "30/30"), (h["work50_rest50"], h["work25_rest75"], "#f19b80", "15/45"),
         (h["work25_rest75"], 60, "#e06a5f", "stop moderate work")]
ylo, yhi = min(15, np.nanmin(fc.temperature_2m) - 1), max(33, np.nanmax(fc.wbgt_sun) + 1)
for lo, hi, col, lab in bands:
    ax[0].axhspan(lo, hi, color=col, alpha=0.55, lw=0)
    if lo > ylo and lo < yhi:
        ax[0].text(t[-1], min((lo + hi) / 2, yhi - 0.6), f" {lab}", fontsize=7.5, color=INK2, va="center")
ax[0].plot(t, fc.wbgt_sun, color=RED, lw=2, label="WBGT in the sun")
ax[0].plot(t, fc.wbgt_shade, color=GREEN, lw=2, label="WBGT in shade")
ax[0].plot(t, fc.temperature_2m, color=INK2, lw=1.2, ls="--", label="air temperature")
ax[0].set_ylim(ylo, yhi); ax[0].set_ylabel("°C")
ax[0].legend(loc="upper left", ncol=3, frameon=False, fontsize=8)
ax[0].set_title(f"Site weather · {A.name} ({A.lat:.3f}, {A.lon:.3f}) · issued {now:%a %d %b %Y %H:%M}",
                loc="left", fontsize=12, fontweight="bold", color=INK)
ax[0].text(0.005, 0.04, "work/rest bands: moderate work, acclimatised workers; heavier work or new workers need lower limits",
           transform=ax[0].transAxes, fontsize=7.5, color=INK2)
ax[1].bar(t, fc.precipitation, width=0.035, color=BLUE)
rmax = max(5, fc.precipitation.max() * 1.25)
ax[1].set_ylim(0, rmax); ax[1].set_ylabel("rain mm/h")
for lv, col in ((1, AMBER), (2, RED)):
    k = fc.thunder.values == lv
    ax[1].scatter(t[k], np.full(k.sum(), rmax * 0.92), marker="v", s=22, color=col, zorder=3)
daily = fc.precipitation.resample("D").sum(); pmax = fc.precipitation_probability.resample("D").max()
ax[1].set_title("Rain · triangles: thunderstorm risk (amber possible, red likely) · daily total (chance): " +
                "  ".join(f"{d:%a} {v:.0f} mm ({p:.0f}%)" for (d, v), p in zip(daily.items(), pmax.values)),
                loc="left", fontsize=8.5, color=INK2)
cr = TH["crane"]
ax[2].plot(t, fc.wind_speed_10m, color=MUTED, lw=1.2, label="mean wind, 10 m")
ax[2].plot(t, fc.gust_hook, color=INK, lw=1.6, label=f"gust at {cr['hook_height_m']} m hook height")
for y, col, lab in ((cr["gust_caution_ms"], AMBER, "crane caution"), (cr["gust_stop_ms"], RED, "crane stop")):
    ax[2].axhline(y, color=col, lw=1, ls="--"); ax[2].text(t[0], y + 0.3, f" {lab} {y:.0f} m/s", fontsize=7.5, color=INK2)
ax[2].set_ylim(0, max(cr["gust_stop_ms"] + 3, fc.gust_hook.max() + 2)); ax[2].set_ylabel("m/s")
ax[2].legend(loc="upper right", ncol=2, frameon=False, fontsize=8)
c = TH["concrete"]
ax[3].fill_between(t, fc.evap_kg_m2_h, color="#f8c4ad", lw=0); ax[3].plot(t, fc.evap_kg_m2_h, color=ORANGE, lw=1.4)
for y, col, lab in ((c["evap_caution_kg_m2_h"], AMBER, "protect fresh concrete"), (c["evap_stop_kg_m2_h"], RED, "plastic-cracking risk (ACI 305)")):
    ax[3].axhline(y, color=col, lw=1, ls="--"); ax[3].text(t[0], y + 0.03, f" {lab}", fontsize=7.5, color=INK2)
ax[3].set_ylim(0, max(1.3, fc.evap_kg_m2_h.max() * 1.15)); ax[3].set_ylabel("evaporation\nkg/m²/h")
co = TH["coatings"]
ax[4].plot(t, fc.dewpoint_margin, color=BLUE, lw=1.4)
ax[4].axhline(co["dewpoint_margin_stop_C"], color=RED, lw=1, ls="--")
ax[4].text(t[0], co["dewpoint_margin_stop_C"] + 0.3, " surface must be 3 °C above dew point (ISO 8502-4)", fontsize=7.5, color=INK2)
wet = fc.relative_humidity_2m.values > co["rh_stop_pct"]
ax[4].fill_between(t, 0, 1, where=wet, transform=ax[4].get_xaxis_transform(), color=BLUE, alpha=0.08, lw=0)
ax[4].set_ylim(0, max(12, fc.dewpoint_margin.max() + 1)); ax[4].set_ylabel("surface − dew\npoint °C")
ax[4].text(0.005, 0.82, f"shaded: relative humidity above {co['rh_stop_pct']} %", transform=ax[4].transAxes, fontsize=7.5, color=INK2)
grid = np.array([np.where(fc.work_hour, fc[f"st_{k}"].values + 1, 0) for k in KEYS])
edges = mdates.date2num(list(t) + [t[-1] + pd.Timedelta(hours=1)])
ax[5].pcolormesh(edges, np.arange(len(KEYS) + 1), grid, cmap=ListedColormap([OFF, GREEN, AMBER, RED]), vmin=0, vmax=3,
                 edgecolors="#ffffff", linewidth=0.3)
ax[5].set_yticks(np.arange(len(KEYS)) + 0.5); ax[5].set_yticklabels(TRADES, fontsize=8); ax[5].invert_yaxis(); ax[5].grid(False)
ax[5].set_title(f"Workable hours {W0:02d}-{W1:02d} h · green go · amber caution · red stop · grey outside working hours",
                loc="left", fontsize=8.5, color=INK2)
ax[5].xaxis.set_major_locator(mdates.DayLocator(tz=t.tz)); ax[5].xaxis.set_major_formatter(mdates.DateFormatter("%a %d %b", tz=t.tz))
for a in ax:
    for d in pd.date_range(today, periods=8, freq="D"):
        a.axvline(d, color="#c3c2b7", lw=0.6)
fig.text(0.01, -0.012, "Screening guidance from the Open-Meteo forecast (open-meteo.com). Thresholds: docs/site-weather/thresholds.json; "
         "the contract specification, crane manufacturer's limits and site H&S plan take precedence. Kampala Heat Atlas · Gava Charles",
         fontsize=7, color=MUTED)
fig.savefig(SITE / "site_meteogram_latest.png", dpi=130, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- the site note
def day_lines(d, g):
    w = g[g.work_hour]
    rain, pm, uv = g.precipitation.sum(), g.precipitation_probability.max(), g.uv_index.max()
    storm = runs(g.thunder.values >= 1, g.index)
    head = (f"rain {rain:.0f} mm (chance up to {pm:.0f} %)" + (f", thunderstorm risk {', '.join(storm)}" if storm else "") +
            f"; gusts to {g.gust_hook.max():.0f} m/s at hook height; WBGT in sun peaks {g.wbgt_sun.max():.1f} °C at {g.wbgt_sun.idxmax():%H}:00")
    out = []
    hot = w[w.st_labour >= 1]
    if len(hot):
        worst = hot.wbgt_sun.max()
        out.append(f"Heat: {', '.join(runs(w.st_labour.values >= 1, w.index))} need a work/rest cycle; at the peak, {work_rest(worst)}. "
                   "Water at least every 20 min, shade for breaks, heavy tasks early, lighter duties for workers new to site.")
    else:
        out.append("Heat: continuous moderate work is fine through the day; keep water and shade available.")
    win, n = longest(w.st_concrete.values == 0, w.index)
    ev = w.evap_kg_m2_h.max()
    cn = f"Concrete: best pour window {win} ({n} h)" if n else "Concrete: no clear pour window in working hours"
    if ev >= TH["concrete"]["evap_caution_kg_m2_h"]:
        cn += f"; evaporation up to {ev:.1f} kg/m²/h, so have curing compound, wet hessian or windbreaks ready"
    out.append(cn + ".")
    aw, an = longest(w.st_asphalt.values == 0, w.index)
    out.append(f"Asphalt & surfacing: dry window {aw} ({an} h); no laying or spraying on a wet surface." if an
               else "Asphalt & surfacing: no dry window; do not lay or spray on wet surfaces or with rain imminent.")
    cr_bad = runs(w.st_crane.values >= 1, w.index)
    out.append("Cranes: " + (f"check wind and storms before lifts at {', '.join(cr_bad)}." if cr_bad else "no gust limits expected in working hours."))
    ht = runs(w.st_height.values >= 1, w.index)
    out.append("Work at height: " + (f"caution or stop at {', '.join(ht)} (rain, gusts or storms)." if ht else "no weather limits expected."))
    ok = runs(w.st_coatings.values == 0, w.index)
    out.append("Coatings: " + (f"suitable {', '.join(ok)}." if ok else "no suitable window (humidity, dew point or rain)."))
    ew = w.rain_prev24.max()
    out.append("Earthworks: " + (f"up to {ew:.0f} mm in the previous 24 h, so expect soft, wet ground; protect formation and stockpiles."
                                 if ew >= TH["earthworks"]["rain_prev24h_caution_mm"] else "ground conditions not limited by recent rain."))
    if uv >= 8:
        out.append(f"UV index up to {uv:.0f} (very high): hats with neck cover, long sleeves, sunscreen.")
    if aq is not None:
        a = aq[aq.index.normalize() == d]
        if len(a) and a.max() >= 35:
            out.append(f"Dust/air quality: PM2.5 up to {a.max():.0f} µg/m³; damp down haul roads and cutting work; masks for dusty tasks.")
    hours = {k: int((w[f'st_{k}'] == 0).sum()) for k in KEYS}
    out.append("Workable hours (go): " + ", ".join(f"{lab.split(' (')[0].lower()} {hours[k]}" for lab, k in zip(TRADES, KEYS)) + f" of {len(w)}.")
    return head, out


days = [(d, g) for d, g in fc.groupby(fc.index.normalize())]
if now.hour >= W1:                                          # evening run: the note starts with tomorrow
    days = days[1:]
md = [f"# Site note · {A.name}", "", f"Issued {now:%A %d %B %Y, %H:%M} ({TZ}). Site {A.lat:.4f}, {A.lon:.4f}. Working hours {W0:02d}:00-{W1:02d}:00, lunch break excluded ({NW} h).", ""]
for d, g in days[:3]:
    head, lines = day_lines(d, g)
    md += [f"## {d:%A %d %B}", "", f"**{head[0].upper() + head[1:]}.**", ""] + [f"- {x}" for x in lines] + [""]
md += ["## Outlook", ""]
for d, g in days[3:]:
    w = g[g.work_hour]; win, n = longest(w.st_concrete.values == 0, w.index)
    md.append(f"- **{d:%a %d %b}**: rain {g.precipitation.sum():.0f} mm ({g.precipitation_probability.max():.0f} %)"
              f"{', thunderstorm risk' if (g.thunder >= 1).any() else ''}; WBGT peak {g.wbgt_sun.max():.1f} °C; "
              f"concrete window {win if n else 'none'}.")
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
                     rain_pct_of_normal_pro_rata=round(100 * rain / (m["rain_mean_mm_month"] * nd / days_in_month), 0) if nd else None,
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
               f"({r['rain_pct_of_normal_pro_rata']:.0f} % of normal; full-month normal {m['rain_mean_mm_month']:.0f} mm, 1991-2020).",
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
months_html = "".join(f"<tr><td><a href='archive/monthly/{r['month']}.md'>{r['month']}</a></td><td>{r['days_recorded']}</td>"
                      f"<td>{r['rain_mm']}</td><td>{r['rain_pct_of_normal_pro_rata']:.0f} %</td><td>{r['heavy_rain_days']}</td>"
                      f"<td>{r['exceptional_rain_days']}</td><td>{r['lost_hours_concrete']}</td><td>{r['lost_hours_labour']}</td>"
                      f"<td><a href='archive/observed/{r['month']}.csv'>CSV</a></td></tr>"
                      for r in reversed(rows)) or "<tr><td colspan='9'>The record starts tomorrow.</td></tr>"
page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Site Weather · {html.escape(A.name)}</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wght@800&family=Inter:wght@400;500&display=swap">
<style>
:root{{--bg:#f6f6f3;--surface:#fff;--ink:#0b0b0b;--ink-2:#52514e;--line:#e1e0d9;--blue:#2a78d6}}
@media (prefers-color-scheme: dark){{:root:not([data-theme="light"]){{--bg:#111312;--surface:#1a1c1b;--ink:#ecedea;--ink-2:#b6b9b2;--line:#2d302e;--blue:#6da7ec;color-scheme:dark}}}}
body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 Inter,Arial,sans-serif;padding:24px 16px}}
main{{max-width:1100px;margin:0 auto;display:grid;gap:20px}} h1{{font:800 34px/1.05 Archivo,Arial,sans-serif;margin:0}}
h2{{font:800 24px/1.1 Archivo,Arial,sans-serif;margin:0}} h3{{margin:14px 0 4px;font-size:17px}} a{{color:var(--blue)}}
.lede{{color:var(--ink-2);max-width:72ch;margin:0}} nav{{display:flex;gap:8px;flex-wrap:wrap}}
nav a{{color:var(--ink);text-decoration:none;border:1px solid var(--line);background:var(--surface);padding:6px 12px;border-radius:999px;font-size:13px}}
section{{display:grid;gap:10px;border-top:1px solid var(--line);padding-top:16px}} .note{{background:var(--surface);border:1px solid var(--line);border-radius:8px;padding:8px 20px}}
.note li{{margin:3px 0}} .wrap{{overflow-x:auto}} table{{border-collapse:collapse;font-size:13px;min-width:640px}}
th,td{{border-bottom:1px solid var(--line);padding:6px 10px;text-align:right}} th:first-child,td:first-child{{text-align:left}}
img{{width:100%;height:auto;border:1px solid var(--line);border-radius:8px;background:#fff}}
</style></head><body><main>
<nav><a href="../../index.html">Atlas</a><a href="../../3d.html">3D model</a><a href="../../stories.html">Stories</a><a href="../../daynight.html">Day vs night</a><a href="../../meteogram.html">Meteogram</a><a href="../">Site weather</a><a href="../../gallery.html">Figures</a><a href="https://github.com/gavacharles/kampala-heat-atlas">Code</a></nav>
<h1>Site weather · {html.escape(A.name)}</h1>
<p class="lede">For contractors and site managers: the next seven days as workable hours per trade, with a short site note,
refreshed every morning at 06:00. Every day is also archived, so the record of what the weather did, month by month and
compared with the site's 1991-2020 climate, builds up on its own.</p>
<section><h2>Today's site note</h2><div class="note">{note_html}</div></section>
<section><h2>Seven-day site meteogram</h2>
<img src="site_meteogram_latest.png" alt="Seven-day site meteogram: heat stress, rain and storms, wind at crane height, concrete evaporation, coating conditions, workable hours per trade">
<p class="lede"><a href="site_forecast_latest.csv">Hourly numbers and per-trade status (CSV)</a> · <a href="../thresholds.json">Thresholds used</a> ·
<a href="archive/forecasts/">Forecasts as issued</a></p></section>
<section><h2>Cards to share</h2>
<p class="lede">The same forecast as images for WhatsApp, LinkedIn, X, Instagram and Facebook, refreshed each morning.
<a href="social/today_4x5.png">Today</a> · <a href="social/today_9x16.png">Today (Status / Stories)</a> ·
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
</main></body></html>"""
(SITE / "index.html").write_text(page)
sites = sorted(p.parent.name for p in OUT.glob("*/index.html"))
(OUT / "index.html").write_text("<!doctype html><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'>"
                                "<title>Site Weather</title><meta http-equiv='refresh' content='0; url=" + (sites[0] if len(sites) == 1 else SLUG) + "/'>"
                                "<body style='font-family:Arial;padding:24px'>" + "".join(f"<p><a href='{s}/'>{s}</a></p>" for s in sites))
print("wrote", SITE, "| days:", [f"{d:%a}" for d, _ in days], "| months recorded:", [r["month"] for r in rows])
