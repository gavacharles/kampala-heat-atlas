"""Site meteogram builder used by scripts/29_site_weather.py.

    build(fc, ctx) -> writes site_meteogram_latest.png (everything) and three trade panels:
        site_meteogram_concrete.png   concrete, asphalt, painting, earthworks
        site_meteogram_lifting.png    cranes and work at height
        site_meteogram_heat.png       outdoor labour

Sections (each a chart and/or rows of 3-hourly values, sharing one time axis):
  heat     WBGT in sun and shade, air temperature, ECMWF ensemble 10-90 % range of temperature, work/rest
           bands, daily high (WBGT) and low (air) labels; rows: air, WBGT sun, RH, UV
  rain     hourly rain, storm markers, running total; rows: 3-h rain, chance, ensemble chance of >= 1 mm, storms
  wind     mean 10 m wind and gust at hook height (band between = gust gain), crane limits; rows: direction
           arrows, mean, gust
  sun      sunshine on the ground against clear sky (Haurwitz); rows: sunshine % of clear sky, cloud total /
           low / mid / high, visibility
  concrete evaporation from fresh concrete; coating dew-point margin with RH > 85 % shaded; rows: evap, margin, RH
  trades   go strip per trade with the go windows written in each day
  daily    one column per day: rain (ensemble range), chance, storms, peak WBGT, max gust, min visibility,
           sunshine, go windows per trade
  maps     daily rain and maximum gust on a 7 x 7 grid, about 90 x 90 km around the site
Header: site, coordinates, elevation, issue time, forecast and ensemble model with run time. Footer: methods,
forecast hour row.
"""
import json
import urllib.request

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pvlib
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.patches import Rectangle
from pathlib import Path
from matplotlib import font_manager

for _f in (Path(__file__).parent / "fonts").glob("Manrope-*.ttf"):          # same typeface as the cards
    font_manager.fontManager.addfont(str(_f))

INK, INK2, MUTED, GRID = "#0b0b0b", "#3a3a3e", "#55555a", "#e1e0d9"
BLUE, ORANGE, RED, GREEN, AMBER, OFF = "#2a78d6", "#eb6834", "#c23434", "#2f8f3a", "#e0a526", "#ecebe6"
EYEBROW = "#755200"
STATUS = [GREEN, AMBER, RED]
TINT = {0: ("#e8f4eb", "#1f7a35"), 1: ("#fdf3d7", "#755200"), 2: ("#f9eaea", "#b42318")}
LONG = {"labour": "Outdoor labour (heat)", "concrete": "Concrete pour", "asphalt": "Asphalt & surfacing", "crane": "Crane lifts",
        "height": "Work at height / roofing", "coatings": "Painting & coatings", "earthworks": "Earthworks"}
EXTRA_TOWNS = [("Entebbe", 0.0512, 32.4637), ("Mukono", 0.3533, 32.7553), ("Wakiso", 0.4044, 32.4594), ("Masaka", -0.3338, 31.7341),
               ("Fort Portal", 0.6710, 30.2750), ("Kasese", 0.1833, 30.0833), ("Soroti", 1.7146, 33.6111), ("Lira", 2.2499, 32.8999),
               ("Tororo", 0.6928, 34.1809), ("Kabale", -1.2486, 29.9899), ("Masindi", 1.6744, 31.7150), ("Iganga", 0.6092, 33.4686),
               ("Kitgum", 3.2783, 32.8867), ("Nebbi", 2.4783, 31.0889), ("Ibanda", -0.1342, 30.4950), ("Busia", 0.4669, 34.0900)]
plt.rcParams.update({"font.family": ["Manrope ExtraLight", "Helvetica Neue", "Arial", "DejaVu Sans"], "font.size": 9, "axes.edgecolor": "#c3c2b7",
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.labelcolor": INK2})


def _get(url):
    for k in range(4):
        try:
            return json.loads(urllib.request.urlopen(url, timeout=90).read())
        except Exception:  # noqa: BLE001
            if k == 3:
                raise
            import time
            time.sleep(8 * (k + 1))


def fetch_extras(ctx):
    """Ensemble (ECMWF IFS 0.25 ENS, 51 members), model run time, and the regional grid. Each part is optional."""
    lat, lon = ctx["lat"], ctx["lon"]
    ex = {}
    try:
        js = _get(f"https://ensemble-api.open-meteo.com/v1/ensemble?latitude={lat}&longitude={lon}"
                  "&hourly=temperature_2m,precipitation&models=ecmwf_ifs025&forecast_days=7&timezone=auto")
        h = pd.DataFrame(js["hourly"]); h["time"] = pd.to_datetime(h["time"]).dt.tz_localize(js["timezone"]); h = h.set_index("time")
        t = h[[c for c in h if c.startswith("temperature_2m_member")] + ["temperature_2m"]]
        p = h[[c for c in h if c.startswith("precipitation_member")] + ["precipitation"]]
        ex["ens"] = pd.DataFrame({"t10": t.quantile(0.1, axis=1), "t90": t.quantile(0.9, axis=1)})
        ex["ens_p"] = p                                                  # hourly rain for every member
        ex["ens_n"] = p.shape[1]
    except Exception:  # noqa: BLE001
        pass
    try:
        for model, key in (("ecmwf_ifs025", "init_ens"),):
            m = _get(f"https://api.open-meteo.com/data/{model}/static/meta.json")
            ex[key] = pd.Timestamp(m["last_run_initialisation_time"], unit="s", tz="UTC")
    except Exception:  # noqa: BLE001
        pass
    try:
        g = np.linspace(-0.4, 0.4, 7)
        lats = [round(lat + a, 4) for a in g for _ in g]; lons = [round(lon + b, 4) for _ in g for b in g]
        js = _get("https://api.open-meteo.com/v1/forecast?latitude=" + ",".join(map(str, lats)) + "&longitude=" + ",".join(map(str, lons)) +
                  "&daily=precipitation_sum,wind_gusts_10m_max&timezone=auto&forecast_days=7&wind_speed_unit=ms")
        rain = np.array([j["daily"]["precipitation_sum"] for j in js], float).reshape(7, 7, -1)
        gust = np.array([j["daily"]["wind_gusts_10m_max"] for j in js], float).reshape(7, 7, -1)
        ex["grid"] = dict(rain=rain, gust=gust, days=pd.to_datetime(js[0]["daily"]["time"]), lat=g + lat, lon=g + lon)
    except Exception:  # noqa: BLE001
        pass
    return ex


# ---------------------------------------------------------------- helpers
def blocks(t):
    """3-hourly blocks over the forecast: list of (start, end)."""
    start = t[0].normalize(); end = t[-1] + pd.Timedelta(hours=1)
    edges = pd.date_range(start, end + pd.Timedelta(hours=3), freq="3h")
    return [(a, b) for a, b in zip(edges[:-1], edges[1:]) if b > t[0] and a <= t[-1]]


def agg(s, bl, how):
    out = []
    for a, b in bl:
        v = s[(s.index >= a) & (s.index < b)]
        out.append(np.nan if v.dropna().empty else getattr(v, how)())
    return np.array(out, float)


def lum(rgb):
    r, g, b = [c / 255 if c > 1 else c for c in rgb[:3]]
    return 0.299 * r + 0.587 * g + 0.114 * b


def cell_rows(ax, bl, rows, xlim):
    """Coloured value cells, one row per spec: (label, values, fmt, colour function)."""
    ax.set_xlim(*xlim); ax.set_ylim(len(rows), 0); ax.grid(False)
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(False)
    ax.set_yticks(np.arange(len(rows)) + 0.5); ax.set_yticklabels([r[0] for r in rows], fontsize=7.6, color=INK2)
    ax.tick_params(axis="y", length=0); ax.tick_params(axis="x", length=0, labelbottom=False)
    for i, (lab, vals, fmt, colf) in enumerate(rows):
        for (a, b), v in zip(bl, vals):
            x0, x1 = mdates.date2num(a), mdates.date2num(b)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                ax.add_patch(Rectangle((x0, i), x1 - x0, 1, facecolor="#f6f6f3", edgecolor="white", lw=0.6)); continue
            fc = colf(v)
            ax.add_patch(Rectangle((x0, i), x1 - x0, 1, facecolor=fc, edgecolor="white", lw=0.6))
            txt = fmt(v) if callable(fmt) else fmt.format(v)
            rgb = matplotlib.colors.to_rgb(fc)
            ax.text((x0 + x1) / 2, i + 0.55, txt, ha="center", va="center", fontsize=6.6, fontweight="bold",
                    color="white" if lum(rgb) < 0.5 else INK)


def cmap_fn(colors, lo, hi):
    cm = LinearSegmentedColormap.from_list("c", colors)
    return lambda v: cm(min(max((v - lo) / (hi - lo), 0), 1))


def steps_fn(edges, colors):
    return lambda v: colors[int(np.searchsorted(edges, v, side="right"))]


def day_lines(ax, days, ymin=0, ymax=1):
    for d in days[1:]:
        ax.axvline(d, color="#b9b8ae", lw=0.7, zorder=0)


def runs_txt(st, idx):
    out, s0, prev = [], None, None
    for v, hh in zip(list(st) + [1], list(idx.hour) + [None]):
        if s0 is not None and (v != 0 or hh != prev + 1):
            out.append(f"{s0:02d}-{prev + 1:02d}"); s0 = None
        if v == 0 and hh is not None and s0 is None:
            s0 = hh
        prev = hh
    return out


# ---------------------------------------------------------------- the figure
def build(fc, ctx, out_dir):
    ex = fetch_extras(ctx)
    t = fc.index; TH = ctx["TH"]; WORK = ctx["WORK"]; W0, W1 = TH["working_hours"]
    bl = blocks(t)
    xlim = (mdates.date2num(t[0]), mdates.date2num(t[-1] + pd.Timedelta(hours=1)))
    days = list(pd.date_range(t[0].normalize(), t[-1].normalize(), freq="D"))
    loc = pvlib.location.Location(ctx["lat"], ctx["lon"], tz=str(t.tz), altitude=ctx["alt"])
    clear = loc.get_clearsky(t - pd.Timedelta(minutes=30), model="haurwitz")["ghi"].values
    fc = fc.copy(); fc["clear_ghi"] = clear
    for c in ("visibility", "cloud_cover_low", "cloud_cover_mid", "cloud_cover_high"):
        if c not in fc:
            fc[c] = np.nan
    ens = ex.get("ens")
    # ensemble chance of >= 1 mm in each 3-h block
    ens_chance = None
    if "ens_p" in ex:
        p = ex["ens_p"]
        ens_chance = np.array([((p[(p.index >= a) & (p.index < b)].sum() >= 1).mean() * 100) if len(p[(p.index >= a) & (p.index < b)]) else np.nan
                               for a, b in bl])
    # values per 3-h block
    V = dict(ta=agg(fc.temperature_2m, bl, "mean"), wb=agg(fc.wbgt_sun, bl, "max"), rh=agg(fc.relative_humidity_2m, bl, "mean"),
             uv=agg(fc.uv_index, bl, "max"), rain=agg(fc.precipitation, bl, "sum"), prob=agg(fc.precipitation_probability, bl, "max"),
             storm=agg(fc.thunder.astype(float), bl, "max"), wind=agg(fc.wind_speed_10m, bl, "mean"), gust=agg(fc.gust_hook, bl, "max"),
             dir=agg(fc.wind_direction_10m, bl, "median"), sun=agg(fc.shortwave_radiation, bl, "sum") / np.maximum(agg(fc.clear_ghi, bl, "sum"), 1) * 100,
             cl=agg(fc.cloud_cover, bl, "mean"), cll=agg(fc.cloud_cover_low, bl, "mean"), clm=agg(fc.cloud_cover_mid, bl, "mean"),
             clh=agg(fc.cloud_cover_high, bl, "mean"), vis=agg(fc.visibility / 1000, bl, "min"), ev=agg(fc.evap_kg_m2_h, bl, "max"),
             dm=agg(fc.dewpoint_margin, bl, "min") + 0.0)
    sun_ok = agg(fc.clear_ghi, bl, "sum") > 50
    V["sun"] = np.where(sun_ok, np.clip(V["sun"], 0, 100), np.nan)
    h = TH["heat_wbgt_moderate_work_acclimatised_C"]; cr = TH["crane"]; c = TH["concrete"]; co = TH["coatings"]
    temp_c = cmap_fn(["#2c7bb6", "#abd9e9", "#ffffbf", "#fdae61", "#d7191c"], 12, 34)
    wb_c = steps_fn([h["continuous"], h["work75_rest25"], h["work50_rest50"], h["work25_rest75"]],
                    ["#e8f4eb", "#fde0d2", "#f8c4ad", "#f19b80", "#c23434"])
    blue_c = cmap_fn(["#ffffff", "#c6dbef", "#6baed6", "#2171b5", "#08306b"], 0, 100)
    rain_c = steps_fn([0.1, 1, 3, 8, 15], ["#ffffff", "#deebf7", "#9ecae1", "#4292c6", "#2171b5", "#08306b"])
    grey_c = cmap_fn(["#ffffff", "#d9d9d9", "#969696", "#525252"], 0, 100)
    gust_c = steps_fn([cr["gust_caution_ms"] * 0.6, cr["gust_caution_ms"], cr["gust_stop_ms"]], ["#ffffff", "#fff1c1", "#f6b34a", "#c23434"])
    wind_c = cmap_fn(["#ffffff", "#d4e8d4", "#74b874", "#2f8f3a"], 0, 8)
    vis_c = steps_fn([1, 5, 10], ["#c23434", "#f6b34a", "#fff1c1", "#ffffff"])
    ev_c = steps_fn([c["evap_caution_kg_m2_h"], c["evap_stop_kg_m2_h"]], ["#ffffff", "#f6b34a", "#c23434"])
    dm_c = steps_fn([co["dewpoint_margin_stop_C"], co["dewpoint_margin_caution_C"]], ["#c23434", "#f6b34a", "#ffffff"])
    uv_c = steps_fn([3, 6, 8, 11], ["#ffffff", "#fff1c1", "#f6b34a", "#e06a3a", "#8e44ad"])
    sun_c = cmap_fn(["#f2f2f2", "#fff1c1", "#fbca0c"], 0, 100)
    storm_c = steps_fn([0.5, 1.5], ["#ffffff", "#f6b34a", "#c23434"])

    def daily_vals():
        out = {}
        for dd in days:
            g = fc[fc.index.normalize() == dd]; w = g[g.index.hour.isin(WORK)]
            e = None
            if "ens_p" in ex:
                pe = ex["ens_p"]; pe = pe[pe.index.normalize() == dd].sum()
                e = (pe.quantile(0.1), pe.quantile(0.9))
            out[dd] = dict(g=g, w=w, rain=g.precipitation.sum(), ens=e, prob=g.precipitation_probability.max(),
                           storm=int(g.thunder.max()), wb=g.wbgt_sun.max(), gust=g.gust_hook.max(), vis=(w.visibility.min() / 1000) if len(w) else np.nan,
                           sun=(g.shortwave_radiation.sum() / max(g.clear_ghi.sum(), 1) * 100))
        return out
    DV = daily_vals()

    # ------------------------------------------------ section painters
    def s_daystrip(ax):
        ax.set_xlim(*xlim); ax.set_ylim(0, 2); ax.axis("off")
        for dd in days:
            a = max(mdates.date2num(dd), xlim[0]); b = min(mdates.date2num(dd + pd.Timedelta(days=1)), xlim[1])
            ax.add_patch(Rectangle((a, 1), b - a, 1, facecolor="#f6f6f3", edgecolor="white", lw=1.2))
            if b - a > 0.3:
                ax.text((a + b) / 2, 1.5, f"{dd:%A %d %b}".upper(), ha="center", va="center", fontsize=8, fontweight="bold", color=EYEBROW)
        for a, b in bl:
            ax.text(mdates.date2num(a) + 0.0625, 0.45, f"{a:%H}", ha="center", va="center", fontsize=6.6, color=MUTED)
        ax.text(xlim[0] - 0.02 * (xlim[1] - xlim[0]), 0.45, f"local ({ctx['tz_abbr']})", ha="right", va="center", fontsize=7, color=INK2)

    def s_heat(ax):
        for lo, hi, col in ((h["continuous"], h["work75_rest25"], "#fde0d2"), (h["work75_rest25"], h["work50_rest50"], "#f8c4ad"),
                            (h["work50_rest50"], h["work25_rest75"], "#f19b80"), (h["work25_rest75"], 60, "#e06a5f")):
            ax.axhspan(lo, hi, color=col, alpha=0.45, lw=0)
        if ens is not None:
            e = ens.reindex(t)
            ax.fill_between(t, e.t10, e.t90, color="#9aa3ad", alpha=0.28, lw=0, label="air temperature, ensemble 10-90 %")
        ax.plot(t, fc.temperature_2m, color=INK2, lw=1.2, ls="--", label="air temperature")
        ax.plot(t, fc.wbgt_shade, color=GREEN, lw=1.8, label="WBGT in shade")
        ax.plot(t, fc.wbgt_sun, color=RED, lw=2.1, label="WBGT in sun")
        lo_ = min(14, np.nanmin(fc.temperature_2m) - 1); hi_ = max(h["work25_rest75"] + 2, np.nanmax(fc.wbgt_sun) + 2.5)
        ax.set_ylim(lo_, hi_); ax.set_ylabel("°C")
        for dd in days:
            g = fc[fc.index.normalize() == dd]
            if len(g) < 6:
                continue
            i, j = g.wbgt_sun.idxmax(), g.temperature_2m.idxmin()
            ax.annotate(f"H {g.wbgt_sun.max():.0f}", (i, g.wbgt_sun.max()), xytext=(0, 5), textcoords="offset points", ha="center",
                        fontsize=8, fontweight="bold", color=RED)
            ax.annotate(f"L {g.temperature_2m.min():.0f}", (j, g.temperature_2m.min()), xytext=(0, -11), textcoords="offset points",
                        ha="center", fontsize=8, fontweight="bold", color=BLUE)
        ax.legend(loc="upper left", ncol=4, frameon=False, fontsize=7.5)
        ax.set_title("Heat: bands = work/rest for moderate work · H = day's peak WBGT in sun · L = night's lowest air temperature",
                     loc="left", fontsize=8, color=INK2)

    def r_heat(ax):
        cell_rows(ax, bl, [("Air °C", V["ta"], "{:.0f}", temp_c), ("WBGT sun °C", V["wb"], "{:.0f}", wb_c),
                           ("Humidity %", V["rh"], "{:.0f}", blue_c), ("UV index", V["uv"], "{:.0f}", uv_c)], xlim)

    def s_rain(ax):
        ax.bar(t, fc.precipitation, width=0.038, color=BLUE, align="edge")
        rmax = max(5, fc.precipitation.max() * 1.3)
        ax.set_ylim(0, rmax); ax.set_ylabel("rain mm/h")
        for lv, col in ((1, AMBER), (2, RED)):
            k = fc.thunder.values == lv
            ax.scatter(t[k], np.full(k.sum(), rmax * 0.93), marker="v", s=20, color=col, zorder=3)
        ax2 = ax.inset_axes([0, 0, 1, 1], sharex=ax); ax2.patch.set_alpha(0); ax2.grid(False)
        acc = fc.precipitation.cumsum()
        ax2.plot(t, acc, color="#08306b", lw=1.3); ax2.set_ylim(0, max(10, acc.max() * 1.15))
        ax2.axis("off")
        ax2.text(t[-2], acc.iloc[-1] * 0.97, f"{acc.iloc[-1]:.0f} mm in 7 days ", fontsize=7.5, color="#08306b", va="top", ha="right", fontweight="bold")
        ax.set_title("Rain: bars hourly mm · triangles storm risk (amber possible, red likely) · dark line running total",
                     loc="left", fontsize=8, color=INK2)

    def r_rain(ax):
        rows = [("Rain 3 h mm", V["rain"], lambda v: "" if v < 0.05 else f"{v:.1f}" if v < 10 else f"{v:.0f}", rain_c),
                ("Chance %", V["prob"], "{:.0f}", blue_c)]
        if ens_chance is not None:
            rows.append((f"Ens ≥1 mm %", ens_chance, "{:.0f}", blue_c))
        rows.append(("Storms", V["storm"], lambda v: ["", "poss", "likely"][int(v)], storm_c))
        cell_rows(ax, bl, rows, xlim)

    def s_wind(ax):
        ax.fill_between(t, fc.wind_speed_10m, fc.gust_hook, color="#9fc5e8", alpha=0.35, lw=0, label="gust gain (mean → gust at hook)")
        ax.plot(t, fc.wind_speed_10m, color=MUTED, lw=1.2, label="mean wind 10 m")
        ax.plot(t, fc.gust_hook, color=INK, lw=1.6, label=f"gust at {cr['hook_height_m']} m hook height")
        for y, col, lab in ((cr["gust_caution_ms"], AMBER, "crane caution"), (cr["gust_stop_ms"], RED, "crane stop")):
            ax.axhline(y, color=col, lw=1, ls="--"); ax.text(t[-3], y + 0.3, f"{lab} {y:.0f} m/s", fontsize=7, color=INK2, va="bottom", ha="right")
        ax.set_ylim(0, max(cr["gust_stop_ms"] + 3, fc.gust_hook.max() + 2)); ax.set_ylabel("m/s")
        ax.legend(loc="upper left", ncol=3, frameon=False, fontsize=7.5)

    def r_wind(ax):
        cell_rows(ax, bl, [("Direction", np.full(len(bl), np.nan), "", lambda v: "#fff"), ("Mean m/s", V["wind"], "{:.0f}", wind_c),
                           ("Gust hook m/s", V["gust"], "{:.0f}", gust_c)], xlim)
        for (a, b), d in zip(bl, V["dir"]):
            if np.isnan(d):
                continue
            x = mdates.date2num(a) + 0.0625; to = np.radians(d + 180)
            ax.annotate("", xy=(x, 0.5), xytext=(-7 * np.sin(to), 7 * np.cos(to)), textcoords="offset points",
                        arrowprops=dict(arrowstyle="<|-", color=BLUE if 135 <= d <= 225 else INK2, lw=1.1, mutation_scale=7))

    def s_sun(ax):
        ax.bar(t, fc.shortwave_radiation, width=0.038, color="#fbca0c", align="edge", label="sunshine on the ground")
        ax.plot(t + pd.Timedelta(minutes=30), fc.clear_ghi, color=ORANGE, lw=1, ls="--", label="clear sky")
        ax.set_ylim(0, max(1100, np.nanmax(fc.clear_ghi) * 1.1)); ax.set_ylabel("W/m²")
        ax.legend(loc="upper left", ncol=2, frameon=False, fontsize=7.5)

    def r_sun(ax):
        cell_rows(ax, bl, [("Sun % of clear", V["sun"], "{:.0f}", sun_c), ("Cloud total %", V["cl"], "{:.0f}", grey_c),
                           ("Low %", V["cll"], "{:.0f}", grey_c), ("Middle %", V["clm"], "{:.0f}", grey_c), ("High %", V["clh"], "{:.0f}", grey_c),
                           ("Visibility km", V["vis"], lambda v: f"{v:.0f}" if v >= 10 else f"{v:.1f}", vis_c)], xlim)

    def s_evap(ax):
        ax.fill_between(t, fc.evap_kg_m2_h, color="#f8c4ad", lw=0); ax.plot(t, fc.evap_kg_m2_h, color=ORANGE, lw=1.4)
        for y, col, lab in ((c["evap_caution_kg_m2_h"], AMBER, "protect fresh concrete"), (c["evap_stop_kg_m2_h"], RED, "cracking risk (ACI 305)")):
            ax.axhline(y, color=col, lw=1, ls="--"); ax.text(t[-3], y + 0.02, lab, fontsize=7, color=INK2, va="bottom", ha="right")
        ax.set_ylim(0, max(1.3, fc.evap_kg_m2_h.max() * 1.15)); ax.set_ylabel("evaporation\nkg/m²/h")

    def s_dew(ax):
        wet = fc.relative_humidity_2m.values > co["rh_stop_pct"]
        ax.fill_between(t, 0, 1, where=wet, transform=ax.get_xaxis_transform(), color=BLUE, alpha=0.08, lw=0)
        ax.plot(t, fc.dewpoint_margin, color=BLUE, lw=1.4)
        ax.axhline(co["dewpoint_margin_stop_C"], color=RED, lw=1, ls="--")
        ax.text(t[-30], co["dewpoint_margin_stop_C"] + 0.3, "3 °C minimum (ISO 8502-4)", fontsize=7, color=INK2, va="bottom")
        ax.set_ylim(0, max(12, fc.dewpoint_margin.max() + 1)); ax.set_ylabel("surface − dew\npoint °C")
        ax.set_title(f"Painting: surface temperature above dew point · shaded: humidity above {co['rh_stop_pct']} %", loc="left", fontsize=8, color=INK2)

    def r_conc(ax):
        cell_rows(ax, bl, [("Evap kg/m²/h", V["ev"], "{:.1f}", ev_c), ("Dew margin °C", V["dm"], lambda v: f"{round(v):.0f}".replace("-0", "0"), dm_c),
                           ("Humidity %", V["rh"], "{:.0f}", blue_c)], xlim)

    def s_trades(ax, keys):
        grid = np.array([np.where(fc.work_hour, fc[f"st_{k}"].values + 1, 0) for k in keys])
        edges = mdates.date2num(list(t) + [t[-1] + pd.Timedelta(hours=1)])
        ax.pcolormesh(edges, np.arange(len(keys) + 1), grid, cmap=ListedColormap([OFF, GREEN, AMBER, RED]), vmin=0, vmax=3,
                      edgecolors="#ffffff", linewidth=0.3)
        ax.set_yticks(np.arange(len(keys)) + 0.5); ax.set_yticklabels([LONG[k] for k in keys], fontsize=7.8); ax.invert_yaxis(); ax.grid(False)
        ax.set_xlim(*xlim)
        for dd in days:
            wd = fc[(fc.index.normalize() == dd) & fc.work_hour]
            if not len(wd):
                continue
            xc = mdates.date2num(dd + pd.Timedelta(hours=(W0 + W1) / 2))
            for r_i, k in enumerate(keys):
                st = wd[f"st_{k}"].values
                lab = "all day" if (st == 0).all() else ", ".join(runs_txt(st, wd.index)) or "none"
                ax.text(xc, r_i + 0.5, lab, ha="center", va="center", fontsize=6.4, color=INK, fontweight="bold" if lab != "none" else "normal",
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
        ax.set_title(f"Go hours per trade ({W0:02d}-{W1:02d} h, lunch excluded) · green go · amber care · red stop · grey not working",
                     loc="left", fontsize=8.5, color=INK2)

    def s_daily(ax, keys):
        rows = [("Rain mm (ens 10-90 %)", "rain"), ("Chance max %", "prob"), ("Storms", "storm"), ("Peak WBGT sun °C", "wb"),
                ("Max gust hook m/s", "gust"), ("Min visibility, work hours km", "vis"), ("Sunshine % of clear", "sun")] + [(LONG[k], k) for k in keys]
        ax.set_xlim(*xlim); ax.set_ylim(len(rows), 0); ax.grid(False)
        for s in ("top", "right", "left", "bottom"):
            ax.spines[s].set_visible(False)
        ax.set_yticks(np.arange(len(rows)) + 0.5); ax.set_yticklabels([r[0] for r in rows], fontsize=7.6, color=INK2)
        ax.tick_params(length=0, labelbottom=False)
        for dd in days:
            x0 = max(mdates.date2num(dd), xlim[0]); x1 = min(mdates.date2num(dd + pd.Timedelta(days=1)), xlim[1])
            dv = DV[dd]
            for i, (lab, key) in enumerate(rows):
                if key == "rain":
                    txt = f"{dv['rain']:.0f}" + (f"  ({dv['ens'][0]:.0f}-{dv['ens'][1]:.0f})" if dv["ens"] else ""); col = rain_c(dv["rain"] / 3)
                elif key == "prob":
                    txt, col = f"{dv['prob']:.0f}", blue_c(dv["prob"])
                elif key == "storm":
                    txt, col = ["low", "possible", "likely"][dv["storm"]], storm_c(dv["storm"])
                elif key == "wb":
                    txt, col = f"{dv['wb']:.1f}", wb_c(dv["wb"])
                elif key == "gust":
                    txt, col = f"{dv['gust']:.0f}", gust_c(dv["gust"])
                elif key == "vis":
                    v = dv["vis"]; txt, col = ("–", "#f6f6f3") if np.isnan(v) else (f"{v:.1f}", vis_c(v))
                elif key == "sun":
                    txt, col = f"{dv['sun']:.0f}", sun_c(dv["sun"])
                else:
                    w = dv["w"]
                    if not len(w):
                        txt, col = "", "#f6f6f3"
                    else:
                        st = w[f"st_{key}"].values; share = (st == 0).mean()
                        win = "all day" if (st == 0).all() else ", ".join(runs_txt(st, w.index)) or "none"
                        txt = win; bg, fg = TINT[0 if share >= 0.66 else 1 if share >= 0.33 else 2]; col = bg
                ax.add_patch(Rectangle((x0, i), x1 - x0, 1, facecolor=col, edgecolor="white", lw=1))
                tc = INK if lum(matplotlib.colors.to_rgb(col)) >= 0.5 else "white"
                if key in LONG:
                    tc = TINT[0 if share >= 0.66 else 1 if share >= 0.33 else 2][1] if len(dv["w"]) else INK
                ax.text((x0 + x1) / 2, i + 0.55, txt, ha="center", va="center", fontsize=7.4, fontweight="bold", color=tc)
        ax.set_title("Each day at a glance", loc="left", fontsize=8.5, color=INK2)

    def s_maps(fig, sub):
        G = ex.get("grid")
        if not G:
            return
        mdays = [d for d in G["days"] if pd.Timestamp(d).tz_localize(str(t.tz)) >= days[0]][:7]
        towns = [(s["name"].replace("Central ", ""), s["lat"], s["lon"]) for s in ctx["sites"]] + EXTRA_TOWNS
        ext = (G["lon"][0] - 0.067, G["lon"][-1] + 0.067, G["lat"][0] - 0.067, G["lat"][-1] + 0.067)
        for row, (key, cm, vmax, unit) in enumerate((("rain", "Blues", max(10, np.nanmax(G["rain"])), "mm"),
                                                    ("gust", "Oranges", max(cr["gust_stop_ms"], np.nanmax(G["gust"]) * 1.4), "m/s"))):
            for j, dd in enumerate(mdays):
                ax = fig.add_subplot(sub[row, j]); k = list(G["days"]).index(dd)
                arr = G[key][:, :, k] * (1.17 if key == "gust" else 1)        # gust scaled to hook height, as on the chart
                ax.imshow(arr, origin="lower", extent=ext, cmap=cm, vmin=0, vmax=vmax, interpolation="bicubic", aspect="auto")
                for name, la, lo in towns:
                    if ext[0] < lo < ext[1] and ext[2] < la < ext[3] and name != ctx["short"] and \
                            abs(lo - ctx["lon"]) + abs(la - ctx["lat"]) > 0.18:
                        ax.plot(lo, la, "o", ms=2.5, color=INK); ax.text(lo, la + 0.04, name, fontsize=6, ha="center", color=INK)
                ax.plot(ctx["lon"], ctx["lat"], "o", ms=5, mfc="white", mec=INK)
                ci = arr[3, 3]
                ax.text(ctx["lon"], ctx["lat"] + 0.07, f"{ci:.0f} {unit}", ha="center", va="bottom", fontsize=8, fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=INK, lw=0.6))
                ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
                for s in ax.spines.values():
                    s.set_visible(True); s.set_color("#c3c2b7")
                ax.set_title(f"{pd.Timestamp(dd):%a %d %b}".upper(), fontsize=7.5, color=EYEBROW, fontweight="bold", loc="left")
                if j == 0:
                    ax.set_ylabel("daily rain mm" if key == "rain" else f"max gust at\n{cr['hook_height_m']} m, m/s", fontsize=8, color=INK2)

    # ------------------------------------------------ assemble
    def draw(path, sections, keys, title_extra):
        hmap = {"day": 0.55, "heat": 2.4, "heat_r": 1.2, "rain": 1.6, "rain_r": 1.25, "wind": 1.5, "wind_r": 0.95, "sun": 1.2,
                "sun_r": 1.75, "evap": 1.0, "dew": 1.0, "conc_r": 0.9, "trades": 0.3 * len(keys) + 0.35, "daily": 0.3 * (7 + len(keys)) + 0.3,
                "maps": 3.6 if ex.get("grid") else 0}
        secs = [s for s in sections if hmap[s] > 0]
        heights = [hmap[s] for s in secs]
        fig = plt.figure(figsize=(17, sum(heights) + 1.6))
        gs = fig.add_gridspec(len(secs), 1, height_ratios=heights, hspace=0.32, left=0.085, right=0.955, top=1 - 0.95 / (sum(heights) + 1.6),
                              bottom=0.85 / (sum(heights) + 1.6))
        axes = {}
        for i, s in enumerate(secs):
            if s == "maps":
                axes[s] = gs[i].subgridspec(2, 7, wspace=0.06, hspace=0.25)
                continue
            ax = fig.add_subplot(gs[i]); axes[s] = ax
            ax.set_xlim(*xlim)
            ax.xaxis.set_major_locator(mdates.DayLocator(tz=t.tz)); ax.tick_params(axis="x", labelbottom=False, length=0)
        P = {"day": s_daystrip, "heat": s_heat, "heat_r": r_heat, "rain": s_rain, "rain_r": r_rain, "wind": s_wind, "wind_r": r_wind,
             "sun": s_sun, "sun_r": r_sun, "evap": s_evap, "dew": s_dew, "conc_r": r_conc}
        for s in secs:
            if s in P:
                P[s](axes[s])
            elif s == "trades":
                s_trades(axes[s], keys)
            elif s == "daily":
                s_daily(axes[s], keys)
            elif s == "maps":
                s_maps(fig, axes[s])
            if s not in ("day", "maps"):
                day_lines(axes[s], days)
        # header and run stamp
        H = sum(heights) + 1.6
        fig.text(0.085, 1 - 0.28 / H, f"Site weather · {ctx['name']}{title_extra}", fontsize=15, fontweight="bold", color=INK, va="top")
        stamp = f"{ctx['lat']:.4f}°, {ctx['lon']:.4f}° · {ctx['elev']:.0f} m · issued {ctx['now']:%a %d %b %Y %H:%M} {ctx['tz_abbr']} · 7 days, hourly · values in cells: 3-hour blocks"
        fig.text(0.085, 1 - 0.62 / H, stamp, fontsize=8.5, color=INK2, va="top")
        model = "Forecast: Open-Meteo best match (ECMWF IFS 9 km over East Africa, blended)"
        if "init_ens" in ex:
            model += f" · ensemble: ECMWF IFS 0.25° ENS, {ex.get('ens_n', 51)} members, run {ex['init_ens']:%Y-%m-%d %H}Z"
        fig.text(0.955, 1 - 0.28 / H, "By Gava Charles · Kampala Heat Atlas", fontsize=10, fontweight="bold", color=INK2, va="top", ha="right")
        fig.text(0.955, 1 - 0.62 / H, model, fontsize=7.5, color=INK2, va="top", ha="right")
        # forecast hour row under the last time axis
        last = [s for s in secs if s not in ("maps",)][-1]
        ax = axes[last]
        if "init_ens" in ex:
            for a, _ in bl[::2]:
                fh = int((a.tz_convert("UTC") - ex["init_ens"]).total_seconds() // 3600)
                ax.annotate(f"F{fh:03d}", (mdates.date2num(a) + 0.0625, 0), xycoords=("data", "axes fraction"), xytext=(0, -10),
                            textcoords="offset points", ha="center", fontsize=6.4, color=MUTED)
        fig.text(0.085, 0.1 / H, "WBGT: 0.7 Tnw + 0.2 Tg + 0.1 Ta from a radiant budget (screening; a WBGT meter on site is the reference). "
                 "Gust at hook height: 10 m gust × (hook/10)^1/7. Concrete evaporation: Uno (1998), ACI 305. Coatings: ISO 8502-4. Clear sky: Haurwitz. "
                 "\nMaps: daily rain and gust on a 7 × 7 grid (about 90 × 90 km). Thresholds: docs/site-weather/thresholds.json; "
                 "the contract, crane manufacturer's limits and site H&S plan take precedence. Data: Open-Meteo (CC BY 4.0), ECMWF.",
                 fontsize=6.8, color=INK2, va="bottom")
        fig.savefig(path, dpi=115)
        plt.close(fig)

    allk = list(LONG)
    draw(out_dir / "site_meteogram_latest.png",
         ["day", "heat", "heat_r", "rain", "rain_r", "wind", "wind_r", "sun", "sun_r", "evap", "dew", "conc_r", "trades", "daily", "maps"], allk, "")
    draw(out_dir / "site_meteogram_concrete.png",
         ["day", "rain", "rain_r", "evap", "dew", "conc_r", "sun", "sun_r", "trades", "daily"], ["concrete", "asphalt", "coatings", "earthworks"],
         " · concrete, asphalt, painting, earthworks")
    draw(out_dir / "site_meteogram_lifting.png",
         ["day", "wind", "wind_r", "rain", "rain_r", "sun_r", "trades", "daily"], ["crane", "height"], " · cranes and work at height")
    draw(out_dir / "site_meteogram_heat.png",
         ["day", "heat", "heat_r", "sun", "sun_r", "trades", "daily"], ["labour"], " · heat and outdoor labour")
    return ex
