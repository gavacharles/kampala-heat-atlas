"""Kampala heat & solar meteogram: the 7-day forecast translated through the 3D model's findings.

    python scripts/25_meteogram.py <out_dir>

Standalone (no project data needed) so a daily GitHub Action can refresh it. Forecast: Open-Meteo
(https://open-meteo.com, free, no key; only the city-centre coordinates are sent), hourly, local time.
Panels
  1 air temperature, and felt heat (UTCI) for a pedestrian in an open street (sky view 0.8, in sun)
    and under a street tree (shaded, sky view 0.15); UTCI heat-stress bands behind
  2 cloud cover and precipitation
  3 10 m wind speed with direction arrows; lake-breeze hours (from 135-225 deg) marked
  4 rooftop PV output per kWp (flat panels, PR 0.80) and what 1 GWp of Kampala roofs would give
    against Uganda's average demand (0.53 GW, Umeme 2024)
Radiant budget as in scripts/18_scenarios.py (simplified SOLWEIG-style, standing person). Under a
tree the felt heat is lower by day and slightly higher at night (the canopy holds in long-wave heat).
"""
import json
import sys
import urllib.request
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pvlib
from pythermalcomfort.models import utci

LAT, LON, ALT = 0.3136, 32.5811, 1190
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
OUT.mkdir(parents=True, exist_ok=True)
SIGMA = 5.67e-8
INK, INK2, MUTED, GRID = "#0b0b0b", "#3a3a3e", "#55555a", "#e1e0d9"
BLUE, ORANGE, RED, GREEN = "#2a78d6", "#eb6834", "#c23434", "#2f8f3a"
plt.rcParams.update({"font.family": ["Helvetica Neue", "Arial", "DejaVu Sans"], "font.size": 9, "axes.edgecolor": "#c3c2b7",
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.labelcolor": INK2})

vars_ = "temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m,shortwave_radiation,direct_normal_irradiance,diffuse_radiation,cloud_cover,precipitation"
url = (f"https://api.open-meteo.com/v1/forecast?latitude={LAT}&longitude={LON}&hourly={vars_}"
       "&timezone=Africa%2FKampala&forecast_days=7&wind_speed_unit=ms")
for _try in range(5):                                       # the forecast service occasionally times out
    try:
        h = json.loads(urllib.request.urlopen(url, timeout=90).read())["hourly"]
        break
    except Exception:  # noqa: BLE001
        if _try == 4:
            raise
        import time
        time.sleep(10 * (_try + 1))
df = pd.DataFrame(h); df["time"] = pd.to_datetime(df["time"]).dt.tz_localize("Africa/Kampala")
df = df.set_index("time")
sp = pvlib.location.Location(LAT, LON, tz="Africa/Kampala", altitude=ALT).get_solarposition(df.index - pd.Timedelta(minutes=30))
elev = np.clip(sp["apparent_elevation"].values, 0, 90)


def tmrt(ta, dni, dhi, ghi, elev, shaded, svf):
    e = np.radians(elev)
    fp = 0.308 * np.cos(e * (1 - elev ** 2 / 48402))
    S = 0.7 * (fp * dni * (~shaded) * (elev > 0) + 0.5 * dhi * svf + 0.5 * 0.2 * ghi)
    ta_k = ta + 273.15
    tg = ta_k + np.where(shaded | (elev <= 0), 2.0, 2.0 + 13.0 * np.clip(ghi / 900, 0, 1))
    L = 0.97 * (0.5 * (0.85 * SIGMA * ta_k ** 4 * svf + SIGMA * (ta_k + 2) ** 4 * (1 - svf)) + 0.5 * 0.95 * SIGMA * tg ** 4)
    return ((S + L) / (0.97 * SIGMA)) ** 0.25 - 273.15


ta, rh = df.temperature_2m.values, df.relative_humidity_2m.values
v = np.clip(df.wind_speed_10m.values * 0.6, 0.5, 17)
dni, dhi, ghi = df.direct_normal_irradiance.values, df.diffuse_radiation.values, df.shortwave_radiation.values
u_sun = np.asarray(utci(tdb=ta, tr=tmrt(ta, dni, dhi, ghi, elev, np.zeros(len(df), bool), 0.8), v=v, rh=rh, limit_inputs=False)["utci"], float)
u_tree = np.asarray(utci(tdb=ta, tr=tmrt(ta, dni, dhi, ghi, elev, np.ones(len(df), bool), 0.15), v=v, rh=rh, limit_inputs=False)["utci"], float)
pv = np.clip(ghi, 0, None) / 1000 * 0.80 * 0.985

t = df.index
fig, ax = plt.subplots(4, 1, figsize=(11, 10.5), sharex=True, gridspec_kw=dict(height_ratios=[3, 1.3, 1.4, 1.8]), constrained_layout=True)
for lo, hi, col, lab in ((26, 32, "#fde0d2", "moderate"), (32, 38, "#f8b89c", "strong"), (38, 46, "#ec7372", "very strong")):
    ax[0].axhspan(lo, hi, color=col, alpha=0.45, lw=0)
    ax[0].text(t[-1], (lo + hi) / 2 if hi < 46 else 39, f" {lab} heat stress", fontsize=8, color=INK2, va="center")
ax[0].plot(t, u_sun, color=RED, lw=2, label="felt heat, open street in sun (UTCI)")
ax[0].plot(t, u_tree, color=GREEN, lw=2, label="felt heat, under a street tree (UTCI)")
ax[0].plot(t, ta, color=INK2, lw=1.3, ls="--", label="air temperature")
ax[0].set_ylabel("°C"); ax[0].set_ylim(min(15, np.nanmin(ta) - 1), max(40, np.nanmax(u_sun) + 1))
ax[0].legend(loc="upper left", ncol=3, frameon=False, fontsize=8)
i = int(np.nanargmax(u_sun))
ax[0].annotate(f"{u_sun[i]:.0f} °C in the sun\n{u_tree[i]:.0f} °C in tree shade", (t[i], u_sun[i]), xytext=(10, 4),
               textcoords="offset points", fontsize=8, color=INK)
ax[0].set_title(f"Kampala heat & solar meteogram · forecast issued {pd.Timestamp.now(tz='Africa/Kampala'):%d %b %Y %H:%M} EAT",
                loc="left", fontsize=12, fontweight="bold", color=INK)
ax[1].fill_between(t, df.cloud_cover, color="#c3c2b7", alpha=0.6, lw=0)
ax[1].set_ylim(0, 100); ax[1].set_ylabel("cloud %")
a2 = ax[1].twinx(); a2.bar(t, df.precipitation, width=0.035, color=BLUE)
a2.set_ylabel("rain mm/h", color=BLUE); a2.set_ylim(0, max(5, df.precipitation.max() * 1.2)); a2.grid(False); a2.spines["right"].set_visible(True)
ws, wd = df.wind_speed_10m.values, df.wind_direction_10m.values
lake = (wd >= 135) & (wd <= 225)
ax[2].plot(t, ws, color=INK2, lw=1.4)
for k in range(0, len(t), 3):
    to = np.radians(wd[k] + 180)
    ax[2].annotate("", xy=(mdates.date2num(t[k]) + 0.04 * np.sin(to), ws[k] + 0.6 * np.cos(to)), xytext=(mdates.date2num(t[k]), ws[k]),
                   arrowprops=dict(arrowstyle="-|>", color=BLUE if lake[k] else MUTED, lw=1.1))
ax[2].set_ylabel("wind m/s"); ax[2].set_ylim(0, max(6, ws.max() + 1.5))
ax[2].text(0.005, 0.86, "arrows point where the air goes · blue = lake breeze from the south", transform=ax[2].transAxes, fontsize=8, color=INK2)
ax[3].fill_between(t, pv, color="#f8b89c", lw=0); ax[3].plot(t, pv, color=ORANGE, lw=1.6)
ax[3].set_ylabel("kW per kWp"); ax[3].set_ylim(0, 1)
a4 = ax[3].twinx(); a4.set_ylim(0, 1); a4.set_ylabel("GW from 1 GWp of roofs", color=ORANGE); a4.grid(False); a4.spines["right"].set_visible(True)
a4.axhline(4674 / 8760, color=INK, ls="--", lw=1); a4.text(t[0], 4674 / 8760 + 0.03, " Uganda's average demand 0.53 GW", fontsize=8, color=INK)
daily = pd.Series(pv, index=t).resample("D").sum()
ax[3].text(0.005, 0.86, "kWh per kWp a day: " + "  ".join(f"{d:%a} {v:.1f}" for d, v in daily.items()), transform=ax[3].transAxes, fontsize=8, color=INK2)
ax[3].xaxis.set_major_locator(mdates.DayLocator(tz=t.tz)); ax[3].xaxis.set_major_formatter(mdates.DateFormatter("%a %d %b", tz=t.tz))
fig.text(0.01, -0.01, "Forecast: Open-Meteo (open-meteo.com). Felt heat: UTCI with a simplified radiant budget for a standing person; "
         "street wind = 0.6 x 10 m wind. Kampala Heat Atlas · Gava Charles", fontsize=7.5, color=MUTED)
fig.savefig(OUT / "meteogram_latest.png", dpi=150, bbox_inches="tight")
df.assign(utci_sun=u_sun, utci_tree=u_tree, pv_kw_per_kwp=pv).to_csv(OUT / "meteogram_latest.csv")
print("wrote", OUT / "meteogram_latest.png", "| max UTCI sun %.1f, tree %.1f; PV kWh/kWp next 7 days %.1f" % (np.nanmax(u_sun), np.nanmax(u_tree), pv.sum()))
