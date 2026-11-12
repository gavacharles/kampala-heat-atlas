"""Module 0: sun path, optimal panel tilt and facade exposure for Kampala (runs offline)."""
import matplotlib.pyplot as plt
import numpy as np

from kampala import sun
from kampala.config import OUTPUTS, FIGURES

sp = sun.solar_positions()

# sun-path diagram (polar: azimuth vs zenith)
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(projection="polar")
ax.set_theta_zero_location("N"); ax.set_theta_direction(-1)
for month, colour in [(3, "tab:green"), (6, "tab:red"), (12, "tab:blue")]:
    day = sp[(sp.index.month == month) & (sp.index.day == 21)]
    ax.plot(np.radians(day["azimuth"]), day["apparent_zenith"], color=colour, label=f"21 {day.index[0]:%b}")
ax.set_rmax(90); ax.set_rticks([15, 30, 45, 60, 75, 90])
ax.set_title("Sun path over Kampala (0.31°N)\nradius = zenith angle", pad=20)
ax.legend(loc="lower left", bbox_to_anchor=(-0.1, -0.1))
fig.savefig(FIGURES / "m0_sun_path.png", dpi=200, bbox_inches="tight")

poa = sun.clear_sky_poa()
poa.to_csv(OUTPUTS / "tables" / "m0_clear_sky_tilt_azimuth.csv", index=False)
best = poa.loc[poa["kwh_m2_yr"].idxmax()]
flat = poa.loc[poa["tilt"] == 0, "kwh_m2_yr"].iloc[0]
print("Clear-sky annual irradiation by tilt (kWh/m2/yr):")
print(poa.pivot(index="tilt", columns="azimuth", values="kwh_m2_yr").round(0))
print(f"\nBest: tilt {best.tilt:.0f}°, azimuth {best.azimuth:.0f}° -> {best.kwh_m2_yr:.0f} "
      f"({100 * (best.kwh_m2_yr / flat - 1):+.1f}% vs flat)")

fac = sun.facade_insolation()
fac.to_csv(OUTPUTS / "tables" / "m0_facade_direct_beam.csv")
print("\nDirect beam on vertical facades (kWh/m2/yr):")
print(fac.round(0))
print(f"Max solar elevation range: noon sun stays above {sp.groupby(sp.index.date)['apparent_elevation'].max().min():.1f}°")
