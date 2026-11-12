"""Module 0: ERA5-Land wind climatology at Kampala -> wind rose + by-hour summary."""
import ee
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from kampala import config, cube, ee_utils
from kampala.config import DATA, OUTPUTS, FIGURES

ee_utils.init()
cfg = config.load()
c = cfg["study_area"]["centre"]
pt = ee.Geometry.Point([c["lon"], c["lat"]])
y0, y1 = cfg["cube"]["wind_years"]

frames = []
for y in range(y0, y1 + 1):
    print("fetching", y)
    frames.append(cube.wind_hourly(pt, y))
df = pd.concat(frames)
df.to_csv(DATA / "interim" / "m0_era5land_wind_kampala.csv", index=False)

# wind rose (16 sectors)
bins = np.arange(-11.25, 360, 22.5)
sector = pd.cut((df["dir_from"] + 11.25) % 360 - 11.25, bins, labels=False)
freq = sector.value_counts(normalize=True).sort_index()
fig = plt.figure(figsize=(6, 6)); ax = fig.add_subplot(projection="polar")
ax.set_theta_zero_location("N"); ax.set_theta_direction(-1)
ax.bar(np.radians(freq.index * 22.5), freq.values * 100, width=np.radians(20), color="tab:blue", alpha=.8)
ax.set_title(f"Wind direction frequency (%), ERA5-Land {y0}-{y1}")
fig.savefig(FIGURES / "m0_wind_rose.png", dpi=200, bbox_inches="tight")

df["hour"] = df["time"].dt.hour
print(df.groupby("hour")[["speed", "dir_from"]].median().round(1))
