"""Module 0: sun geometry for Kampala (feeds building orientation + rooftop solar)."""
import numpy as np
import pandas as pd
import pvlib

from . import config


def site() -> pvlib.location.Location:
    c = config.load()["study_area"]["centre"]
    return pvlib.location.Location(c["lat"], c["lon"], tz="Africa/Kampala", altitude=1190)


def solar_positions(year: int = 2025, freq: str = "15min") -> pd.DataFrame:
    """Sun elevation/azimuth for every daylight time step of a year."""
    times = pd.date_range(f"{year}-01-01", f"{year + 1}-01-01", freq=freq, tz="Africa/Kampala", inclusive="left")
    sp = site().get_solarposition(times)
    return sp[sp["apparent_elevation"] > 0]


def clear_sky_poa(tilts=range(0, 31, 5), azimuths=(0, 90, 180, 270), year: int = 2025) -> pd.DataFrame:
    """Annual clear-sky plane-of-array irradiation (kWh/m2/yr) by tilt and azimuth.

    Clear-sky (Ineichen) only - this is a geometric screening. Module 5 replaces
    it with measured/reanalysis irradiance and 3D shading.
    """
    loc = site()
    times = pd.date_range(f"{year}-01-01", f"{year + 1}-01-01", freq="1h", tz="Africa/Kampala", inclusive="left")
    sp = loc.get_solarposition(times)
    cs = loc.get_clearsky(times, model="ineichen")
    out = []
    for tilt in tilts:
        for az in azimuths:
            poa = pvlib.irradiance.get_total_irradiance(
                tilt, az, sp["apparent_zenith"], sp["azimuth"], cs["dni"], cs["ghi"], cs["dhi"]
            )["poa_global"]
            out.append({"tilt": tilt, "azimuth": az, "kwh_m2_yr": poa.clip(lower=0).sum() / 1000})
    return pd.DataFrame(out)


def facade_insolation(year: int = 2025) -> pd.Series:
    """Annual clear-sky direct beam on vertical walls facing N/E/S/W (kWh/m2/yr).

    Answers 'which way should long facades face?' for building orientation.
    """
    loc = site()
    times = pd.date_range(f"{year}-01-01", f"{year + 1}-01-01", freq="1h", tz="Africa/Kampala", inclusive="left")
    sp = loc.get_solarposition(times)
    cs = loc.get_clearsky(times, model="ineichen")
    res = {}
    for name, az in {"north": 0, "east": 90, "south": 180, "west": 270}.items():
        aoi = pvlib.irradiance.aoi(90, az, sp["apparent_zenith"], sp["azimuth"])
        beam = cs["dni"] * np.cos(np.radians(aoi)).clip(lower=0)
        res[name] = beam.sum() / 1000
    return pd.Series(res, name="direct_kwh_m2_yr")
