"""Module 0: the Kampala cube - terrain, 2.5D buildings, wind climatology."""
import ee
import numpy as np
import pandas as pd

from . import config


def terrain(region: ee.Geometry) -> ee.Image:
    """FABDEM bare-earth elevation (30 m), slope, aspect, hillshade."""
    dem = (
        ee.ImageCollection(config.load()["cube"]["dem"])
        .filterBounds(region)
        .mosaic()
        .select([0])
        .setDefaultProjection("EPSG:4326", None, 30)
        .rename("elevation")
    )
    t = ee.Terrain.products(dem).select(["elevation", "slope", "aspect", "hillshade"])
    return t.clip(region).toFloat()


def buildings(year: int, region: ee.Geometry) -> ee.Image:
    """Open Buildings 2.5D Temporal (4 m): presence, height (m), fractional count.

    Available yearly 2016-2023 (one inference per year, around 30 June).
    """
    col = (
        ee.ImageCollection(config.load()["cube"]["buildings_temporal"])
        .filter(ee.Filter.calendarRange(year, year, "year"))  # as in kampala-cbd-fire-safety
        .filterBounds(region)
    )
    return (
        col.mosaic()
        .setDefaultProjection(col.first().projection())  # mosaic drops the native 0.5 m grid
        .select(["building_presence", "building_height", "building_fractional_count"])
        .clip(region)
        .toFloat()
    )


def building_metrics(year: int, region: ee.Geometry, scale: int = 30) -> ee.Image:
    """Aggregate 2.5D buildings to an analysis grid (default 30 m, matching Landsat LST).

    built_frac   share of the cell covered by building pixels (presence > 0.5)
    height_mean  mean height of building pixels (m)
    height_p90   90th-percentile building height (m); p90 tracked OSM storeys best
                 (r = 0.86) in kampala-cbd-fire-safety, so it is the default height metric
    n_buildings  number of buildings (sum of fractional counts)
    """
    crs = config.load()["study_area"]["crs"]
    b = buildings(year, region)
    is_b = b.select("building_presence").gt(0.5)
    h = b.select("building_height").updateMask(is_b)
    # step 1: 0.5 m -> 4 m (model's effective resolution), step 2: 4 m -> scale
    h4 = h.reduceResolution(ee.Reducer.mean(), maxPixels=1024).reproject(crs=crs, scale=4)
    f4 = is_b.unmask(0).reduceResolution(ee.Reducer.mean(), maxPixels=1024).reproject(crs=crs, scale=4)
    # count = mean fractional count x (cell area / 0.25 m2), as in uganda-trade-corridors 27_growth_temporal.py
    c4 = b.select("building_fractional_count").reduceResolution(ee.Reducer.mean(), maxPixels=1024).reproject(crs=crs, scale=4)
    built = f4.reduceResolution(ee.Reducer.mean(), maxPixels=4096).reproject(crs=crs, scale=scale).rename("built_frac")
    hm = h4.reduceResolution(ee.Reducer.mean(), maxPixels=4096).reproject(crs=crs, scale=scale).rename("height_mean")
    hp = h4.reduceResolution(ee.Reducer.percentile([90]), maxPixels=4096).reproject(crs=crs, scale=scale).rename("height_p90")
    n = (c4.reduceResolution(ee.Reducer.mean(), maxPixels=4096).reproject(crs=crs, scale=scale)
         .multiply(scale * scale / 0.25).rename("n_buildings"))
    return ee.Image.cat([built, hm, hp, n]).toFloat()


def building_footprints(region: ee.Geometry, min_conf: float = 0.70) -> ee.FeatureCollection:
    """Open Buildings v3 polygons (used for rooftop solar in Module 5)."""
    return (
        ee.FeatureCollection(config.load()["cube"]["buildings_polygons"])
        .filterBounds(region)
        .filter(ee.Filter.gte("confidence", min_conf))
    )


def wind_hourly(point: ee.Geometry, year: int) -> pd.DataFrame:
    """ERA5-Land hourly 10 m wind at a point for one year -> speed & direction."""
    era5 = ee.ImageCollection(config.load()["cube"]["era5_land"]).select(
        ["u_component_of_wind_10m", "v_component_of_wind_10m"]
    )
    parts = []
    for m in range(1, 13):  # month by month keeps each request under the memory limit
        start = ee.Date.fromYMD(year, m, 1)
        rows = era5.filterDate(start, start.advance(1, "month")).getRegion(point, 11132).getInfo()
        parts.append(pd.DataFrame(rows[1:], columns=rows[0]))
    df = pd.concat(parts, ignore_index=True)
    df["time"] = pd.to_datetime(df["time"], unit="ms", utc=True).dt.tz_convert("Africa/Kampala")
    u, v = df["u_component_of_wind_10m"], df["v_component_of_wind_10m"]
    df["speed"] = (u**2 + v**2) ** 0.5
    # meteorological convention: direction the wind blows FROM, degrees from north
    df["dir_from"] = (np.degrees(np.arctan2(-u, -v)) + 360) % 360
    return df[["time", "speed", "dir_from"]]
