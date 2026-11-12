"""Image stacks for the U-Net from Microsoft Planetary Computer (no Earth Engine; used when the EE
project is in Restricted Mode). Window: 32.45-32.72 E, 0.18-0.45 N (~30 x 30 km, Kampala + fringe).

  s2_<epoch>.tif   B02 B03 B04 B08 B11 B12 dry-season median of a three-year window (as for the RF),
                   SCL clear classes only (4 vegetation, 5 bare, 6 water, 11 snow), baseline offset
                   removed for processing baseline >= 04.00, x 10000, int16, 10 m, UTM 36N
  io_<epoch>.tif   Impact Observatory annual 10 m land cover (2017 / 2021 / 2023) recoded to thesis
                   classes (built kept as 9 for the dense/sparse/bare split), uint8
Approach and offset handling follow uganda-rainfall-climate/src/kp_s2.py. Existing files are kept.
"""
import sys

import numpy as np
import planetary_computer as pc
import pystac_client
import rasterio
from odc.stac import load

from kampala.config import DATA

BBOX = [32.45, 0.18, 32.72, 0.45]
OUT = DATA / "processed" / "m1b"; OUT.mkdir(parents=True, exist_ok=True)
EPOCHS = {"t1": (2017, 2017), "t2": (2021, 2021), "t3": (2025, 2023)}     # epoch: (imagery year, IO LULC year)
BANDS = ["B02", "B03", "B04", "B08", "B11", "B12"]
cat = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=pc.sign_inplace)


def windows(y):
    out = []
    for yy in (y - 1, y, y + 1):
        out += [(f"{yy - 1}-12-01", f"{yy}-02-28"), (f"{yy}-06-01", f"{yy}-08-31")]
    return out


def write(path, arr, geobox, dtype, nodata):
    prof = dict(driver="GTiff", height=arr.shape[-2], width=arr.shape[-1], count=1 if arr.ndim == 2 else arr.shape[0],
                dtype=dtype, crs="EPSG:32636", transform=geobox.transform, nodata=nodata, compress="deflate", tiled=True)
    with rasterio.open(path, "w", **prof) as d:
        d.write(arr if arr.ndim == 3 else arr[None])


for ep in (sys.argv[1:] or EPOCHS):
    yr, io_yr = EPOCHS[ep]
    s2_path, io_path = OUT / f"s2_{ep}.tif", OUT / f"io_{ep}.tif"
    if s2_path.exists():
        with rasterio.open(s2_path) as r:
            from odc.geo.geobox import GeoBox
            geobox = GeoBox((r.height, r.width), r.transform, "EPSG:32636")
        print(ep, "s2 exists", flush=True)
    else:
        items = []
        for a, b in windows(yr):
            items += list(cat.search(collections=["sentinel-2-l2a"], bbox=BBOX, datetime=f"{a}/{b}",
                                     query={"eo:cloud_cover": {"lt": 40}}).items())
        print(ep, len(items), "scenes", flush=True)
        ds = load(items, bands=BANDS + ["SCL"], bbox=BBOX, crs="EPSG:32636", resolution=10, groupby="solar_day",
                  chunks={"x": 1024, "y": 1024, "time": 1}, fail_on_error=False)
        base = {i.datetime.date(): float(i.properties.get("s2:processing_baseline", "0")) for i in items}
        off = np.array([1000 if base.get(np.datetime64(t, "D").astype(object), 0) >= 4 else 0 for t in ds.time.values])
        clear = ds.SCL.isin([4, 5, 6, 11])
        stack = [(ds[b].astype("float32") - off[:, None, None]).where(clear & (ds[b] > 0)).median("time", skipna=True)
                 for b in BANDS]
        arr = np.nan_to_num(np.clip(np.stack([s.compute().values for s in stack]), 0, 10000)).astype("int16")
        geobox = ds.odc.geobox
        write(s2_path, arr, geobox, "int16", 0)
        print(ep, "s2 written", arr.shape, "valid", float((arr[2] > 0).mean()), flush=True)
    if io_path.exists():
        continue
    io = list(cat.search(collections=["io-lulc-annual-v02"], bbox=BBOX, datetime=f"{io_yr}-01-01/{io_yr}-12-31").items())
    dl = load(io, bands=["data"], geobox=geobox, resampling="nearest").data.max("time").values
    lut = np.zeros(256, "uint8")
    for src, dst in ((1, 7), (2, 3), (4, 6), (5, 4), (11, 4), (7, 9), (8, 5)):
        lut[src] = dst
    write(io_path, lut[dl.astype("uint8")], geobox, "uint8", 0)
    print(ep, "io labels written", flush=True)
