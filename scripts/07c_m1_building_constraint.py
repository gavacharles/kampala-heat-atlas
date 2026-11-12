"""Module 1 fix: building-evidence constraint -> recovers the bare-soil class.

Problem: Dynamic World weak labels contain almost no bare soil, so dry-season bare cropland and
cleared plots were mapped as built_sparse (about 104 km2 'built -> vegetation' per period, mostly
2017 fields in the south).
Fix: a 10 m pixel may stay built (dense or sparse) only if independent building evidence exists
within ~30 m:
    2017  Open Buildings 2.5D 2016 (30 m built share > 0.02 in a 3x3 cell window)
    2021  Open Buildings 2.5D 2020 (same rule)
    2025  Open Buildings 2.5D 2023 (same rule) OR Overture 2026 footprints rasterised at 10 m,
          dilated by 2 pixels (catches buildings finished after 2023)
Built pixels without evidence become bare_soil (5): fallow fields, cleared plots, murram yards,
construction sites. The raw RF maps are kept as *_raw.tif.
"""
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from rasterio.warp import Resampling, reproject
from scipy import ndimage

from kampala import config
from kampala.config import DATA, OUTPUTS

M0, M1 = DATA / "processed" / "m0", DATA / "processed" / "m1"
EP = config.load()["epochs"]
BYEAR = {"t1": 2016, "t2": 2020, "t3": 2023}
rows = []

for ep, spec in EP.items():
    canon = M1 / f"landcover_{ep}_{spec['year']}_10m.tif"
    raw = M1 / f"landcover_{ep}_{spec['year']}_10m_raw.tif"
    if not raw.exists():
        canon.rename(raw)
    with rasterio.open(raw) as r:
        lc, prof = r.read(1), r.profile

    with rasterio.open(M0 / f"buildings_{BYEAR[ep]}_30m.tif") as r:
        bf = r.read(1, masked=True).filled(0)
        ev30 = ndimage.maximum_filter(bf, size=3) > 0.02
        ev = np.zeros(lc.shape, dtype="uint8")
        reproject(ev30.astype("uint8"), ev, src_transform=r.transform, src_crs=r.crs,
                  dst_transform=prof["transform"], dst_crs=prof["crs"], resampling=Resampling.nearest)
    evidence = ev.astype(bool)

    if ep == "t3":
        b = gpd.read_parquet(DATA / "raw" / "overture_buildings.parquet", columns=["geometry"]).to_crs(prof["crs"])
        ov = rasterize(((g, 1) for g in b.geometry), out_shape=lc.shape, transform=prof["transform"],
                       fill=0, dtype="uint8", all_touched=True)
        evidence |= ndimage.binary_dilation(ov.astype(bool), iterations=2)

    built = np.isin(lc, [1, 2])
    to_bare = built & ~evidence
    fixed = lc.copy()
    fixed[to_bare] = 5
    with rasterio.open(canon, "w", **{**prof, "compress": "deflate"}) as dst:
        dst.write(fixed, 1)
    rows.append(dict(epoch=ep, year=spec["year"], built_raw_km2=built.sum() * 1e-4,
                     reassigned_to_bare_km2=to_bare.sum() * 1e-4,
                     built_kept_km2=(built & evidence).sum() * 1e-4,
                     share_reassigned_pct=100 * to_bare.sum() / max(built.sum(), 1)))
    print(rows[-1], flush=True)

pd.DataFrame(rows).to_csv(OUTPUTS / "tables" / "m1_building_constraint.csv", index=False)
