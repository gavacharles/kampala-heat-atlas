"""The 3D urban climate model ('the cube'): 2 m district models of terrain + buildings, with
sun, shade, sky view, roof irradiance and a lake-breeze flow field.

District model (2 x 2 km, 2 m cells)
    DSM        FABDEM bare-earth terrain + extruded Overture footprints. Building height from
               Overture `height`, else `num_floors` x 3.2 m, else Open Buildings 2.5D p90 at the
               centroid (2023), else 3.5 m (single storey); capped at 120 m.
    Shadows    ray-marching on the DSM for every daylight hour of the 21st of each month
               (pvlib solar positions; sun elevation > 5 deg).
    SVF        sky view factor from horizon angles in 16 directions out to 100 m:
               SVF = 1 - mean(sin^2 horizon).
    Irradiance horizontal surfaces: beam x sunlit x sin(elev) + diffuse x SVF, from Ineichen clear
               sky scaled so annual GHI matches ERA5-Land (1,917 kWh/m2/yr), i.e. a mean
               cloudiness correction.
    Shade      hours of shade at ground level between 10:00 and 16:00 (pedestrian heat exposure).
    Wind       2D potential flow around the building footprints for the daytime southerly lake
               breeze (Laplace equation, Jacobi relaxation, 4 m). Inviscid: it shows corridors and
               blocked areas, not wakes or turbulence (CFD is the next step).
City-wide building metrics
    orientation  long-axis azimuth from the minimum rotated rectangle (0-180 deg).
    wall_sun     annual clear-sky direct beam on the four walls, area-weighted (kWh/m2/yr), and
                 the penalty vs the same building turned to an east-west long axis.
    frontal      frontal area index for wind from the south on 100 m cells (lambda_f).
"""
import geopandas as gpd
import numpy as np
import pandas as pd
import pvlib
import rasterio
import shapely
import shapely.geometry
from scipy import ndimage
from rasterio.features import rasterize
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject

from . import config
from .config import DATA

RES = 2.0
GHI_ERA5 = 1916.7  # kWh/m2/yr, mean of ERA5-Land 2015-2024 over the study area (M5)


# ----------------------------------------------------------------------------- district DSM
def footprint_heights_ee(gdf, cache, chunk=400, workers=4):
    """p90 of Open Buildings 2.5D (2023) building_height inside each footprint at 4 m.

    Method carried over from kampala-cbd-fire-safety scripts/gee/01_building_heights.py, where
    the footprint p90 tracked OSM storey counts best (r = 0.86). Cached per district.
    """
    import ee
    from concurrent.futures import ThreadPoolExecutor

    from . import ee_utils

    if cache.exists():
        return pd.read_parquet(cache)["h_p90"].values
    ee_utils.init()
    g = gdf[["geometry"]].to_crs("EPSG:4326").copy()
    g["geometry"] = g.geometry.simplify(0.000005)   # ~0.5 m; keeps payloads small (as in fire-safety)
    g["i"] = np.arange(len(g))

    def one(i):
        part = g.iloc[i:i + chunk]
        region = ee.Geometry.Rectangle(list(part.total_bounds))
        # filterBounds before mosaic and no projection override: the recipe that works in fire-safety
        img = (ee.ImageCollection(config.load()["cube"]["buildings_temporal"]).filterBounds(region)
               .filter(ee.Filter.calendarRange(2023, 2023, "year")).mosaic().select(["building_height"]))
        for attempt in range(6):
            try:
                res = img.reduceRegions(collection=ee.FeatureCollection(part.__geo_interface__),
                                        reducer=ee.Reducer.percentile([90]), scale=4)
                res = res.select([".*"], retainGeometry=False).getInfo()
                return [(f["properties"]["i"], f["properties"].get("p90")) for f in res["features"]]
            except Exception:  # noqa: BLE001
                if attempt == 5:
                    raise
                import time
                time.sleep(20 * (attempt + 1))

    with ThreadPoolExecutor(workers) as pool:
        out = [x for part in pool.map(one, range(0, len(g), chunk)) for x in part]
    h = np.full(len(g), np.nan)
    for i, v in out:
        h[i] = np.nan if v is None else v
    pd.DataFrame({"h_p90": h}).to_parquet(cache)
    return h


def district_buildings(cx, cy, half=1000.0):
    crs = config.load()["study_area"]["crs"]
    b = gpd.read_parquet(DATA / "raw" / "overture_buildings.parquet").to_crs(crs)
    b = b.cx[cx - half - 50: cx + half + 50, cy - half - 50: cy + half + 50]
    b = b[b.geometry.type.isin(["Polygon", "MultiPolygon"])].copy().reset_index(drop=True)
    cache = DATA / "processed" / "m6" / f"heights_{int(cx)}_{int(cy)}_{int(half)}.parquet"
    cache.parent.mkdir(parents=True, exist_ok=True)
    p90 = footprint_heights_ee(b, cache)
    p90 = np.where(np.isfinite(p90) & (p90 > 2.0), p90, np.nan)
    h = (b["height"].astype(float)
         .fillna(b["num_floors"].astype(float) * 3.2)
         .fillna(pd.Series(p90, index=b.index))
         .fillna(3.5).clip(2.5, 120))
    b["h"] = h.values
    b["h_source"] = np.select([b["height"].notna(), b["num_floors"].notna(), np.isfinite(p90)],
                              ["overture height", "floors x 3.2", "open buildings footprint p90"], "default 3.5 m")
    return b


def district_dsm(cx, cy, half=1000.0, res=RES):
    n = int(2 * half / res)
    tr = from_origin(cx - half, cy + half, res, res)
    crs = config.load()["study_area"]["crs"]
    terrain = np.zeros((n, n), dtype="float32")
    with rasterio.open(DATA / "processed" / "m0" / "terrain_fabdem_30m.tif") as r:
        reproject(rasterio.band(r, 1), terrain, dst_transform=tr, dst_crs=crs, resampling=Resampling.bilinear)
    b = district_buildings(cx, cy, half)
    bh = rasterize(((g, h) for g, h in zip(b.geometry, b["h"])), out_shape=(n, n), transform=tr,
                   fill=0, dtype="float32", merge_alg=rasterio.enums.MergeAlg.replace)
    bid = rasterize(((g, i + 1) for i, g in enumerate(b.geometry)), out_shape=(n, n), transform=tr,
                    fill=0, dtype="int32")
    return dict(terrain=terrain, bh=bh, dsm=terrain + bh, bid=bid, buildings=b, transform=tr, res=res)


# ----------------------------------------------------------------------------- sun & shade
def sun_positions(year=2025, hours=range(6, 19), elev_min=5.0):
    c = config.load()["study_area"]["centre"]
    loc = pvlib.location.Location(c["lat"], c["lon"], tz="Africa/Kampala", altitude=1190)
    times = pd.DatetimeIndex([pd.Timestamp(year, m, 21, h, 30, tz="Africa/Kampala") for m in range(1, 13) for h in hours])
    sp = loc.get_solarposition(times)
    cs = loc.get_clearsky(times, model="ineichen")
    df = pd.DataFrame({"azimuth": sp["azimuth"], "elevation": sp["apparent_elevation"],
                       "dni": cs["dni"], "dhi": cs["dhi"], "ghi": cs["ghi"]})
    df = df[df["elevation"] > elev_min]
    # one representative day per month -> weight each hour by days in the month
    df["weight"] = df.index.days_in_month
    # scale clear sky to ERA5 annual GHI (mean cloudiness)
    clear_annual = (df["ghi"] * df["weight"]).sum() / 1000
    df["k"] = GHI_ERA5 / clear_annual
    return df


def shadow(dsm, azimuth, elevation, res=RES, max_dist=400.0):
    """True where a cell is shaded by terrain/buildings for a sun at (azimuth, elevation)."""
    az, el = np.radians(azimuth), np.radians(elevation)
    dx, dy = np.sin(az), -np.cos(az)     # towards the sun: +col east, -row north
    tan_el = np.tan(el)
    n = dsm.shape
    shaded = np.zeros(n, dtype=bool)
    hmax = float(dsm.max())
    steps = int(min(max_dist, (hmax - float(dsm.min())) / max(tan_el, 1e-3)) / res)
    for s in range(1, steps + 1):
        oc, orow = int(round(s * dx)), int(round(s * dy))
        if oc == 0 and orow == 0:
            continue
        rise = s * res * tan_el
        # neighbour along the ray towards the sun: dsm[r + orow, c + oc]
        r0, r1 = max(0, -orow), n[0] - max(0, orow)
        c0, c1 = max(0, -oc), n[1] - max(0, oc)
        nb = dsm[r0 + orow:r1 + orow, c0 + oc:c1 + oc]
        shaded[r0:r1, c0:c1] |= nb > dsm[r0:r1, c0:c1] + rise
        if rise > hmax:
            break
    return shaded


def sky_view_factor(dsm, res=RES, max_dist=100.0, n_dir=16):
    n = dsm.shape
    acc = np.zeros(n, dtype="float32")
    for k in range(n_dir):
        a = 2 * np.pi * k / n_dir
        dx, dy = np.sin(a), -np.cos(a)
        hz = np.zeros(n, dtype="float32")
        for s in range(1, int(max_dist / res) + 1):
            oc, orow = int(round(s * dx)), int(round(s * dy))
            r0, r1 = max(0, -orow), n[0] - max(0, orow)
            c0, c1 = max(0, -oc), n[1] - max(0, oc)
            nb = dsm[r0 + orow:r1 + orow, c0 + oc:c1 + oc]
            ang = np.arctan((nb - dsm[r0:r1, c0:c1]) / (s * res))
            np.maximum(hz[r0:r1, c0:c1], ang, out=hz[r0:r1, c0:c1])
        acc += np.sin(np.clip(hz, 0, None)) ** 2
    return 1 - acc / n_dir


def solar_run(model, log=print):
    """Annual horizontal irradiance (kWh/m2/yr), sunlit share and ground shade hours 10-16h."""
    dsm = model["dsm"]
    sp = sun_positions()
    svf = sky_view_factor(dsm)
    irr = np.zeros(dsm.shape, dtype="float64")
    shade_hours = np.zeros(dsm.shape, dtype="float32")
    sunlit_frac = np.zeros(dsm.shape, dtype="float32")
    wsum = 0.0
    frames = {}
    for i, (t, row) in enumerate(sp.iterrows()):
        sh = shadow(dsm, row.azimuth, row.elevation)
        beam = row.dni * np.sin(np.radians(row.elevation))
        irr += row.weight * row.k * (beam * (~sh) + row.dhi * svf) / 1000.0
        sunlit_frac += row.weight * (~sh)
        wsum += row.weight
        if 10 <= t.hour < 16:
            shade_hours += row.weight * sh / 365.0       # mean daily shade hours 10-16 h
        if t.month in (3, 6, 12):
            frames[(t.month, t.hour)] = sh
        if i % 24 == 0:
            log(f"    sun position {i + 1}/{len(sp)}")
    return dict(irr=irr.astype("float32"), svf=svf, shade_hours=shade_hours,
                sunlit=sunlit_frac / wsum, shadow_frames=frames, sun=sp)


def roof_table(model, solar):
    """Per-building roof irradiance with real 3D shading (flat-roof assumption)."""
    bid, irr, sun = model["bid"], solar["irr"], solar["sunlit"]
    ids = bid[bid > 0]
    n = len(model["buildings"]) + 1
    cnt = np.bincount(ids, minlength=n)
    tot = np.bincount(ids, weights=irr[bid > 0], minlength=n)
    sl = np.bincount(ids, weights=sun[bid > 0], minlength=n)
    b = model["buildings"].copy()
    b["roof_px"] = cnt[1:len(b) + 1]
    b["roof_kwh_m2"] = np.where(b.roof_px > 0, tot[1:len(b) + 1] / np.maximum(b.roof_px, 1), np.nan)
    b["roof_sunlit"] = np.where(b.roof_px > 0, sl[1:len(b) + 1] / np.maximum(b.roof_px, 1), np.nan)
    b["shading_loss_pct"] = 100 * (1 - b["roof_kwh_m2"] / GHI_ERA5)
    return b


# ----------------------------------------------------------------------------- wind
def potential_flow(model, wind_from=180.0, res=4.0):
    """2D potential flow for a uniform wind blowing FROM `wind_from` degrees, around buildings.

    Laplace equation for the velocity potential on fluid cells (5-point stencil, sparse direct
    solve). Outer boundary = free-stream potential; building walls = no normal flow (solid
    neighbours are dropped from the stencil). Returns u (east), v (north), speed ratio (free
    stream = 1) on a `res` grid.
    """
    import scipy.sparse as sps
    from scipy.sparse.linalg import spsolve

    f = int(res / model["res"])
    bh = model["bh"]
    bh = bh[: (bh.shape[0] // f) * f, : (bh.shape[1] // f) * f]
    solid = bh.reshape(bh.shape[0] // f, f, bh.shape[1] // f, f).max(axis=(1, 3)) > 0
    n, m = solid.shape
    to = np.radians(wind_from + 180.0)          # direction the air moves TO
    ue, vn = np.sin(to), np.cos(to)
    yy, xx = np.mgrid[0:n, 0:m]
    phi0 = (ue * xx - vn * yy).astype("float64")  # rows increase southwards
    border = np.zeros((n, m), bool); border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = True
    unknown = ~solid & ~border
    idx = -np.ones((n, m), dtype=np.int64); idx[unknown] = np.arange(unknown.sum())
    rows, cols, vals = [], [], []
    rhs = np.zeros(unknown.sum())
    r_u, c_u = np.nonzero(unknown)
    me = idx[r_u, c_u]
    diag = np.zeros(len(me))
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        rn, cn = r_u + dr, c_u + dc
        nb_solid = solid[rn, cn]
        nb_border = border[rn, cn] & ~nb_solid
        nb_unknown = unknown[rn, cn]
        diag += (~nb_solid).astype(float)
        rows.append(me[nb_unknown]); cols.append(idx[rn[nb_unknown], cn[nb_unknown]]); vals.append(-np.ones(nb_unknown.sum()))
        np.add.at(rhs, me[nb_border], phi0[rn[nb_border], cn[nb_border]])
    rows.append(me); cols.append(me); vals.append(diag)
    eps = 1e-6   # tiny anchor so courtyards sealed by buildings do not make the system singular
    A = sps.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(len(me), len(me)))
    A = A + eps * sps.identity(len(me), format="csr")
    phi = phi0.copy()
    phi[unknown] = spsolve(A.tocsc(), rhs + eps * phi0[unknown])
    # fill walls with the nearest fluid value so the gradient at walls has no normal component
    _, (ir, ic) = ndimage.distance_transform_edt(solid, return_indices=True)
    phif = phi[ir, ic]
    gy, gx = np.gradient(phif)
    u, v = gx, -gy
    u[solid] = v[solid] = 0
    return dict(u=u, v=v, speed=np.hypot(u, v), solid=solid, res=res)


# ----------------------------------------------------------------------------- city-wide metrics
def wall_sun_lookup():
    """Annual clear-sky direct beam (kWh/m2/yr, ERA5-scaled) on a vertical wall facing each azimuth 0..359."""
    sp = sun_positions()
    az = np.arange(360)
    out = np.zeros(360)
    for _, row in sp.iterrows():
        inc = np.cos(np.radians(row.elevation)) * np.cos(np.radians(row.azimuth - az))
        out += row.weight * row.k * row.dni * np.clip(inc, 0, None) / 1000
    return out


def building_orientation(gdf):
    """Long-axis azimuth (0-180, 0 = N-S, 90 = E-W), length, width of each footprint."""
    mrr = shapely.minimum_rotated_rectangle(gdf.geometry.values)
    coords = shapely.get_coordinates(shapely.get_exterior_ring(mrr)).reshape(-1, 5, 2)[:, :3, :]
    e1 = coords[:, 1] - coords[:, 0]; e2 = coords[:, 2] - coords[:, 1]
    l1, l2 = np.hypot(*e1.T), np.hypot(*e2.T)
    long_vec = np.where((l1 >= l2)[:, None], e1, e2)
    az = (np.degrees(np.arctan2(long_vec[:, 0], long_vec[:, 1])) + 180) % 180
    return az, np.maximum(l1, l2), np.minimum(l1, l2)


def wall_sun_penalty(az_long, length, width, lookup):
    """Area-weighted wall irradiance for the actual orientation vs an east-west long axis.

    Long walls face az_long +/- 90; short walls face az_long and az_long + 180.
    """
    def walls(a):
        ai = np.round(a).astype(int)
        long_faces = lookup[(ai + 90) % 360] + lookup[(ai + 270) % 360]
        short_faces = lookup[ai % 360] + lookup[(ai + 180) % 360]
        return (length * long_faces + width * short_faces) / (2 * (length + width))
    actual = walls(az_long)
    best = walls(np.full_like(az_long, 90.0))
    return actual, actual - best
