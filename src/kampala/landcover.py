"""Module 1: three-date semantic land-cover classification and change detection (Sentinel-2, 10 m).

Design
------
Features   dry-season S2 median bands + NDVI/NDBI/MNDWI/NDRE/BSI, NDVI 10th/90th percentile
           (seasonality), NIR GLCM texture (contrast, entropy; dense vs sparse fabric),
           FABDEM elevation and slope (valley-floor wetlands).
Labels     weak labels from Google Dynamic World (built on the same S2 scenes, so 2017 is covered),
           kept where the dry-season mean probability of the top class is >= 0.55. Wetland is
           supplemented with ESA WorldCover herbaceous wetland (class 90). Built samples are split
           into dense / sparse *locally* with the M0 Open Buildings 2.5D built fraction (30 m).
Model      Random Forest (300 trees), one per epoch (post-classification comparison).
           The same samples train scikit-learn (diagnostics, importance) and Earth Engine (mapping).
Accuracy   hold-out agreement with weak labels = *label agreement*, not map accuracy. Independent
           accuracy needs manually interpreted points (validation_points()).

Why this split: interactive Earth Engine requests have memory/time limits and this account cannot
write EE assets, so heavy steps are kept small (samples) or tiled (map download).
"""
import ee
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from shapely.geometry import Point

from . import config, cube, sentinel2

CLASSES = config.load()["landcover_classes"]  # 1..7
BUILT_TMP = 9  # Dynamic World 'built' before the dense/sparse split
# Dynamic World label -> thesis class (snow (8) dropped)
DW_FROM = [0, 1, 2, 3, 4, 5, 6, 7]
DW_TO = [7, 3, 4, 6, 4, 4, BUILT_TMP, 5]
BUILDING_YEAR = {"t1": 2016, "t2": 2020, "t3": 2023}
DENSE_BUILT_FRAC = 0.30  # >= 30 % of the 30 m cell under roofs -> built_dense


def _uses_l1c(epoch: str) -> bool:
    return config.load()["epochs"][epoch]["year"] < 2018


def features(epoch: str, region: ee.Geometry) -> ee.Image:
    cfg = config.load()
    s2 = cfg["sentinel2"]
    ep = cfg["epochs"][epoch]
    l1c = _uses_l1c(epoch)
    col = sentinel2.collection(epoch, region, use_fallback=l1c)
    comp = col.median()

    # fill cloud gaps from a wider all-month window around the epoch
    full = (
        ee.ImageCollection(s2["fallback_collection"] if l1c else s2["collection"])
        .filterBounds(region)
        .filterDate(ee.Date(ep["start"]).advance(-6, "month"), ee.Date(ep["end"]).advance(4, "month"))
        .linkCollection(ee.ImageCollection(s2["cloud_score_plus"]), ["cs_cdf"])
        .map(lambda i: i.updateMask(i.select("cs_cdf").gte(s2["cs_threshold"])).select(s2["bands"]).divide(10000))
    )
    filled = sentinel2.add_indices(comp.unmask(full.median()))

    ndvi = col.map(lambda i: i.normalizedDifference(["B8", "B4"]).rename("NDVI"))
    spread = ndvi.reduce(ee.Reducer.percentile([10, 90])).rename(["NDVI_p10", "NDVI_p90"])
    nir = filled.select("B8").multiply(1000).toInt().unmask(0)
    glcm = nir.glcmTexture(size=2).select(["B8_contrast", "B8_ent"])
    terrain = cube.terrain(region).select(["elevation", "slope"])
    return ee.Image.cat([filled, spread, glcm, terrain]).toFloat()


def dw_labels(epoch: str, region: ee.Geometry, min_prob: float = 0.55) -> ee.Image:
    """Dynamic World probabilities merged to thesis classes *before* taking the top class.

    grass + crops + shrub are summed into grass_cropland; otherwise their probability is
    split three ways and they almost never pass the confidence threshold.
    """
    ep = config.load()["epochs"][epoch]
    dw = sentinel2.dry_season(
        ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1").filterBounds(region).filterDate(ep["start"], ep["end"])
    ).mean()
    merged = ee.Image.cat([
        dw.select("trees"),                                                    # -> 3
        dw.select("grass").add(dw.select("crops")).add(dw.select("shrub_and_scrub")),  # -> 4
        dw.select("bare"),                                                     # -> 5
        dw.select("flooded_vegetation"),                                       # -> 6
        dw.select("water"),                                                    # -> 7
        dw.select("built"),                                                    # -> BUILT_TMP
    ])
    order = ee.List([3, 4, 5, 6, 7, BUILT_TMP])
    idx = merged.toArray().arrayArgmax().arrayGet(0)
    lab = idx.remap(ee.List.sequence(0, 5), order)
    conf = merged.reduce(ee.Reducer.max())
    wc_wet = ee.ImageCollection("ESA/WorldCover/v200").first().eq(90)
    lab = lab.where(wc_wet.And(lab.neq(BUILT_TMP)).And(lab.neq(7)), 6)
    return lab.updateMask(conf.gte(min_prob)).rename("label").toByte()


def sample(epoch: str, region: ee.Geometry, n_points: int = 40000, batch: int = 2000,
           seed: int = 42) -> gpd.GeoDataFrame:
    """Features + weak labels at random points (EE evaluates only those pixels, in batches).

    Class balancing is done afterwards with balance(); stratifiedSample over the whole area
    times out because it evaluates every pixel.
    """
    w, s_, e, n = config.load()["study_area"]["extended_bbox"]
    rng = np.random.default_rng(seed)
    xs, ys = rng.uniform(w, e, n_points), rng.uniform(s_, n, n_points)
    img = features(epoch, region).addBands(dw_labels(epoch, region))
    import json
    import time
    from concurrent.futures import ThreadPoolExecutor

    from .config import DATA

    cache = DATA / "interim" / f"m1_sample_batches_{epoch}"   # resumable: one file per batch
    cache.mkdir(parents=True, exist_ok=True)

    def one(i):
        f = cache / f"batch_{i:06d}.json"
        if f.exists():
            return json.loads(f.read_text())
        pts = ee.FeatureCollection([ee.Feature(ee.Geometry.Point([float(x), float(y)]), {"pid": i + k})
                                    for k, (x, y) in enumerate(zip(xs[i:i + batch], ys[i:i + batch]))])
        for attempt in range(1, 9):
            try:
                feats = img.sampleRegions(collection=pts, scale=10, geometries=True, tileScale=4).getInfo()["features"]
                break
            except Exception as e:  # noqa: BLE001
                if attempt == 8 or not any(k in str(e) for k in ("Too Many Requests", "concurrency", "429", "timed out")):
                    raise
                time.sleep(45 * attempt)
        f.write_text(json.dumps(feats))
        print(f"  {epoch}: batch at {i} -> {len(feats)} labelled", flush=True)
        return feats

    with ThreadPoolExecutor(max_workers=4) as pool:
        out = [f for part in pool.map(one, range(0, n_points, batch)) for f in part]
    gdf = gpd.GeoDataFrame.from_features(out, crs="EPSG:4326")
    gdf["epoch"] = epoch
    return gdf


def balance(gdf: gpd.GeoDataFrame, per_class: int = 1500, seed: int = 42) -> gpd.GeoDataFrame:
    """Cap each class at per_class samples (rare classes keep all they have)."""
    parts = [d.sample(min(len(d), per_class), random_state=seed) for _, d in gdf.groupby("label")]
    return pd.concat(parts)


def split_built(gdf: gpd.GeoDataFrame, buildings_tif) -> gpd.GeoDataFrame:
    """Assign built samples to dense (1) / sparse (2) from the M0 built fraction raster."""
    with rasterio.open(buildings_tif) as r:
        pts = gdf.to_crs(r.crs)
        vals = np.array([v[0] for v in r.sample(zip(pts.geometry.x, pts.geometry.y), indexes=1)])
    gdf = gdf.copy()
    gdf["built_frac"] = vals
    is_b = gdf["label"] == BUILT_TMP
    gdf.loc[is_b, "label"] = np.where(gdf.loc[is_b, "built_frac"] >= DENSE_BUILT_FRAC, 1, 2)
    return gdf


def ee_classifier(train: pd.DataFrame, bands: list, seed: int = 42) -> ee.Classifier:
    """Train an EE random forest from a client-side table (cheap to re-evaluate per tile)."""
    feats = [ee.Feature(None, {**{b: float(r[b]) for b in bands}, "label": int(r["label"])})
             for _, r in train.iterrows()]
    return ee.Classifier.smileRandomForest(numberOfTrees=300, minLeafPopulation=2, seed=seed).train(
        ee.FeatureCollection(feats), "label", bands
    )


def classify(epoch: str, region: ee.Geometry, clf: ee.Classifier) -> ee.Image:
    """Class map with a 3x3 majority filter as minimum mapping unit."""
    raw = features(epoch, region).classify(clf)
    return raw.focalMode(1, "square", "pixels").rename("class").toByte()


def transitions(a: np.ndarray, b: np.ndarray) -> pd.DataFrame:
    """From-to area matrix in km2 for two co-registered 10 m class arrays."""
    ok = (a > 0) & (b > 0)
    codes = a[ok].astype(int) * 10 + b[ok].astype(int)
    vals, counts = np.unique(codes, return_counts=True)
    m = pd.DataFrame(0.0, index=list(CLASSES.values()), columns=list(CLASSES.values()))
    for v, c in zip(vals, counts):
        m.iloc[v // 10 - 1, v % 10 - 1] = c * 100 / 1e6
    m.index.name = "from \\ to"
    return m


def validation_points(class_tif, n_per_class: int = 50, seed: int = 7) -> gpd.GeoDataFrame:
    """Stratified random points on a class map, for manual interpretation (Google Earth Pro)."""
    rng = np.random.default_rng(seed)
    with rasterio.open(class_tif) as r:
        a = r.read(1)
        rows = []
        for k, name in CLASSES.items():
            idx = np.argwhere(a == k)
            take = idx[rng.choice(len(idx), size=min(n_per_class, len(idx)), replace=False)]
            for i, j in take:
                x, y = r.xy(i, j)
                rows.append({"map_class": k, "map_name": name, "ref_class": None, "geometry": Point(x, y)})
        return gpd.GeoDataFrame(rows, crs=r.crs).to_crs("EPSG:4326")
