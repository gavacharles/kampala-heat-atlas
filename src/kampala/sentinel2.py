"""Module 1 inputs: cloud-free dry-season Sentinel-2 composites + spectral indices."""
import ee

from . import config


def dry_season(col: ee.ImageCollection) -> ee.ImageCollection:
    """Keep only dry-season months (Dec-Feb, Jun-Aug by default)."""
    months = config.load()["dry_season_months"]
    return col.filter(ee.Filter.Or(*[ee.Filter.calendarRange(m, m, "month") for m in months]))


def collection(epoch: str, region: ee.Geometry, use_fallback: bool = False) -> ee.ImageCollection:
    """Dry-season S2 scenes for an epoch, masked with Cloud Score+."""
    cfg = config.load()
    s2 = cfg["sentinel2"]
    ep = cfg["epochs"][epoch]
    coll_id = s2["fallback_collection"] if use_fallback else s2["collection"]
    cs = ee.ImageCollection(s2["cloud_score_plus"])
    thr = s2["cs_threshold"]

    col = (
        ee.ImageCollection(coll_id)
        .filterBounds(region)
        .filterDate(ep["start"], ep["end"])
        .linkCollection(cs, ["cs_cdf"])
    )
    col = dry_season(col)
    return col.map(
        lambda img: img.updateMask(img.select("cs_cdf").gte(thr))
        .select(s2["bands"])
        .divide(10000)
        .copyProperties(img, ["system:time_start"])
    )


def add_indices(img: ee.Image) -> ee.Image:
    """Indices used for land-cover separation and thermal drivers."""
    ndvi = img.normalizedDifference(["B8", "B4"]).rename("NDVI")
    ndbi = img.normalizedDifference(["B11", "B8"]).rename("NDBI")
    mndwi = img.normalizedDifference(["B3", "B11"]).rename("MNDWI")
    ndre = img.normalizedDifference(["B8A", "B5"]).rename("NDRE")
    bsi = img.expression(
        "((S + R) - (N + B)) / ((S + R) + (N + B))",
        {"S": img.select("B11"), "R": img.select("B4"), "N": img.select("B8"), "B": img.select("B2")},
    ).rename("BSI")
    return img.addBands([ndvi, ndbi, mndwi, ndre, bsi])


def composite(epoch: str, region: ee.Geometry, use_fallback: bool = False) -> ee.Image:
    """Median composite + indices + valid-observation count (quality layer)."""
    col = collection(epoch, region, use_fallback)
    comp = add_indices(col.median())
    count = col.select("B4").count().rename("n_obs")
    return comp.addBands(count).clip(region).toFloat()
