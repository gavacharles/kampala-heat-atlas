"""Module 2 inputs: Landsat 8/9 land surface temperature (LST), MODIS day/night LST, LCZ.

Gap-free LST
------------
The Collection-2 surface-temperature product (ST_B10) is blank wherever its ASTER GED emissivity
input is missing; over the study area that leaves only ~60-68 % coverage (the west is mostly
empty), which is why the earlier P3 work fell back on 1 km MODIS. Here LST is derived from TOA
brightness temperature (B10) with NDVI-threshold emissivity (Sobrino et al. 2004), which covers
100 % of the area, and is then calibrated per epoch to the C2 ST product by linear regression on
pixels where both exist (raw agreement r = 0.96; raw bias -5.8 C from missing atmospheric
correction).

Composites use the dry-season months of a three-year window centred on each epoch (+/-12 months),
giving a median of 8-18 clear observations per pixel instead of 1-5.
"""
import ee

from . import config
from .sentinel2 import dry_season

L8L9_TOA = ["LANDSAT/LC08/C02/T1_TOA", "LANDSAT/LC09/C02/T1_TOA"]
L8L9_L2 = ["LANDSAT/LC08/C02/T1_L2", "LANDSAT/LC09/C02/T1_L2"]


def _window(epoch: str):
    cfg = config.load()
    ep, pad = cfg["epochs"][epoch], cfg["landsat"]["pad_months"]
    return ee.Date(ep["start"]).advance(-pad, "month"), ee.Date(ep["end"]).advance(pad, "month")


def _merged(ids, region, start, end):
    col = ee.ImageCollection(ids[0])
    for i in ids[1:]:
        col = col.merge(ee.ImageCollection(i))
    col = col.filterBounds(region).filterDate(start, end).filter(
        ee.Filter.lt("CLOUD_COVER", config.load()["landsat"]["max_cloud"]))
    return dry_season(col)


def _clear(img):
    return img.select("QA_PIXEL").bitwiseAnd(0b11110).eq(0)  # dilated cloud, cirrus, cloud, shadow


def _lst_toa(img: ee.Image) -> ee.Image:
    bt = img.select("B10")  # at-sensor brightness temperature (K)
    ndvi = img.normalizedDifference(["B5", "B4"])
    pv = ndvi.subtract(0.2).divide(0.3).clamp(0, 1).pow(2)
    eps = (ee.Image(0.973).multiply(pv).add(ee.Image(0.966).multiply(ee.Image(1).subtract(pv))).add(0.005)
           .where(ndvi.lt(0.2), 0.966).where(ndvi.lt(0), 0.991))
    lst = bt.divide(ee.Image(1).add(ee.Image(10.895e-6).multiply(bt).divide(1.4388e-2).multiply(eps.log()))).subtract(273.15)
    return lst.updateMask(_clear(img).And(lst.gt(10))).rename("LST_toa").addBands(ndvi.rename("NDVI_L"))


def _lst_st(img: ee.Image) -> ee.Image:
    st = img.select("ST_B10").multiply(0.00341802).add(149.0).subtract(273.15)
    ok = _clear(img).And(img.select("ST_QA").multiply(0.01).lt(4)).And(st.gt(15))
    return st.updateMask(ok).rename("LST_st")


def lst_composite(epoch: str, region: ee.Geometry) -> ee.Image:
    """Bands: LST (calibrated, gap-free), LST_anom (minus area mean), LST_st (C2 product, gappy),
    NDVI_L, n_obs. Calibration coefficients are stored as image properties."""
    start, end = _window(epoch)
    toa = _merged(L8L9_TOA, region, start, end).map(_lst_toa)
    st = _merged(L8L9_L2, region, start, end).map(_lst_st)
    raw = toa.select("LST_toa").median()
    ref = st.median()

    fit = ee.Image.cat([ee.Image(1), raw, ref]).rename(["c", "x", "y"]).reduceRegion(
        ee.Reducer.linearRegression(2, 1), region, scale=300, maxPixels=1e10, tileScale=4)
    coef = ee.Array(fit.get("coefficients")).project([0])
    a, b = ee.Number(coef.get([0])), ee.Number(coef.get([1]))
    lst = raw.multiply(b).add(a).rename("LST")
    mean = lst.reduceRegion(ee.Reducer.mean(), region, scale=90, maxPixels=1e10).get("LST")
    anom = lst.subtract(ee.Number(mean)).rename("LST_anom")
    n = toa.select("LST_toa").count().rename("n_obs")
    out = ee.Image.cat([lst, anom, ref, toa.select("NDVI_L").median(), n]).clip(region).toFloat()
    return out.set({"calib_intercept": a, "calib_slope": b, "area_mean_LST": mean})


def modis_lst(epoch: str, region: ee.Geometry) -> ee.Image:
    """MODIS Aqua (MYD11A2, 1 km) dry-season median day (~13:30) and night (~01:30) LST,
    same three-year window as Landsat. Adds the night view that Landsat cannot give."""
    start, end = _window(epoch)
    col = dry_season(ee.ImageCollection("MODIS/061/MYD11A2").filterDate(start, end))
    to_c = lambda b: col.select(b).median().multiply(0.02).subtract(273.15)
    return ee.Image.cat([to_c("LST_Day_1km").rename("modis_day"),
                         to_c("LST_Night_1km").rename("modis_night")]).clip(region).toFloat()


def lcz(region: ee.Geometry) -> ee.Image:
    """Global Local Climate Zone map (Demuzere et al. 2022, 100 m, ~2018)."""
    return (ee.ImageCollection("RUB/RUBCLIM/LCZ/global_lcz_map/latest").mosaic()
            .select(["LCZ_Filter", "LCZ_Probability"]).clip(region))
