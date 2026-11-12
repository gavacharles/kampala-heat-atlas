"""Module 5 (new): rooftop solar potential per building and its scalability across Kampala.

Per building (Overture footprints, release 2026-10-21.0):
    roof area (UTM), usable share (0.5 small roofs < 30 m2, 0.65 otherwise; edges, setbacks,
    access), height from Overture or M0 p90, shading flag where surrounding p90 height (M0, 30 m,
    3x3 window) exceeds the building's own by > 6 m (0.85 yield factor).
Irradiance: ERA5-Land monthly surface solar radiation, mean annual 2015-2024 (real cloudiness,
    unlike the clear-sky screening in M0). Panels flat-ish (5-10 deg tilt): M0 showed tilt
    changes annual yield by < 1.5 %, so POA ~= GHI.
Yield: kWh/yr = usable area x GHI x module efficiency (0.21) x performance ratio (0.80) x shading.
Scalability: adoption scenarios (share of suitable roofs, largest-first and random) and the split
    by M3 settlement group (informal-type roofs carry structural and tenure caveats).
"""
import ee
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import Resampling, reproject
from scipy import ndimage

from kampala import config, ee_utils
from kampala.config import DATA, OUTPUTS

EFF, PR, SHADE = 0.21, 0.80, 0.85
M0, M3, M5 = DATA / "processed" / "m0", DATA / "processed" / "m3", DATA / "processed" / "m5"
M5.mkdir(parents=True, exist_ok=True)
crs = config.load()["study_area"]["crs"]

# irradiance (kWh/m2/yr) from ERA5-Land
ghi_tif = M5 / "ghi_era5land_2015_2024.tif"
if not ghi_tif.exists():
    ee_utils.init()
    ext = ee_utils.extended_area()
    ssrd = (ee.ImageCollection("ECMWF/ERA5_LAND/MONTHLY_AGGR").filterDate("2015-01-01", "2025-01-01")
            .select("surface_solar_radiation_downwards_sum").sum().divide(10).divide(3.6e6).rename("ghi_kwh_m2_yr"))
    ee_utils.download(ssrd.clip(ext.buffer(15000)), ghi_tif, ext.buffer(15000), 1000)

b = gpd.read_parquet(DATA / "raw" / "overture_buildings.parquet").to_crs(crs)
b = b[b.geometry.type.isin(["Polygon", "MultiPolygon"])].copy()
b["roof_m2"] = b.geometry.area
b = b[b["roof_m2"].between(10, 50000)]
c = b.geometry.centroid
xy = list(zip(c.x, c.y))


def sample(tif, band=1):
    with rasterio.open(tif) as r:
        return np.array([v[0] for v in r.sample(xy, indexes=band)], dtype=float)


b["ghi"] = sample(ghi_tif)
p90 = sample(M0 / "buildings_2023_30m.tif", 3)
b["height_m"] = b["height"].fillna(pd.Series(b["num_floors"], index=b.index) * 3.0).fillna(pd.Series(p90, index=b.index))
with rasterio.open(M0 / "buildings_2023_30m.tif") as r:
    h = r.read(3, masked=True).filled(0)
    nb_max = ndimage.maximum_filter(h, size=3)
    rows, cols = rasterio.transform.rowcol(r.transform, c.x.values, c.y.values)
    rows = np.clip(rows, 0, h.shape[0] - 1); cols = np.clip(cols, 0, h.shape[1] - 1)
    b["neighbour_p90"] = nb_max[rows, cols]
b["shaded"] = (b["neighbour_p90"] - b["height_m"].fillna(0)) > 6
b["usable_m2"] = b["roof_m2"] * np.where(b["roof_m2"] < 30, 0.5, 0.65)
b["kwh_yr"] = b["usable_m2"] * b["ghi"] * EFF * PR * np.where(b["shaded"], SHADE, 1.0)
b["kwp"] = b["usable_m2"] * EFF  # 1 kWp per ~4.8 m2 at 21 %

# settlement group (M3, 2025)
grp = M3 / "settlement_groups_t3_2025_90m.tif"
if grp.exists():
    names = {1: "informal-type", 2: "formal dense", 3: "sparse built", 4: "non-built"}
    b["settlement"] = pd.Series(sample(grp), index=b.index).map(names)

b.drop(columns="geometry").to_parquet(M5 / "rooftop_solar_buildings.parquet")
tot = dict(buildings=len(b), roof_km2=b.roof_m2.sum() / 1e6, usable_km2=b.usable_m2.sum() / 1e6,
           capacity_GWp=b.kwp.sum() / 1e6, generation_GWh_yr=b.kwh_yr.sum() / 1e6,
           mean_ghi=b.ghi.mean(), shaded_share=b.shaded.mean(),
           specific_yield_kWh_per_kWp=(b.kwh_yr.sum() / b.kwp.sum()))
print(pd.Series(tot).round(3).to_string())

# scalability: adoption scenarios
scen = []
order = b.sort_values("kwh_yr", ascending=False)
for share in (0.10, 0.25, 0.50, 1.00):
    k = int(len(b) * share)
    scen.append(dict(adoption=f"{int(share * 100)}% of roofs", largest_first_GWh=order.kwh_yr.iloc[:k].sum() / 1e6,
                     random_GWh=b.kwh_yr.sum() * share / 1e6, capacity_GWp_largest_first=order.kwp.iloc[:k].sum() / 1e6))
scen = pd.DataFrame(scen)
scen.to_csv(OUTPUTS / "tables" / "m5_solar_scenarios.csv", index=False)
print(scen.round(2).to_string(index=False))
if "settlement" in b:
    by = b.groupby("settlement").agg(buildings=("kwh_yr", "size"), GWh_yr=("kwh_yr", lambda s: s.sum() / 1e6),
                                     mean_roof_m2=("roof_m2", "mean"), shaded_pct=("shaded", lambda s: 100 * s.mean()))
    by.to_csv(OUTPUTS / "tables" / "m5_solar_by_settlement.csv")
    print(by.round(2))
pd.Series(tot).to_csv(OUTPUTS / "tables" / "m5_solar_totals.csv", header=["value"])

# ---- scalability vs demand -------------------------------------------------------------------
# National distribution sales: 4,674 GWh in 2024 (Umeme Annual Report 2024, last full year before the
# concession ended on 31 Mar 2025; 3,201 GWh in 2020, Umeme AR 2020 p.13). No public source splits sales by
# region (Umeme reports Kampala East / Kampala West regions without sales). Study-area share is uncertain:
# 25 % = Kampala+Wakiso+Mukono+Mpigi share of modelled customers in the UEDCL outage study grid;
# 50 % = upper bound (to be replaced with UEDCL Central/Kampala region sales when obtained).
NATIONAL_GWH = 4674.0
demand = pd.DataFrame([dict(benchmark=k, demand_GWh=v) for k, v in {
    "national distribution sales 2024 (Umeme)": NATIONAL_GWH,
    "study area, low (25 % of national)": 0.25 * NATIONAL_GWH,
    "study area, high (50 % of national)": 0.50 * NATIONAL_GWH}.items()])
for _, r in scen.iterrows():
    demand[r["adoption"] + " (largest first) %"] = 100 * r["largest_first_GWh"] / demand["demand_GWh"]
demand.to_csv(OUTPUTS / "tables" / "m5_solar_vs_demand.csv", index=False)
print(demand.round(1).to_string(index=False))
