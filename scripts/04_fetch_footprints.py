"""Building footprints for the whole study area from Overture Maps (Google + Microsoft + OSM, conflated).

Method and release carried over from kampala-cbd-fire-safety scripts/gis/01_fetch.py (CBD only there);
here extended to the full study area. Used by M3 (settlement morphology) and M5 (rooftop solar).
"""
import duckdb
import geopandas as gpd
from shapely import wkb

from kampala import config
from kampala.config import DATA

OVERTURE_RELEASE = "2026-10-21.0"   # same release as kampala-cbd-fire-safety
out = DATA / "raw" / "overture_buildings.parquet"
w, s, e, n = config.load()["study_area"]["extended_bbox"]

con = duckdb.connect()
con.sql("INSTALL httpfs; LOAD httpfs; SET s3_region='us-west-2';")
df = con.sql(f"""
    SELECT id, geometry, height, num_floors, class, subtype, sources[1].dataset AS source
    FROM read_parquet('s3://overturemaps-us-west-2/release/{OVERTURE_RELEASE}/theme=buildings/type=building/*',
                      hive_partitioning=1)
    WHERE bbox.xmin > {w} AND bbox.xmax < {e} AND bbox.ymin > {s} AND bbox.ymax < {n}
""").df()
geom = df.pop("geometry").apply(lambda b: wkb.loads(bytes(b)))
gdf = gpd.GeoDataFrame(df, geometry=geom, crs="EPSG:4326")
gdf["overture_release"] = OVERTURE_RELEASE
gdf.to_parquet(out)
print("wrote", out.name, len(gdf), "buildings")
print(gdf["source"].value_counts().to_string())
