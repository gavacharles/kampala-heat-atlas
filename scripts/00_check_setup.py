"""Check Earth Engine access and that every dataset the thesis needs is reachable."""
import ee

from kampala import config, ee_utils

ee_utils.init()
cfg = config.load()
core = ee_utils.core_area()
print(f"Kampala core area: {core.area().divide(1e6).getInfo():.1f} km2")

checks = {
    "Sentinel-2 L2A": cfg["sentinel2"]["collection"],
    "Sentinel-2 L1C": cfg["sentinel2"]["fallback_collection"],
    "Cloud Score+": cfg["sentinel2"]["cloud_score_plus"],
    "Landsat 8": cfg["landsat"]["collections"][0],
    "Copernicus DEM": cfg["cube"]["dem"],
    "Open Buildings 2.5D": cfg["cube"]["buildings_temporal"],
}
for name, cid in checks.items():
    n = ee.ImageCollection(cid).filterBounds(core).size().getInfo()
    print(f"  {name:22s} {n:6d} images over Kampala")

# Sentinel-2 L2A coverage per epoch -> decides whether t1 needs the L1C fallback
from kampala import sentinel2
for ep in cfg["epochs"]:
    n = sentinel2.collection(ep, core).size().getInfo()
    n1c = sentinel2.collection(ep, core, use_fallback=True).size().getInfo()
    print(f"  epoch {ep}: {n} L2A / {n1c} L1C dry-season scenes")
