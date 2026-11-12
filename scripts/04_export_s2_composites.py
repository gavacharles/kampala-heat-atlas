"""Module 1: export dry-season Sentinel-2 composites for the three epochs."""
from kampala import config, ee_utils, sentinel2

ee_utils.init()
ext = ee_utils.extended_area()
for ep, spec in config.load()["epochs"].items():
    # L2A coverage over East Africa is patchy before 2018 -> use L1C for t1
    fallback = spec["year"] < 2018
    img = sentinel2.composite(ep, ext, use_fallback=fallback)
    ee_utils.export_image(img, f"m1_s2_{ep}_{spec['year']}{'_L1C' if fallback else ''}", ext, 10)
