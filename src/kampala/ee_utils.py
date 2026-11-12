"""Earth Engine helpers: initialisation, study-area geometry, exports."""
import ee

from . import config


def init() -> None:
    project = config.load()["earth_engine"]["project"]
    if project.startswith("YOUR-"):
        raise SystemExit(
            "Set earth_engine.project in config/config.yaml to your Google Cloud "
            "project ID (see https://developers.google.com/earth-engine/guides/access)."
        )
    ee.Initialize(project=project)


def core_area() -> ee.Geometry:
    """Kampala (KCCA) district boundary from FAO GAUL 2025 (~181 km2)."""
    name = config.load()["study_area"]["core_name"]
    return (
        ee.FeatureCollection("FAO/GAUL/2025/level2")
        .filter(ee.Filter.And(ee.Filter.eq("GAUL0_NAME", "Uganda"), ee.Filter.eq("GAUL2_NAME", name)))
        .geometry()
    )


def extended_area() -> ee.Geometry:
    """Core city + peri-urban growth frontier (rectangle from config)."""
    return ee.Geometry.Rectangle(config.load()["study_area"]["extended_bbox"])


def export_image(img: ee.Image, name: str, region: ee.Geometry, scale: int) -> ee.batch.Task:
    """Export an image to Google Drive as a Cloud-Optimised GeoTIFF in UTM 36N."""
    cfg = config.load()
    task = ee.batch.Export.image.toDrive(
        image=img,
        description=name,
        folder=cfg["earth_engine"]["export_folder"],
        fileNamePrefix=name,
        region=region,
        scale=scale,
        crs=cfg["study_area"]["crs"],
        maxPixels=1e13,
        formatOptions={"cloudOptimized": True},
    )
    task.start()
    print(f"started export: {name}")
    return task


def download(img: ee.Image, path, region: ee.Geometry, scale: float, dtype: str = "float32",
             tile_mb: float = 8, tile_dim: int = 512, threads: int = 4, retries: int = 6,
             min_valid: float = 0.0) -> None:
    """Tiled direct download of an EE image to a local GeoTIFF (UTM 36N); skips if present.

    Small tiles keep each request under Earth Engine's interactive memory limit. Few threads and
    retries with back-off are needed because the project runs in EE 'Restricted Mode' (low
    concurrency limit); a failed attempt's partial file is removed before retrying.
    min_valid: required share of valid pixels in band 1 inside the region's bounding box
    (silently failed tiles show up as masked blocks); below it the file is deleted and retried.
    """
    import time
    from pathlib import Path

    import geemap

    path = Path(path)
    if path.exists():
        print(f"exists, skipping: {path.name}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, retries + 1):
        try:
            geemap.download_ee_image(
                img, str(path), region=region, crs=config.load()["study_area"]["crs"], scale=scale,
                dtype=dtype, overwrite=True, max_tile_size=tile_mb, max_tile_dim=tile_dim, num_threads=threads,
            )
            if min_valid > 0:
                import numpy as np
                import rasterio

                with rasterio.open(path) as r:
                    valid = 1 - np.ma.getmaskarray(r.read(1, masked=True)).mean()
                if valid < min_valid:
                    raise RuntimeError(f"Too Many Requests? only {valid:.1%} valid pixels in {path.name}")
            print(f"downloaded: {path.name} ({path.stat().st_size / 1e6:.1f} MB)", flush=True)
            return
        except Exception as e:  # noqa: BLE001 - EE wraps quota errors in several exception types
            path.unlink(missing_ok=True)
            if attempt == retries or not any(k in str(e) for k in ("Too Many Requests", "concurrency", "429", "timed out")):
                raise
            wait = 60 * attempt
            print(f"  {path.name}: EE busy (attempt {attempt}), retrying in {wait}s", flush=True)
            time.sleep(wait)
