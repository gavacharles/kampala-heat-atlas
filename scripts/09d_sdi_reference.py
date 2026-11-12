"""Literature reference for informal settlements: the SDI / NSDFU / ACTogether 'Kampala Slum Profiles'
map book (23 Aug 2014), whose settlement boundaries were mapped by the slum dwellers' federation itself.

The five division maps (pages 5-9) store the settlement outlines as vector paths (stroke RGB
0.84, 0.16, 0.18). For each page:
  scale   from the vector scale bar (-1 ... 3 km = 4 km)
  north   up (map grid north; UTM 36N convergence at 32.6 E is < 0.3 deg, ignored)
  offset  from the settlement labels: each label's page position (read off the 170 dpi render) is
          paired with Overture division points of the same name; the translation most labels agree
          on (RANSAC, 350 m inlier tolerance) is taken, then the median of its inliers. Duplicate
          place names across the city are rejected as outliers automatically.
Output: DATA/processed/m3/sdi_2014_settlements.gpkg (one polygon per settlement, page, fit RMS)
and the agreement of the M3 morphology rule with these polygons.
"""
import ee
import fitz
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from scipy.spatial import cKDTree
from shapely.geometry import Polygon, shape
from shapely.ops import unary_union

from kampala import config, ee_utils
from kampala.config import DATA, OUTPUTS

PDF = DATA / "raw" / "literature" / "sdi_kampala_slum_profiles_2014.pdf"
CRS = config.load()["study_area"]["crs"]
RED = (0.84, 0.16, 0.18)
# label centres (x, y) in pixels of the 170 dpi page render, read off each division map, and the
# name used to look the place up (first part of compound names, numbers dropped)
LABELS = {
    5: [("Kamwokya", 860, 279), ("Kagugube", 598, 543), ("Bukesa", 515, 591), ("Kisenyi 1", 620, 748), ("Mengo", 518, 779),
        ("Kisenyi 2", 642, 810), ("Kisenyi 3", 664, 871)],
    6: [("Ssebaggala - Kawempe 2", 363, 234), ("Mpererwe", 772, 284), ("Kawempe 1", 537, 318), ("Kanyanya", 697, 426),
        ("Kazo Angola", 410, 506), ("Bwaise 2", 445, 632), ("Bwaise 1", 615, 623), ("Kyebando-Kisalosalo", 936, 708),
        ("Bwaise 3", 488, 730), ("Makerere 3", 594, 724), ("Kalerwe", 658, 741), ("Nsooba", 729, 731), ("Kifumbira", 840, 780),
        ("Mulago 2", 716, 808), ("Katanga", 720, 949)],
    7: [("Bukoto 1", 304, 370), ("Banda", 932, 493), ("Naguru", 476, 607), ("Kinawataka", 842, 638), ("Luzira", 1083, 1060),
        ("Butabika", 1209, 1027)],
    8: [("Wabigalo", 766, 227), ("Kisugu", 886, 254), ("Kibuli", 700, 291), ("Bukasa-Namuwongo", 1032, 320),
        ("Kabalagala - Kikubamutwe", 838, 361), ("Kabalagala - Kataba", 783, 387), ("Katwe 2", 477, 390), ("Katwe 1", 390, 427),
        ("Nsambya Kevina", 514, 447), ("Nsambya Gogonya", 653, 479), ("Kibuye 1", 489, 507), ("Kibuye 2", 365, 553),
        ("Kansanga", 823, 569), ("Salaama", 665, 933), ("Ggaba", 1250, 1037)],
    9: [("Namungoona", 542, 195), ("Kawaala", 755, 187), ("Kasubi", 729, 291), ("Nankulabye", 784, 396), ("Kosovo", 546, 502),
        ("Namirembe-Bakuli", 904, 576), ("Busega", 178, 653), ("Busega", 346, 714), ("Natete", 427, 779), ("Ndeeba", 888, 862),
        ("Kizito Block - Najjanankumbi 2", 1017, 911), ("Mutundwe", 537, 886), ("Wankulukuku", 743, 978), ("Najjanankumbi", 1022, 1022)],
}
LOOKUP = {"Ssebaggala - Kawempe 2": "Kawempe", "Kyebando-Kisalosalo": "Kyebando", "Bukasa-Namuwongo": "Namuwongo",
          "Kabalagala - Kikubamutwe": "Kabalagala", "Kabalagala - Kataba": "Kabalagala", "Nsambya Kevina": "Nsambya",
          "Nsambya Gogonya": "Nsambya", "Namirembe-Bakuli": "Bakuli", "Kizito Block - Najjanankumbi 2": "Najjanankumbi",
          "Natete": "Nateete", "Makerere 3": "Makerere"}
PX_PER_PT = 170 / 72


def base_name(n):
    n = LOOKUP.get(n, n)
    return " ".join(w for w in n.split() if not w.isdigit())


def overture_places(names):
    import duckdb
    from shapely import wkb
    con = duckdb.connect(); con.sql("INSTALL httpfs; LOAD httpfs; SET s3_region='us-west-2';")
    q = " OR ".join([f"names.primary ILIKE '{n}%'" for n in sorted(set(names))])
    df = con.sql(f"""SELECT names.primary AS name, geometry FROM read_parquet(
      's3://overturemaps-us-west-2/release/2026-10-21.0/theme=divisions/type=division/*', hive_partitioning=1)
      WHERE bbox.xmin > 32.4 AND bbox.xmax < 32.8 AND bbox.ymin > 0.15 AND bbox.ymax < 0.45 AND ({q})""").df()
    g = gpd.GeoDataFrame(df, geometry=df.pop("geometry").apply(lambda x: wkb.loads(bytes(x))), crs="EPSG:4326").to_crs(CRS)
    g["geometry"] = g.geometry.representative_point()
    return g


def page_geometry(page):
    reds, black_pts, bar = [], [], []
    H = page.rect.height
    for d in page.get_drawings():
        col = tuple(round(c, 2) for c in (d.get("color") or (0, 0, 0)))
        pts = []
        for it in d["items"]:
            if it[0] == "l":
                pts += [it[1], it[2]]
            elif it[0] == "c":
                pts += [it[1], it[4]]
            elif it[0] == "re":
                r = it[1]; pts += [r.tl, r.tr, r.br, r.bl]
        xy = np.array([(p.x, H - p.y) for p in pts])      # flip y so north is up
        if len(xy) < 2:
            continue
        if col == RED and len(xy) >= 4:
            poly = Polygon(xy).buffer(0)
            if poly.area > 50:
                reds.append(poly)
        elif col == (0, 0, 0) and (d.get("width") or 0) >= 2.0:
            black_pts.append(xy)
        # scale-bar segments: stroked black, filled black or white, thin, in the bottom strip of the page
        r = d["rect"]
        if (d.get("fill") is not None and d.get("color") is not None and col == (0, 0, 0) and r.height < 8 and r.width > 10 and r.y0 > 0.85 * H):
            bar.append((r.x0, r.x1))
    return reds, (np.vstack(black_pts) if black_pts else np.empty((0, 2))), bar


def densify(xy, step):
    out = []
    for a, b in zip(xy[:-1], xy[1:]):
        n = max(1, int(np.hypot(*(b - a)) / step))
        out.append(a + (b - a) * np.linspace(0, 1, n, endpoint=False)[:, None])
    return np.vstack(out) if out else xy


def main():
    ee_utils.init()
    kla = ee.FeatureCollection("FAO/GAUL/2025/level2").filter(ee.Filter.eq("GAUL2_NAME", "Kampala")).geometry()
    boundary = gpd.GeoSeries([shape(kla.getInfo())], crs="EPSG:4326").to_crs(CRS).iloc[0]
    bxy = np.array(boundary.exterior.coords) if boundary.geom_type == "Polygon" else np.vstack([np.array(g.exterior.coords) for g in boundary.geoms])
    bxy = densify(bxy, 20)
    tree = cKDTree(bxy)

    places = overture_places([base_name(n) for v in LABELS.values() for n, _, _ in v])
    doc = fitz.open(PDF)
    rows, ctrl = [], []
    for pno in range(5, 10):
        page = doc[pno - 1]
        H = page.rect.height
        reds, _, bars = page_geometry(page)
        x0, x1 = min(b[0] for b in bars), max(b[1] for b in bars)
        s_ = 4000.0 / (x1 - x0)                                # metres per PDF unit
        lab = [(n, px / PX_PER_PT, H - py / PX_PER_PT) for n, px, py in LABELS[pno]]
        cands = []                                              # (label index, translation)
        for i, (n, ux, uy) in enumerate(lab):
            bn = base_name(n).lower()
            for _, p in places[places["name"].str.lower().str.startswith(bn)].iterrows():
                cands.append((i, np.array([p.geometry.x - ux * s_, p.geometry.y - uy * s_])))
        best = (0, None)
        for _, t in cands:                                      # RANSAC over candidate translations
            support = {i for i, t2 in cands if np.hypot(*(t2 - t)) < 350}
            if len(support) > best[0]:
                best = (len(support), t)
        inl = [t2 for i, t2 in cands if np.hypot(*(t2 - best[1])) < 350]
        t = np.median(np.array(inl), axis=0)
        resid = float(np.median([np.hypot(*(t2 - t)) for t2 in inl]))
        for i, (n, ux, uy) in enumerate(lab):
            ctrl.append(dict(page=pno, label=n, x=ux * s_ + t[0], y=uy * s_ + t[1]))
        for poly in reds:
            g = Polygon(np.array(poly.exterior.coords) * s_ + t)
            rows.append(dict(page=pno, labels_agreeing=best[0], labels=len(lab), median_resid_m=resid, geometry=g))
        print(f"page {pno}: {len(reds)} outlines, {s_:.1f} m/unit, {best[0]}/{len(lab)} labels agree, median residual {resid:.0f} m", flush=True)
    ctrl = gpd.GeoDataFrame(ctrl, geometry=gpd.points_from_xy([c["x"] for c in ctrl], [c["y"] for c in ctrl]), crs=CRS)

    gdf = gpd.GeoDataFrame(rows, crs=CRS)
    gdf = gdf[gdf.area > 2e4].reset_index(drop=True)
    gdf["name"] = [", ".join(ctrl[(ctrl.page == pg) & ctrl.within(geom)]["label"]) or None for pg, geom in zip(gdf.page, gdf.geometry)]
    out = DATA / "processed" / "m3" / "sdi_2014_settlements.gpkg"
    gdf.to_file(out, driver="GPKG")
    print(f"{len(gdf)} settlement polygons, {gdf.area.sum() / 1e6:.1f} km2 -> {out.name}")

    # agreement with the M3 rule on the 90 m grid, and heat inside the literature polygons
    from rasterio.features import rasterize
    res = []
    for ep, y in (("t2", 2021), ("t3", 2025)):
        with rasterio.open(DATA / "processed" / "m3" / f"settlement_groups_{ep}_{y}_90m.tif") as r:
            g, tr = r.read(1), r.transform
        ref = rasterize(((geom, 1) for geom in gdf.geometry), out_shape=g.shape, transform=tr, fill=0, dtype="uint8").astype(bool)
        kcca = rasterize([(boundary, 1)], out_shape=g.shape, transform=tr, fill=0, dtype="uint8").astype(bool)
        ours = (g == 1) & kcca
        tp = (ours & ref).sum()
        res.append(dict(year=y, ref_km2=ref.sum() * 0.0081, ours_in_kcca_km2=ours.sum() * 0.0081,
                        share_of_ref_flagged_informal=tp / ref.sum(), share_of_ours_inside_ref=tp / max(ours.sum(), 1),
                        ref_share_built_dense_or_informal=((g <= 2) & ref).sum() / ref.sum(),
                        kcca_share_informal=ours.sum() / kcca.sum(), ref_share_informal=tp / ref.sum()))
    ag = pd.DataFrame(res); ag.to_csv(OUTPUTS / "tables" / "m3_sdi_agreement.csv", index=False)
    print(ag.round(3).to_string(index=False))

    with rasterio.open(DATA / "processed" / "m2" / "lst_landsat_t3_2025_30m.tif") as r:
        a25, tr30 = r.read(2, masked=True).filled(np.nan), r.transform
    with rasterio.open(DATA / "processed" / "m2" / "lst_landsat_t1_2017_30m.tif") as r:
        a17 = r.read(2, masked=True).filled(np.nan)
    with rasterio.open(DATA / "processed" / "m0" / "buildings_2023_30m.tif") as r:
        bf = r.read(1, masked=True).filled(0)
    ref30 = rasterize(((geom, 1) for geom in gdf.geometry), out_shape=a25.shape, transform=tr30, fill=0, dtype="uint8").astype(bool)
    kcca30 = rasterize([(boundary, 1)], out_shape=a25.shape, transform=tr30, fill=0, dtype="uint8").astype(bool)
    heat = []
    for nm, m in (("SDI informal settlements (2014)", ref30),
                  ("other built-up in KCCA, same density (built share >= 0.35)", kcca30 & ~ref30 & (bf >= 0.35)),
                  ("other built-up in KCCA (built share >= 0.05)", kcca30 & ~ref30 & (bf >= 0.05))):
        heat.append(dict(group=nm, km2=m.sum() * 9e-4, lst_anom_2017=float(np.nanmean(a17[m])), lst_anom_2025=float(np.nanmean(a25[m])),
                         change=float(np.nanmean(a25[m]) - np.nanmean(a17[m]))))
    ht = pd.DataFrame(heat); ht.to_csv(OUTPUTS / "tables" / "m3_sdi_heat.csv", index=False)
    print(ht.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
