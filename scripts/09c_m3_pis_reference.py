"""Module 3: independent reference for informal settlements (replaces hand-labelled points).

Reference: 'Mapping informal settlements in the Global South at 10 m' (Zenodo 21459094, CC BY 4.0,
July 2026), permanent informal settlements (PIS) circa 2020, independent accuracy 96.0 %.
1. agreement of the M3 morphology rule (2021, 2025) with PIS on the 90 m grid
2. Paper 1 headline re-run with PIS boundaries: LST anomaly inside PIS vs formal dense / sparse,
   and the 2017 -> 2025 relative warming inside fixed PIS boundaries
"""
import glob

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import Resampling, reproject

from kampala.config import DATA, OUTPUTS

P = DATA / "processed"
tile = [f for f in glob.glob(str(DATA / "raw" / "zenodo_pis" / "PIS" / "**" / "*.tif"), recursive=True) if "026.9848E_02.6769N" in f][0]
with rasterio.open(P / "m3" / "settlement_groups_t3_2025_90m.tif") as r:
    g3, prof = r.read(1), r.profile
with rasterio.open(P / "m3" / "settlement_groups_t2_2021_90m.tif") as r:
    g2 = r.read(1)
pis = np.zeros(g3.shape, "float32")
with rasterio.open(tile) as s:
    reproject(rasterio.band(s, 1), pis, dst_transform=prof["transform"], dst_crs=prof["crs"], resampling=Resampling.average)
inPIS = pis >= 0.25            # >= 25 % of the 90 m block mapped as permanent informal settlement


def agree(ours):
    tp = (ours & inPIS).sum()
    return dict(ours_km2=ours.sum() * 0.0081, pis_km2=inPIS.sum() * 0.0081, overlap_km2=tp * 0.0081,
                precision=tp / max(ours.sum(), 1), recall=tp / max(inPIS.sum(), 1),
                iou=tp / max((ours | inPIS).sum(), 1))


rows = [dict(layer="M3 rule 2021", **agree(g2 == 1)), dict(layer="M3 rule 2025", **agree(g3 == 1))]
ag = pd.DataFrame(rows); ag.to_csv(OUTPUTS / "tables" / "m3_pis_agreement.csv", index=False)
print(ag.round(3).to_string(index=False))


def block90(a):
    h, w = (a.shape[0] // 3) * 3, (a.shape[1] // 3) * 3
    return np.nanmean(a[:h, :w].reshape(h // 3, 3, w // 3, 3), axis=(1, 3))


lst = {}
for ep, y in (("t1", 2017), ("t2", 2021), ("t3", 2025)):
    with rasterio.open(P / "m2" / f"lst_landsat_{ep}_{y}_30m.tif") as r:
        lst[y] = block90(r.read(2, masked=True).filled(np.nan))
H, W = min(g3.shape[0], lst[2025].shape[0]), min(g3.shape[1], lst[2025].shape[1])
pisc, g3c = inPIS[:H, :W], g3[:H, :W]
groups = {"PIS (independent)": pisc, "formal dense (not PIS)": (g3c == 2) & ~pisc, "sparse built (not PIS)": (g3c == 3) & ~pisc}
out = []
for nm, m in groups.items():
    vals = {y: float(np.nanmean(lst[y][:H, :W][m])) for y in lst}
    out.append(dict(group=nm, km2=m.sum() * 0.0081, **{f"lst_anom_{y}": v for y, v in vals.items()},
                    change_2017_2025=vals[2025] - vals[2017]))
res = pd.DataFrame(out); res.to_csv(OUTPUTS / "tables" / "m3_pis_heat.csv", index=False)
print(res.round(2).to_string(index=False))
