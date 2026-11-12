"""Score the hand-labelled validation points (CSV exported from tools/label_points.html).

    python scripts/28_score_validation.py ~/Downloads/kampala_validation_2025.csv

Stratified design (equal points per mapped class) -> accuracies are area-weighted with the mapped
class areas (Olofsson et al. 2014): overall accuracy with a 95 % confidence interval, user's and
producer's accuracy per class, and area-adjusted class areas.
"""
import sys

import numpy as np
import pandas as pd
import rasterio

from kampala import config
from kampala.config import DATA, OUTPUTS

NAMES = list(config.load()["landcover_classes"].values())
df = pd.read_csv(sys.argv[1]).dropna(subset=["ref_class"])
df = df[df.ref_class > 0]
with rasterio.open(DATA / "processed" / "m1" / f"landcover_t3_{config.load()['epochs']['t3']['year']}_10m.tif") as r:
    a = r.read(1)
area = np.array([(a == k).sum() * 1e-4 for k in range(1, 8)])
W = area / area.sum()
n = np.zeros((7, 7))
for m, rr in zip(df.map_class.astype(int), df.ref_class.astype(int)):
    n[m - 1, rr - 1] += 1
ni = n.sum(1)
p = np.where(ni[:, None] > 0, W[:, None] * n / np.maximum(ni[:, None], 1), 0)
oa = np.trace(p)
ua = np.diag(n) / np.maximum(ni, 1)
pa = np.diag(p) / np.maximum(p.sum(0), 1e-12)
var_oa = np.sum(W ** 2 * ua * (1 - ua) / np.maximum(ni - 1, 1))
res = pd.DataFrame({"class": NAMES, "points": ni.astype(int), "users_accuracy": ua, "producers_accuracy": pa,
                    "mapped_km2": area, "area_adjusted_km2": p.sum(0) * area.sum()})
res.to_csv(OUTPUTS / "tables" / "m1_validation_accuracy.csv", index=False)
print(res.round(3).to_string(index=False))
print(f"\noverall accuracy {oa:.3f} ± {1.96 * np.sqrt(var_oa):.3f} (95 % CI), n = {int(n.sum())}")
