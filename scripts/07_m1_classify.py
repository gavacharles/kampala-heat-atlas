"""Module 1 (new): land-cover classification for one epoch.

    python scripts/07_m1_classify.py t1   # or t2 / t3

1. features + Dynamic World weak labels at 40,000 random points (Earth Engine)
2. built -> dense / sparse with the M0 building raster (local)
3. random forest: hold-out label agreement, confusion matrix, importance (scikit-learn)
4. same training table -> EE random forest -> 10 m class map, tiled download
"""
import json
import sys

import geopandas as gpd
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import cohen_kappa_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

from kampala import config, ee_utils, landcover
from kampala.config import DATA, OUTPUTS

epoch = sys.argv[1]
ee_utils.init()
ext = ee_utils.extended_area()
year = config.load()["epochs"][epoch]["year"]
out = DATA / "processed" / "m1"
out.mkdir(parents=True, exist_ok=True)
tables = OUTPUTS / "tables"

# 1-2 samples
samp_file = out / f"samples_{epoch}.parquet"
if samp_file.exists():
    g = gpd.read_parquet(samp_file)
else:
    g = landcover.sample(epoch, ext, n_points=40000)
    g = landcover.split_built(g, DATA / "processed" / "m0" / f"buildings_{landcover.BUILDING_YEAR[epoch]}_30m.tif")
    g.to_parquet(samp_file)
print("labelled samples per class:", g["label"].value_counts().sort_index().to_dict())

# 3 diagnostics
bands = [c for c in g.columns if c not in ("geometry", "label", "pid", "epoch", "built_frac")]
bal = landcover.balance(g, per_class=1500).dropna(subset=bands)
tr, te = train_test_split(bal, test_size=0.3, stratify=bal["label"], random_state=42)
rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=2, n_jobs=-1, random_state=42)
rf.fit(tr[bands], tr["label"])
pred = rf.predict(te[bands])
labels = sorted(bal["label"].unique())
names = [config.load()["landcover_classes"][k] for k in labels]
cm = pd.DataFrame(confusion_matrix(te["label"], pred, labels=labels), index=names, columns=names)
cm.to_csv(tables / f"m1_{epoch}_confusion_holdout.csv")
diag = {
    "epoch": epoch, "year": year, "n_train": len(tr), "n_test": len(te),
    "label_agreement": float((pred == te["label"]).mean()),
    "kappa": float(cohen_kappa_score(te["label"], pred)),
    "f1_per_class": dict(zip(names, map(float, f1_score(te["label"], pred, labels=labels, average=None)))),
    "note": "agreement with Dynamic World weak labels, not independent map accuracy",
}
(tables / f"m1_{epoch}_diagnostics.json").write_text(json.dumps(diag, indent=2))
pd.Series(rf.feature_importances_, index=bands).sort_values(ascending=False).to_csv(
    tables / f"m1_{epoch}_feature_importance.csv", header=["importance"])
print(json.dumps(diag, indent=2))

# 4 map (EE trained on the balanced training table, capped to keep the request small)
train_ee = landcover.balance(tr, per_class=700)
clf = landcover.ee_classifier(train_ee, bands)
ee_utils.download(landcover.classify(epoch, ext, clf), out / f"landcover_{epoch}_{year}_10m.tif", ext, 10,
                  dtype="uint8", tile_mb=4, tile_dim=256, min_valid=0.97)
