"""Module 1b: Siamese semantic-change U-Net (PyTorch, Apple MPS) vs the random forest.

Inputs         Planetary Computer stacks (scripts/21b), 30 x 30 km window over Kampala and its fringe.
Semantic head  one U-Net, shared weights across epochs, 6 Sentinel-2 bands -> 7 classes per epoch.
Change head    features of two epochs -> P(new building).
Labels         weak labels per epoch = Impact Observatory annual land cover (independent of the Dynamic
               World labels that trained the RF) with the RF pipeline's building-evidence rules:
               built -> dense / sparse by Open Buildings built share (>= 0.30); built without building
               evidence within ~30 m -> bare soil. Change label: Open Buildings 2.5D built share
               2016 -> 2023 rises by >= 0.15 in the 30 m cell (new building).
Split          spatial checkerboard of 4 km blocks; one block in five held out, never seen in training.
Evaluation     held-out blocks only, both models: agreement with the weak labels (OA, kappa, macro-F1);
               built-up vs Open Buildings presence (independent of Sentinel-2); new building 2017 -> 2025
               vs the Open Buildings change label (precision, recall, F1).
"""
import time

import numpy as np
import pandas as pd
import rasterio
import torch
import torch.nn as nn
import torch.nn.functional as F
from rasterio.warp import Resampling, reproject
from scipy import ndimage

from kampala import config
from kampala.config import DATA, OUTPUTS

torch.manual_seed(0); np.random.seed(0)
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
P, OUT, TAB = DATA / "processed", DATA / "processed" / "m1b", OUTPUTS / "tables"
EP = config.load()["epochs"]; EPS = list(EP)
BY = {"t1": 2016, "t2": 2020, "t3": 2023}
NCLS, PATCH, BLOCK = 7, 128, 400
NAMES = list(config.load()["landcover_classes"].values())


def read(path, band=None):
    with rasterio.open(path) as r:
        return (r.read() if band is None else r.read(band)), r.profile


def to_grid(arr, src_prof, prof, how=Resampling.nearest):
    out = np.zeros((prof["height"], prof["width"]), arr.dtype)
    reproject(arr, out, src_transform=src_prof["transform"], src_crs=src_prof["crs"], dst_transform=prof["transform"],
              dst_crs=prof["crs"], resampling=how)
    return out


# ---------------------------------------------------------------- data
X, Y, prof = {}, {}, None
for ep in EPS:
    x, prof = read(OUT / f"s2_{ep}.tif")
    X[ep] = np.clip(x.astype("float32") / 10000.0, 0, 1)
    io, _ = read(OUT / f"io_{ep}.tif", 1)
    bf, bprof = read(P / "m0" / f"buildings_{BY[ep]}_30m.tif", 1)
    bf = np.nan_to_num(bf)
    bf10 = to_grid(bf.astype("float32"), bprof, prof)
    ev10 = to_grid((ndimage.maximum_filter(bf, 3) > 0.02).astype("uint8"), bprof, prof).astype(bool)
    lab = np.full(io.shape, -1, "int64")
    for k in (3, 4, 5, 6, 7):
        lab[io == k] = k - 1
    built = io == 9
    lab[built & ev10 & (bf10 >= 0.30)] = 0
    lab[built & ev10 & (bf10 < 0.30)] = 1
    lab[built & ~ev10] = 4
    Y[ep] = lab
bf16, bp = read(P / "m0" / "buildings_2016_30m.tif", 1); bf23, _ = read(P / "m0" / "buildings_2023_30m.tif", 1)
chg = to_grid(((np.nan_to_num(bf23) - np.nan_to_num(bf16)) >= 0.15).astype("uint8"), bp, prof).astype("int64")
H, W = Y["t1"].shape
bi, bj = np.arange(H)[:, None] // BLOCK, np.arange(W)[None, :] // BLOCK
held = ((bi * 7 + bj * 3) % 5) == 0
print(f"grid {H}x{W}, held-out share {held.mean():.2f}, device {DEV}", flush=True)


# ---------------------------------------------------------------- model
def block(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                         nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class UNet(nn.Module):
    def __init__(self, cin=6, c=32):
        super().__init__()
        self.e1, self.e2, self.e3, self.e4 = block(cin, c), block(c, 2 * c), block(2 * c, 4 * c), block(4 * c, 8 * c)
        self.u3, self.d3 = nn.ConvTranspose2d(8 * c, 4 * c, 2, 2), block(8 * c, 4 * c)
        self.u2, self.d2 = nn.ConvTranspose2d(4 * c, 2 * c, 2, 2), block(4 * c, 2 * c)
        self.u1, self.d1 = nn.ConvTranspose2d(2 * c, c, 2, 2), block(2 * c, c)
        self.sem = nn.Conv2d(c, NCLS, 1)
        self.chg = nn.Sequential(block(2 * c, c), nn.Conv2d(c, 1, 1))

    def features(self, x):
        e1 = self.e1(x); e2 = self.e2(F.max_pool2d(e1, 2)); e3 = self.e3(F.max_pool2d(e2, 2)); e4 = self.e4(F.max_pool2d(e3, 2))
        d3 = self.d3(torch.cat([self.u3(e4), e3], 1)); d2 = self.d2(torch.cat([self.u2(d3), e2], 1))
        return self.d1(torch.cat([self.u1(d2), e1], 1))

    def forward(self, xa, xb):
        fa, fb = self.features(xa), self.features(xb)
        return self.sem(fa), self.sem(fb), self.chg(torch.cat([fa, fb], 1)).squeeze(1)


def sample_patches(n):
    out = []
    while len(out) < n:
        i, j = np.random.randint(0, H - PATCH), np.random.randint(0, W - PATCH)
        if not held[i:i + PATCH, j:j + PATCH].any():
            out.append((i, j))
    return out


def batch(coords, ea, eb):
    xa = np.stack([X[ea][:, i:i + PATCH, j:j + PATCH] for i, j in coords])
    xb = np.stack([X[eb][:, i:i + PATCH, j:j + PATCH] for i, j in coords])
    ya = np.stack([Y[ea][i:i + PATCH, j:j + PATCH] for i, j in coords])
    yb = np.stack([Y[eb][i:i + PATCH, j:j + PATCH] for i, j in coords])
    yc = np.stack([chg[i:i + PATCH, j:j + PATCH] for i, j in coords]) if (ea, eb) == ("t1", "t3") else None
    if np.random.rand() < 0.5:
        xa, xb, ya, yb = xa[..., ::-1], xb[..., ::-1], ya[..., ::-1], yb[..., ::-1]
        yc = yc[..., ::-1] if yc is not None else None
    t = lambda a, dt: torch.as_tensor(np.ascontiguousarray(a), dtype=dt, device=DEV)
    return t(xa, torch.float32), t(xb, torch.float32), t(ya, torch.long), t(yb, torch.long), (t(yc, torch.float32) if yc is not None else None)


model = UNet().to(DEV)
opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
counts = np.bincount(np.concatenate([Y[e][~held & (Y[e] >= 0)].ravel()[::50] for e in EPS]), minlength=NCLS).astype(float)
cw = torch.as_tensor((counts.sum() / (NCLS * np.maximum(counts, 1))) ** 0.5, dtype=torch.float32, device=DEV)
pairs = [("t1", "t2"), ("t2", "t3"), ("t1", "t3")]
STEPS, BS = 2400, 12
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=STEPS)
t0 = time.time()
for step in range(STEPS):
    model.train()
    ea, eb = pairs[step % 3]
    xa, xb, ya, yb, yc = batch(sample_patches(BS), ea, eb)
    sa, sb, ch = model(xa, xb)
    loss = F.cross_entropy(sa, ya, weight=cw, ignore_index=-1) + F.cross_entropy(sb, yb, weight=cw, ignore_index=-1)
    if yc is not None:
        loss = loss + F.binary_cross_entropy_with_logits(ch, yc, pos_weight=torch.tensor(4.0, device=DEV))
    opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    if step % 200 == 0:
        print(f"step {step} loss {loss.item():.3f} ({time.time() - t0:.0f}s)", flush=True)
torch.save(model.state_dict(), OUT / "unet_siamese.pt")


@torch.no_grad()
def predict(ea, eb, tile=512, pad=32):
    model.eval()
    sem_a = np.zeros((H, W), "uint8"); sem_b = np.zeros((H, W), "uint8"); chp = np.zeros((H, W), "float32")
    for i in range(0, H, tile):
        for j in range(0, W, tile):
            i0, j0, i1, j1 = max(0, i - pad), max(0, j - pad), min(H, i + tile + pad), min(W, j + tile + pad)
            hh, ww = ((i1 - i0 + 7) // 8) * 8, ((j1 - j0 + 7) // 8) * 8
            xa = np.zeros((6, hh, ww), "float32"); xb = np.zeros((6, hh, ww), "float32")
            xa[:, :i1 - i0, :j1 - j0] = X[ea][:, i0:i1, j0:j1]; xb[:, :i1 - i0, :j1 - j0] = X[eb][:, i0:i1, j0:j1]
            sa, sb, ch = model(torch.as_tensor(xa[None], device=DEV), torch.as_tensor(xb[None], device=DEV))
            a = sa.argmax(1)[0].cpu().numpy(); b = sb.argmax(1)[0].cpu().numpy(); c = torch.sigmoid(ch)[0].cpu().numpy()
            oi, oj, ti, tj = i - i0, j - j0, min(tile, H - i), min(tile, W - j)
            sem_a[i:i + ti, j:j + tj] = a[oi:oi + ti, oj:oj + tj] + 1
            sem_b[i:i + ti, j:j + tj] = b[oi:oi + ti, oj:oj + tj] + 1
            chp[i:i + ti, j:j + tj] = c[oi:oi + ti, oj:oj + tj]
    return sem_a, sem_b, chp


maps = {}
s1, s3, c13 = predict("t1", "t3"); maps["t1"], maps["t3"] = s1, s3
_, s2, _ = predict("t1", "t2"); maps["t2"] = s2
p8 = prof.copy(); p8.update(count=1, dtype="uint8", nodata=0, compress="deflate")
for ep, m_ in maps.items():
    with rasterio.open(OUT / f"unet_landcover_{ep}_{EP[ep]['year']}_10m.tif", "w", **p8) as d:
        d.write(m_, 1)
pf = prof.copy(); pf.update(count=1, dtype="float32", nodata=None, compress="deflate")
with rasterio.open(OUT / "unet_newbuilt_prob_2017_2025_10m.tif", "w", **pf) as d:
    d.write(c13, 1)


def metrics(pred, ref, mask):
    p, r = pred[mask].astype(int) - 1, ref[mask].astype(int)
    cm = np.bincount(r * NCLS + p, minlength=NCLS * NCLS).reshape(NCLS, NCLS)
    oa = np.trace(cm) / cm.sum(); pe = (cm.sum(0) * cm.sum(1)).sum() / cm.sum() ** 2
    f1 = [2 * cm[k, k] / max(cm[k].sum() + cm[:, k].sum(), 1) for k in range(NCLS) if cm[k].sum() > 0]
    return oa, (oa - pe) / (1 - pe), float(np.mean(f1))


def prf(pred, ref, mask):
    tp = (pred & ref & mask).sum(); fp = (pred & ~ref & mask).sum(); fn = (~pred & ref & mask).sum()
    p_, r_ = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return p_, r_, 2 * p_ * r_ / max(p_ + r_, 1e-9)


rows = []
for ep in EPS:
    rfa, rfp = read(P / "m1" / f"landcover_{ep}_{EP[ep]['year']}_10m.tif", 1)
    rf = to_grid(rfa, rfp, prof)
    bf, bprof = read(P / "m0" / f"buildings_{BY[ep]}_30m.tif", 1)
    bref = to_grid((np.nan_to_num(bf) >= 0.10).astype("uint8"), bprof, prof).astype(bool)
    m = held & (Y[ep] >= 0)
    for name, pred in (("random forest", rf), ("Siamese U-Net", maps[ep])):
        oa, k, f1 = metrics(pred, Y[ep], m & (pred > 0))
        bp_, br_, bf1 = prf(np.isin(pred, [1, 2]), bref, held & (pred > 0))
        rows.append(dict(epoch=EP[ep]["year"], model=name, weak_label_OA=oa, kappa=k, macro_F1=f1,
                         built_precision_vs_OpenBuildings=bp_, built_recall=br_, built_F1=bf1))
rf1 = to_grid(*read(P / "m1" / "landcover_t1_2017_10m.tif", 1), prof)
rf3 = to_grid(*read(P / "m1" / "landcover_t3_2025_10m.tif", 1), prof)
cref = chg.astype(bool)
for name, pred in (("random forest (post-classification)", ~np.isin(rf1, [1, 2]) & np.isin(rf3, [1, 2])),
                   ("Siamese U-Net (change head)", c13 >= 0.5),
                   ("Siamese U-Net (post-classification)", ~np.isin(s1, [1, 2]) & np.isin(s3, [1, 2]))):
    p_, r_, f_ = prf(pred, cref, held)
    rows.append(dict(epoch="2017-2025", model=name, newbuilt_precision_vs_OpenBuildings=p_, newbuilt_recall=r_,
                     newbuilt_F1=f_, newbuilt_km2_heldout=float((pred & held).sum() * 1e-4)))
res = pd.DataFrame(rows); res.to_csv(TAB / "m1b_unet_vs_rf.csv", index=False)
pd.set_option("display.width", 220)
print(res.round(3).to_string(index=False))
area = pd.DataFrame({EP[e]["year"]: [(maps[e] == k + 1).sum() * 1e-4 for k in range(NCLS)] for e in EPS}, index=NAMES)
area.to_csv(TAB / "m1b_unet_class_area.csv"); print(area.round(1))
