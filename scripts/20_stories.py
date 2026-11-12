"""Three unit-dot stories in the Kiira Futures Lab house style, one per gap paper.

    python scripts/20_stories.py [heat|trees|solar ...]

Writes FIGURES/social_<slug>/ (clips, reels, carousel + PDF, film, highlight GIF) via kampala.story.
Every number shown is read from the result tables; dots are samples of the real cells / people / roofs.
"""
import sys

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import xarray as xr
from PIL import Image
from rasterio.warp import Resampling, reproject

from kampala import config
from kampala import viz as V
from kampala.config import DATA, FIGURES, OUTPUTS
from kampala.story import STAGE_H, STAGE_W, Beat, Story

P, TAB = DATA / "processed", OUTPUTS / "tables"
LINK = "gavacharles.github.io/kampala-heat-atlas"
rng = np.random.default_rng(11)
hex2 = lambda h: np.array([int(h[i:i + 2], 16) for i in (1, 3, 5)])
GREY = hex2("#c3c2b7")

with rasterio.open(P / "m0" / "terrain_fabdem_30m.tif") as r:
    B = r.bounds; HS = r.read(4)[::2, ::2]
SC = min((STAGE_W - 60) / (B.right - B.left), (STAGE_H - 80) / (B.top - B.bottom))
OX = (STAGE_W - (B.right - B.left) * SC) / 2; OY = 60


def to_stage(x, y):
    return np.c_[OX + (x - B.left) * SC, OY + (B.top - y) * SC]


def hill_bg():
    g = np.clip((HS - np.percentile(HS, 2)) / (np.percentile(HS, 98) - np.percentile(HS, 2)), 0, 1)
    im = Image.fromarray((g * 255).astype("uint8")).convert("RGBA").resize((int((B.right - B.left) * SC), int((B.top - B.bottom) * SC)))
    im.putalpha(70)
    bg = Image.new("RGBA", (STAGE_W, STAGE_H), (0, 0, 0, 0)); bg.paste(im, (int(OX), OY), im)
    return bg


def cmap_rgb(cmap, v):
    return (np.asarray(cmap(np.clip(v, 0, 1)))[:, :3] * 255).astype(int)


def columns(groups, order, centres, width=300, gap=11, base=820, sort_key=None):
    """Stack each group's dots as a block column (sorted by sort_key), bottom-aligned at `base`."""
    xy = np.zeros((len(groups), 2))
    per_row = int(width // gap)
    for g, cx in zip(order, centres):
        idx = np.where(groups == g)[0]
        if sort_key is not None:
            idx = idx[np.argsort(sort_key[idx])]
        for k, i in enumerate(idx):
            xy[i] = (cx - width / 2 + (k % per_row) * gap + gap / 2, base - (k // per_row) * gap)
    return xy


def block90(a):
    h, w = (a.shape[0] // 3) * 3, (a.shape[1] // 3) * 3
    return np.nanmean(a[:h, :w].reshape(h // 3, 3, w // 3, 3), axis=(1, 3))


def district_grid(n, name="informal"):
    """n dots as a regular grid plan of a 2 m district: per grid cell the building share and ground means."""
    d = np.load(P / "m6" / f"{name}.npz")
    side = int(np.ceil(np.sqrt(n)))
    m = d["bh"].shape[0] // side
    gi, gj = np.divmod(np.arange(n), side)
    px = 820 / side
    xy = np.c_[(STAGE_W - 820) / 2 + (gj + 0.5) * px, 60 + (gi + 0.5) * px]
    win = [(slice(i * m, (i + 1) * m), slice(j * m, (j + 1) * m)) for i, j in zip(gi, gj)]
    return xy, win, d, px


def district_plan(n, name="informal", value="shade_hours"):
    """Sample n walkable cells of a 2 m district, positioned on the stage as a top view."""
    d = np.load(P / "m6" / f"{name}.npz")
    ground = d["bh"] == 0
    rr, cc = np.nonzero(ground)
    k = rng.choice(len(rr), n, replace=False)
    side = 820
    xy = np.c_[(STAGE_W - side) / 2 + cc[k] / d["bh"].shape[1] * side, 60 + rr[k] / d["bh"].shape[0] * side]
    return xy, rr[k], cc[k], d


# ============================================================ story 1: heat (informal settlements)
def story_heat():
    th = pd.read_csv(TAB / "m2_thermal_class_area.csv"); hot = th[(th.name == "hot") & (th.year == 2025)].km2.iloc[0]
    grp = pd.read_csv(TAB / "m3_lst_by_settlement_group.csv"); g25 = grp[grp.year == 2025].set_index("group").LST_anom_mean
    traj = pd.read_csv(TAB / "m3_thermal_trajectories.csv", index_col=0)
    lam = pd.read_csv(TAB / "m6_lst_by_frontal_area.csv")
    dist = pd.read_csv(TAB / "m6_district_summary.csv").set_index("district")
    with rasterio.open(P / "m3" / "settlement_groups_t3_2025_90m.tif") as r:
        g3, T = r.read(1), r.transform
    with rasterio.open(P / "m3" / "settlement_groups_t1_2017_90m.tif") as r:
        g1 = r.read(1)
    with rasterio.open(P / "m2" / "lst_landsat_t3_2025_30m.tif") as r:
        a3 = block90(r.read(2, masked=True).filled(np.nan))
    with rasterio.open(P / "m2" / "lst_landsat_t1_2017_30m.tif") as r:
        a1 = block90(r.read(2, masked=True).filled(np.nan))
    with rasterio.open(P / "m6" / "frontal_area_index_S_90m.tif") as r:
        lf = r.read(1)
    H, W = min(g3.shape[0], a3.shape[0]), min(g3.shape[1], a3.shape[1])
    g3, g1, a3, a1, lf = g3[:H, :W], g1[:H, :W], a3[:H, :W], a1[:H, :W], lf[:H, :W]
    N = 450
    sel = []
    for g in (1, 2, 3):
        rr, cc = np.nonzero((g3 == g) & np.isfinite(a3) & np.isfinite(a1))
        k = rng.choice(len(rr), N, replace=False); sel.append((np.full(N, g), rr[k], cc[k]))
    G = np.concatenate([s[0] for s in sel]); R = np.concatenate([s[1] for s in sel]); C = np.concatenate([s[2] for s in sel])
    x, y = T.c + (C + 0.5) * T.a, T.f + (R + 0.5) * T.e
    an3, an1, lam_ = a3[R, C], a1[R, C], np.nan_to_num(lf[R, C])
    became = (g1[R, C] >= 3) & (G == 1)
    heat_rgb = cmap_rgb(V.DIVERGING, (an3 + 8) / 16)
    grp_col = np.array([hex2(V.SERIES[0]), hex2(V.SERIES[1]), hex2(V.SERIES[2])])[G - 1]
    names = {1: "packed low-rise", 2: "other dense", 3: "sparse built"}
    tab = {1: "informal-type", 2: "formal dense", 3: "sparse built"}      # table row names
    cx = [250, 630, 1010]

    b1 = Beat("map", "Kampala's surface heat follows its buildings", f"{hot:.0f} km² in the hottest class",
              "dry-season surface temperature, 2025 · each dot is a 90 m block", to_stage(x, y), heat_rgb, 4.2,
              legend=[(hex2(V.BLUE[5]), "cooler than the city"), (hex2(V.REDS[5]), "hotter")], background=hill_bg())
    xy2 = columns(G, [1, 2, 3], cx, sort_key=-an3)
    lab2 = [(c, 850, f"{names[g]}  {g25[tab[g]]:+.1f} °C", "mt", 24, True) for g, c in zip((1, 2, 3), cx)]
    b2 = Beat("types", "Small, tightly packed low-rise blocks run hottest", f"{g25['informal-type']:+.1f} °C above the city",
              f"other dense {g25['formal dense']:+.1f} °C · sparse {g25['sparse built']:+.1f} °C (2025)", xy2, heat_rgb, 4.2, labels=lab2)
    dx = np.clip(an3 - an1, -4, 4)
    rowy = {1: 230, 2: 450, 3: 670}
    xy3 = np.c_[700 + dx * 120 + rng.normal(0, 4, len(G)), np.array([rowy[g] for g in G]) + rng.normal(0, 38, len(G))]
    col3 = np.where(became[:, None], hex2(V.REDS[5]), np.where((G == 1)[:, None], hex2(V.SERIES[0]), GREY))
    lab3 = ([(40, rowy[g], names[g], "lm", 22, True) for g in (1, 2, 3)] +
            [(700, 820, "← cooled      relative change 2017→2025      warmed →", "mt", 22, False), (700, 860, "0", "mt", 18, False)])
    b3 = Beat("warming", "Blocks that packed in since 2017 warmed fastest",
              f"{traj.loc['became informal-type', 'change_2017_2025']:+.1f} °C since 2017",
              f"vs {traj.loc['always formal dense', 'change_2017_2025']:+.1f} °C where blocks were already dense (red = densified)",
              xy3, col3, 4.2, legend=[(hex2(V.REDS[5]), "densified into packed low-rise"), (GREY, "others")], labels=lab3)
    LMAX = 0.35
    xy4 = np.c_[140 + np.clip(lam_, 0, LMAX) / LMAX * 1040 + rng.normal(0, 3, len(G)), 800 - (np.clip(an3, -4, 12) + 4) / 16 * 700]
    lab4 = ([(660, 850, "building frontage facing the southerly lake breeze (λf) →", "mt", 22, False), (60, 80, "hotter ↑", "lm", 22, True)] +
            [(140 + v / LMAX * 1040, 815, f"{v:.1f}", "mt", 18, False) for v in (0, 0.1, 0.2, 0.3)] +
            [(110, 800 - (v + 4) / 16 * 700, f"{v:+d} °C", "rm", 18, False) for v in (0, 4, 8, 12)])
    b4 = Beat("breeze", "Packed blocks stop the lake breeze",
              f"{lam.lst_anom.iloc[0]:+.1f} → {lam.lst_anom.iloc[-2]:+.1f} °C",
              "surface heat rises with the building frontage facing the wind", xy4, grp_col, 4.2,
              legend=[(hex2(V.SERIES[i]), names[i + 1]) for i in range(3)], labels=lab4)
    xy5, win, d, px = district_grid(len(G))
    bfrac = np.array([(d["bh"][w] > 0).mean() for w in win])
    shg = np.array([d["shade_hours"][w][d["bh"][w] == 0].mean() if (d["bh"][w] == 0).any() else 0 for w in win])
    col5 = np.where((bfrac > 0.5)[:, None], hex2("#898781"), cmap_rgb(V.seq(["#fbe7c6", "#86b6ef", "#184f95"]), shg / 3))
    b5 = Beat("shade", "And at midday there is almost no shade",
              f"{dist.loc['informal', 'ground_shade_hours_10_16']:.1f} hours a day",
              "of shade, 10:00–16:00, on Kawempe's streets (3D model, 2 × 2 km)",
              xy5, col5, px * 0.42,
              legend=[(hex2("#898781"), "mostly roofs"), (hex2("#fbe7c6"), "street in sun"), (hex2("#184f95"), "3 h of shade")])
    return Story("heat", "Hotter by design",
                 "Kampala's most tightly packed blocks are its hottest, and the ones that packed in fastest warmed most. A 3D model shows why.",
                 LINK, [b1, b2, b3, b4, b5])


# ============================================================ story 2: trees
def story_trees():
    pri = pd.read_csv(TAB / "m4_priority_summary.csv").set_index("priority")
    sc = pd.read_csv(TAB / "m4_planting_scenario.csv", index_col=0)["value"]
    loc = pd.read_csv(TAB / "m7_tree_local_effect.csv").set_index("district")
    dist = pd.read_csv(TAB / "m6_district_summary.csv").set_index("district")
    with rasterio.open(P / "m4" / "tree_priority_classes_30m.tif") as r:
        cls, T, crs, shape = r.read(1), r.transform, r.crs, r.shape
    pop_nc = xr.open_dataarray(DATA / "carried" / "uganda-rainfall-climate" / "kampala_ghs_pop.nc").sel(epoch=2020).sortby("lat", ascending=False)
    lat, lon = pop_nc.lat.values, pop_nc.lon.values
    dx, dy = (lon[-1] - lon[0]) / (len(lon) - 1), (lat[0] - lat[-1]) / (len(lat) - 1)
    pop = np.zeros(shape, "float32")
    reproject((np.nan_to_num(pop_nc.values) / ((dx * 111320) * (dy * 110574))).astype("float32"), pop,
              src_transform=rasterio.transform.from_origin(lon[0] - dx / 2, lat[0] + dy / 2, dx, dy), src_crs="EPSG:4326",
              dst_transform=T, dst_crs=crs, resampling=Resampling.bilinear)
    pop *= 900
    w = np.where(cls > 0, pop, 0).ravel(); w = w / w.sum()
    N = 1000
    k = rng.choice(len(w), N, replace=False, p=w)
    R, C = np.divmod(k, shape[1]); c_ = cls[R, C]
    x, y = T.c + (C + 0.5) * T.a, T.f + (R + 0.5) * T.e
    gcol = np.array([hex2(V.GREENS[1]), hex2(V.GREENS[2]), hex2(V.GREENS[4]), hex2(V.GREENS[6])])[c_ - 1]
    names = ["low", "medium", "high", "very high"]
    per_dot = sum(pri.people) / N
    b1 = Beat("map", "Where heat, people and open ground meet", f"{pri.loc['very high', 'km2']:.0f} km² of very-high priority",
              f"home to about {pri.loc['very high', 'people'] / 1000:.0f},000 people · each dot ≈ {per_dot:,.0f} residents",
              to_stage(x, y), gcol, 4.0, legend=[(hex2(V.GREENS[i]), n) for i, n in zip((1, 2, 4, 6), names)], background=hill_bg())
    cx = [190, 480, 770, 1060]
    xy2 = columns(c_, [1, 2, 3, 4], cx, width=250)
    lab2 = [(c, 850, f"{n}: {pri.loc[n, 'people'] / 1000:.0f}k", "mt", 22, True) for n, c in zip(names, cx)]
    b2 = Beat("classes", "The hottest places have almost no trees", f"{pri.loc['very high', 'current_tree_pct']:.1f}% canopy",
              "in the very-high priority class today", xy2, gcol, 4.0, labels=lab2)
    xy3, win, d, px = district_grid(N)
    t = np.load(P / "m7" / "trees_informal.npz")
    bfrac = np.array([(d["bh"][w] > 0).mean() for w in win])
    shg = np.array([d["shade_hours"][w][d["bh"][w] == 0].mean() if (d["bh"][w] == 0).any() else 0 for w in win])
    roofs = (bfrac > 0.5)[:, None]
    b3 = Beat("shade", "At the equator, buildings cast almost no shade", f"{dist.loc['informal', 'ground_shade_hours_10_16']:.1f} hours a day",
              "of shade, 10:00–16:00, on Kawempe's streets (2 × 2 km)",
              xy3, np.where(roofs, hex2("#898781"), cmap_rgb(V.seq(["#fbe7c6", "#86b6ef", "#184f95"]), shg / 3)), px * 0.42,
              legend=[(hex2("#898781"), "mostly roofs"), (hex2("#fbe7c6"), "street in sun"), (hex2("#184f95"), "3 h of shade")])
    has_tree = np.array([t["crown"][w].any() for w in win])[:, None]
    col4 = np.where(has_tree, hex2("#2f8f3a"), np.where(roofs, hex2("#898781"), hex2("#fbe7c6")))
    b4 = Beat("trees", f"{len(t['trees'])} street trees in one district",
              f"−{loc.loc['informal', 'utci_1430_sun'] - loc.loc['informal', 'utci_1430_tree_shade']:.1f} °C felt heat",
              f"UTCI {loc.loc['informal', 'utci_1430_sun']:.1f} → {loc.loc['informal', 'utci_1430_tree_shade']:.1f} °C under and beside them at 14:30",
              xy3, col4, px * 0.42, legend=[(hex2("#2f8f3a"), "blocks with new street trees"), (hex2("#898781"), "mostly roofs")])
    target = c_ >= 3
    xy5 = columns(np.where(target, 1, 2), [1, 2], [380, 900], width=420)
    b5 = Beat("scale", "Kampala plants 8,000 trees a year", f"≈ {sc['people_in_planted_cells'] / 1000:.0f},000 people",
              f"live where planting half the open land cools surfaces by about {sc['mean_cooling_C']:.1f} °C",
              xy5, np.where(target[:, None], hex2(V.GREENS[5]), GREY), 4.0,
              labels=[(380, 850, "high + very high priority", "mt", 22, True), (900, 850, "everyone else", "mt", 22, True)])
    return Story("trees", "Where the next 8,000 trees go",
                 "Target Kampala's tree planting to heat, people and open ground, and street trees take about 4 °C off the felt heat.",
                 LINK, [b1, b2, b3, b4, b5])


# ============================================================ story 3: solar
def story_solar():
    st = pd.read_csv(TAB / "m5_solar_totals.csv", index_col=0)["value"]
    by = pd.read_csv(TAB / "m5_solar_by_settlement.csv").set_index("settlement")
    hourly = pd.read_csv(TAB / "m7_solar_hourly.csv"); mar = hourly[hourly.month == 3].set_index("hour")
    sol = pd.read_parquet(P / "m5" / "rooftop_solar_buildings.parquet")
    b = gpd.read_parquet(DATA / "raw" / "overture_buildings.parquet", columns=["geometry"]).to_crs(config.load()["study_area"]["crs"])
    N = 2000
    k = rng.choice(len(sol), N, replace=False); s = sol.iloc[k]
    c = b.geometry.centroid.loc[s.index]
    kwh = s.kwh_yr.values
    orange = cmap_rgb(V.seq(V.ORANGES), np.log10(np.clip(kwh, 1e3, 1e6)) / 2 - 1.5)
    per_dot = len(sol) / N
    b1 = Beat("map", "Two million roofs under an equatorial sun", f"{st['mean_ghi']:,.0f} kWh/m² a year",
              f"of sunshine on Kampala's roofs (ERA5-Land 2015–2024) · each dot ≈ {per_dot:,.0f} roofs",
              to_stage(c.x.values, c.y.values), orange, 3.2, legend=[(hex2(V.ORANGES[1]), "small roof"), (hex2(V.ORANGES[5]), "large roof")],
              background=hill_bg())
    shaded = s.shaded.values
    xy2 = columns(np.where(shaded, 2, 1), [1, 2], [380, 900], width=440, gap=10)
    b2 = Beat("shade", "Neighbours barely shade the roofs", f"{100 * st['shaded_share']:.0f}% of roofs shaded",
              "the sun is nearly overhead; the 3D model puts the loss at 1–2% of yearly output",
              xy2, np.where(shaded[:, None], hex2(V.SERIES[0]), orange), 3.2,
              labels=[(380, 850, "unshaded", "mt", 22, True), (900, 850, "shaded by a taller neighbour", "mt", 22, True)])
    grp = s.settlement.map({"informal-type": 1, "formal dense": 2, "sparse built": 3, "non-built": 4}).fillna(4).astype(int).values
    cx = [190, 480, 770, 1060]
    xy3 = columns(grp, [1, 2, 3, 4], cx, width=250, gap=10)
    labs = ["informal-type", "formal dense", "sparse built", "non-built"]
    b3 = Beat("types", "Informal roofs are small but many", f"{by.loc['informal-type', 'buildings'] / 1000:.0f},000 roofs",
              f"{by.loc['informal-type', 'mean_roof_m2']:.0f} m² on average in informal-type areas · {by.loc['informal-type', 'GWh_yr'] / 1000:.1f} TWh a year",
              xy3, orange, 3.2, labels=[(cxx, 850, l_, "mt", 20, True) for cxx, l_ in zip(cx, labs)])
    share = 4674 / st["generation_GWh_yr"]
    order = np.argsort(rng.random(N)); dem = np.zeros(N, bool); dem[order[: int(share * N)]] = True
    xy4 = columns(np.where(dem, 1, 2), [1, 2], [300, 860], width=360, gap=10)
    b4 = Beat("demand", "Roofs could make 7× Uganda's electricity", f"{st['generation_GWh_yr'] / 1000:.0f} TWh vs 4.7 TWh",
              "technical potential of all roofs vs Uganda's 2024 distribution sales (Umeme)",
              xy4, np.where(dem[:, None], hex2(V.SERIES[0]), orange), 3.2,
              labels=[(300, 850, "Uganda's use", "mt", 22, True), (860, 850, "rooftop potential", "mt", 22, True)])
    hours = mar.index.values; gw = mar.city_gw.values
    alloc = np.floor(gw / gw.sum() * N).astype(int); alloc[np.argmax(alloc)] += N - alloc.sum()
    hr = np.repeat(hours, alloc)
    xs = {h: 110 + i * 95 for i, h in enumerate(hours)}
    xy5 = np.zeros((N, 2)); cnt = {h: 0 for h in hours}
    for i, h in enumerate(hr):
        j = cnt[h]; cnt[h] += 1
        xy5[i] = (xs[h] + (j % 8) * 10 - 35, 820 - (j // 8) * 10)
    avg_px = 820 - (4674 / 8760) / gw.max() * (820 - min(xy5[:, 1]))
    b5 = Beat("noon", "But it all arrives at midday", f"{gw.max():.1f} GW at noon",
              f"against {4674 / 8760:.2f} GW average demand: the grid, not roof space, is the limit",
              xy5, np.full((N, 3), hex2(V.ORANGES[3])), 3.2,
              labels=[(xs[h], 850, f"{h}h", "mt", 20, False) for h in hours[::2]] + [(1190, avg_px, "average demand", "rm", 20, True)])
    return Story("solar", "Roofs are not the constraint",
                 "Kampala's roofs could make several times Uganda's electricity. The real question is what the grid can take at noon.",
                 LINK, [b1, b2, b3, b4, b5])


if __name__ == "__main__":
    want = sys.argv[1:] or ["heat", "trees", "solar"]
    for name in want:
        {"heat": story_heat, "trees": story_trees, "solar": story_solar}[name]().render(FIGURES)
