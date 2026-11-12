"""Shared figure style: one palette and one set of map helpers for every thesis figure.

Palette follows the validated reference instance (categorical order, blue<->red diverging with a
gray midpoint, single-hue sequential ramps). Land cover keeps conventional semantic colours
(trees green, water blue) because readers decode those without the legend; it always ships with
a legend.
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
NEUTRAL = "#f0efec"
REDS = ["#fbd5d4", "#f4a3a2", "#ec7372", "#e34948", "#c23434", "#9a2626", "#701b1b"]
ORANGES = ["#fde0d2", "#f8b89c", "#f29068", "#eb6834", "#c95222", "#a1401a", "#7a3013"]
GREENS = ["#d6f0d6", "#a8dca8", "#74c474", "#3fa83f", "#008300", "#006300", "#004700"]

THERMAL = ListedColormap([BLUE[5], BLUE[2], NEUTRAL, REDS[2], REDS[5]])
THERMAL_NAMES = ["very cool", "cool", "moderate", "warm", "hot"]
DIVERGING = LinearSegmentedColormap.from_list("bluered", BLUE[::-1][:6] + [NEUTRAL] + REDS[1:])
LANDCOVER = ["#c23434", "#f29068", "#2f8f3a", "#d6e8a0", "#c9b38b", "#5fb7c4", "#256abf"]
LC_NAMES = ["built dense", "built sparse", "tree cover", "grass / cropland", "bare soil", "wetland", "water"]


def seq(colors, name="seq"):
    return LinearSegmentedColormap.from_list(name, colors)


def style():
    mpl.rcParams.update({
        "font.family": ["Helvetica Neue", "Arial", "DejaVu Sans"],
        "font.size": 10, "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK,
        "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.bbox": "tight",
        "legend.frameon": False, "lines.linewidth": 2, "axes.axisbelow": True,
    })


def read(path, band=1, step=1):
    with rasterio.open(path) as r:
        a = r.read(band, masked=True)[::step, ::step]
        b = r.bounds
    return a, (b.left, b.right, b.bottom, b.top)


def hillshade_under(ax, terrain_tif, extent_like=None, alpha=0.35, step=1):
    hs, ext = read(terrain_tif, 4, step)
    ax.imshow(hs, cmap="gray", extent=ext, alpha=alpha, interpolation="bilinear", zorder=0)
    return ext


def map_axes(ax, title=None, km_bar=True, extent=None):
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    if title:
        ax.set_title(title, loc="left")
    if km_bar and extent:
        x0 = extent[0] + 0.05 * (extent[1] - extent[0]); y0 = extent[2] + 0.04 * (extent[3] - extent[2])
        ax.plot([x0, x0 + 10000], [y0, y0], color=INK, lw=2.5, solid_capstyle="butt", zorder=10)
        ax.text(x0 + 5000, y0 + 0.012 * (extent[3] - extent[2]), "10 km", ha="center", va="bottom",
                color=INK, fontsize=8, zorder=10)


def legend_patches(ax, colors, labels, loc="lower right", title=None, ncol=1):
    h = [mpl.patches.Patch(facecolor=c, edgecolor="white", linewidth=0.5) for c in colors]
    return ax.legend(h, labels, loc=loc, title=title, ncol=ncol, fontsize=8, title_fontsize=8,
                     facecolor="white", framealpha=0.85, frameon=True, edgecolor=GRID)


def masked(a, value=0):
    return np.ma.masked_equal(np.ma.filled(a, value), value)


def cube_arrays(buildings_tif, terrain_tif, lst_tif, cx=451000, cy=34600, half=2700):
    """Terrain, surface (terrain + capped building height) and LST anomaly for a square window
    (UTM 36N), all on the 30 m building grid. Default window: central Kampala, 5.4 x 5.4 km."""
    from rasterio.warp import Resampling, reproject
    from rasterio.windows import from_bounds

    with rasterio.open(buildings_tif) as r:
        win = from_bounds(cx - half, cy - half, cx + half, cy + half, r.transform)
        bf, hp = r.read(1, window=win).astype(float), r.read(3, window=win).astype(float)
        tr = r.window_transform(win)

    def onto(path, band):
        out = np.zeros(bf.shape, dtype="float32")
        with rasterio.open(path) as s:
            reproject(rasterio.band(s, band), out, dst_transform=tr, dst_crs=s.crs, resampling=Resampling.bilinear)
        return out

    z = onto(terrain_tif, 1)
    lst = onto(lst_tif, 2)
    h = np.where(bf > 0.15, np.clip(np.nan_to_num(hp), 0, 45), 0)   # cap noisy heights
    return z, z + h, lst
