"""Vegetation monitoring with NDVI.

NDVI = (NIR − Red) / (NIR + Red), with NIR = B8A (or B8 if that is the column
present) and Red = B4. Only *clear* observations (Flag = 0) are used – a cloud
or snow pixel gives a meaningless NDVI.

For every crop the script computes the **median NDVI per month** over all its
pixels (a phenology curve) and the share of pixels in each vegetation-density
class. The classes describe how much green vegetation covers the pixel; a low
value is normal for bare soil before sowing or after harvest and does **not**
by itself mean an unhealthy crop.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .data import CROP_NAMES, MONTH_LABELS, N_MONTHS, month_index, nir_red_columns, read_dates, read_header, \
    select_bands, tiles_for_split

# Common NDVI interpretation for cropland (thresholds are conventions, not physical constants)
DENSITY_CLASSES = [
    (-1.0, 0.0, "Water / snow / non-vegetated"),
    (0.0, 0.2, "Bare soil or very sparse"),
    (0.2, 0.4, "Sparse vegetation"),
    (0.4, 0.6, "Moderate vegetation"),
    (0.6, 1.01, "Dense, vigorous vegetation"),
]


def ndvi(nir: np.ndarray, red: np.ndarray) -> np.ndarray:
    nir = nir.astype(np.float64)
    red = red.astype(np.float64)
    den = nir + red
    return np.divide(nir - red, den, out=np.full_like(den, np.nan), where=den > 0)


def density_class(v: float) -> Optional[str]:
    if v is None or np.isnan(v):
        return None
    for lo, hi, name in DENSITY_CLASSES:
        if lo <= v < hi:
            return name
    return None


def monthly_ndvi(root, split: str = "test", max_per_class: Optional[int] = 2000, seed: int = 0) -> Dict:
    """Per crop: median NDVI per month (Sep→Aug) over pixels' clear observations."""
    rng = np.random.default_rng(seed)
    per_crop: Dict[int, List[np.ndarray]] = {}
    for tile in tiles_for_split(root, split):
        months = month_index(read_dates(tile))
        for crop_dir in sorted((d for d in tile.iterdir() if d.is_dir() and d.name.isdigit()), key=lambda d: int(d.name)):
            files = sorted(crop_dir.glob("*.csv"))
            if not files:
                continue
            if max_per_class and len(files) > max_per_class:
                files = [files[i] for i in rng.choice(len(files), max_per_class, replace=False)]
            header = read_header(files[0])
            _, names, flag_col = select_bands(header)
            nir_i, red_i = nir_red_columns(header[:-1])
            for p in files:
                a = np.loadtxt(p, delimiter=",", skiprows=1, ndmin=2)
                if a.shape[0] != len(months):
                    continue
                v = ndvi(a[:, nir_i], a[:, red_i])
                v[a[:, flag_col] != 0] = np.nan
                row = np.full(N_MONTHS, np.nan)
                for m in range(N_MONTHS):
                    sel = (months == m) & ~np.isnan(v)
                    if sel.any():
                        row[m] = np.median(v[sel])
                per_crop.setdefault(int(crop_dir.name), []).append(row)

    out = {}
    for c, rows in sorted(per_crop.items()):
        R = np.vstack(rows)
        med = np.nanmedian(np.where(np.isnan(R).all(axis=0), 0, R), axis=0)
        med[np.isnan(R).all(axis=0)] = np.nan
        q25, q75 = np.nanpercentile(R, 25, axis=0), np.nanpercentile(R, 75, axis=0)
        out[CROP_NAMES[c]] = {
            "pixels": len(rows),
            "median": med.tolist(), "q25": q25.tolist(), "q75": q75.tolist(),
            "peak_month": MONTH_LABELS[int(np.nanargmax(med))],
            "status_by_month": [density_class(x) for x in med],
        }
    return out


def write_ndvi_report(stats: Dict, out_dir) -> List[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    # CSV table
    p = out / "ndvi_monthly_median.csv"
    with open(p, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["crop", "pixels"] + MONTH_LABELS + ["peak_month"])
        for crop, d in stats.items():
            w.writerow([crop, d["pixels"]] + [("" if np.isnan(x) else round(x, 3)) for x in d["median"]] +
                       [d["peak_month"]])
    paths.append(p)
    # small multiples: one panel per crop, shared axes
    crops = list(stats)
    cols = 4
    rows = int(np.ceil(len(crops) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(13, 2.6 * rows), sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()
    x = np.arange(N_MONTHS)
    for ax, crop in zip(axes, crops):
        d = stats[crop]
        ax.fill_between(x, d["q25"], d["q75"], color="#2a78d6", alpha=0.18, lw=0)
        ax.plot(x, d["median"], color="#2a78d6", lw=2)
        for lo, _, _ in DENSITY_CLASSES[2:]:
            ax.axhline(lo, color="#c3c2b7", lw=0.8, zorder=0)
        ax.set_title(f"{crop} (n={d['pixels']:,})", fontsize=9)
        ax.set_ylim(-0.1, 1.0)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axes[len(crops):]:
        ax.axis("off")
    for ax in axes[-cols:]:
        ax.set_xticks(x, [m[0] for m in MONTH_LABELS])
    fig.suptitle("Median NDVI per month (Sep 2017 → Aug 2018), clear observations only; band = 25–75 % of pixels",
                 fontsize=11)
    fig.tight_layout()
    p = out / "ndvi_phenology.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    paths.append(p)
    return paths
