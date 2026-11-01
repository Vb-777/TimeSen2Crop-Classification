"""Synthetic data in the exact TimeSen2Crop file format.

Used by the tests and for a quick end-to-end demo without the 1-million-pixel
download. Each crop gets a phenology curve (when it greens up, how long, how
green); reflectances follow that curve; clouds, shadows and snow are injected
and flagged; every tile has its own irregular acquisition dates and a small
phenology shift (a simple stand-in for the spatial domain shift between tiles).

The numbers are NOT real and say nothing about real-world accuracy.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from .data import CROP_NAMES

BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8A", "B11", "B12"]

# (peak day after 1 Sep, width in days, max greenness, autumn green-up for winter crops)
PHENOLOGY = {
    "Legumes": (285, 40, 0.75, 0.0), "Grassland": (250, 120, 0.7, 0.35), "Maize": (330, 40, 0.9, 0.0),
    "Potato": (305, 35, 0.8, 0.0), "Sunflower": (315, 35, 0.8, 0.0), "Soy": (335, 35, 0.85, 0.0),
    "Winter Barley": (245, 35, 0.85, 0.3), "Winter Caraway": (270, 45, 0.6, 0.25), "Rye": (255, 35, 0.8, 0.3),
    "Rapeseed": (240, 40, 0.85, 0.45), "Beet": (325, 45, 0.85, 0.0), "Spring Cereals": (280, 30, 0.8, 0.0),
    "Winter Wheat": (265, 35, 0.85, 0.3), "Winter Triticale": (262, 35, 0.83, 0.3),
    "Permanent Plantation": (300, 110, 0.55, 0.2), "Other Crops": (295, 60, 0.6, 0.1),
}
START = dt.date(2017, 9, 1)


def greenness(doy: np.ndarray, peak: float, width: float, vmax: float, autumn: float) -> np.ndarray:
    v = vmax * np.exp(-0.5 * ((doy - peak) / width) ** 2)
    if autumn:
        v += autumn * np.exp(-0.5 * ((doy - 60) / 30) ** 2)  # emergence of winter crops in Oct–Nov
    return np.clip(v + 0.05, 0, 1)


def reflectance(v: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Very simple linear mix of bare soil and green vegetation (values ×10 000 like Sentinel-2 L2A)."""
    soil = np.array([0.07, 0.10, 0.13, 0.16, 0.19, 0.21, 0.24, 0.30, 0.25])
    veg = np.array([0.03, 0.07, 0.03, 0.10, 0.30, 0.40, 0.45, 0.20, 0.10])
    r = (1 - v)[:, None] * soil + v[:, None] * veg
    r = r * rng.normal(1.0, 0.04, size=r.shape)
    return np.clip(r, 0.001, 1.0) * 10000


def make_dates(rng: np.random.Generator, every: int = 5, keep: float = 0.6) -> List[dt.date]:
    days = [d for d in range(0, 365, every) if rng.random() < keep]
    return [START + dt.timedelta(days=d + int(rng.integers(0, 2))) for d in days]


def write_dataset(out: Path, tiles: Dict[str, Sequence[str]] | None = None, pixels_per_class: int = 40,
                  classes: Sequence[int] | None = None, seed: int = 0) -> Path:
    """Write ``out/{train,validate,test}/<tile>/...`` in TimeSen2Crop format."""
    rng = np.random.default_rng(seed)
    tiles = tiles or {"train": ["32TNT", "33TUM", "33UWP"], "validate": ["33TWN"], "test": ["33UVP"]}
    classes = list(range(len(CROP_NAMES))) if classes is None else list(classes)
    out = Path(out)
    for split, names in tiles.items():
        for tile in names:
            tdir = out / split / tile
            tdir.mkdir(parents=True, exist_ok=True)
            dates = make_dates(rng)
            doy = np.array([(d - START).days for d in dates], dtype=float)
            month = np.array([d.month for d in dates])
            with open(tdir / "dates.csv", "w") as f:
                f.write("acquisition_date\n" + "\n".join(d.strftime("%Y%m%d") for d in dates) + "\n")
            shift = rng.normal(0, 6)  # tile-level phenology shift (days)
            for c in classes:
                peak, width, vmax, autumn = PHENOLOGY[CROP_NAMES[c]]
                cdir = tdir / str(c)
                cdir.mkdir(exist_ok=True)
                for i in range(pixels_per_class):
                    v = greenness(doy, peak + shift + rng.normal(0, 8), width * rng.uniform(0.85, 1.15),
                                  vmax * rng.uniform(0.85, 1.05), autumn)
                    refl = reflectance(v, rng)
                    flag = np.zeros(len(doy), dtype=int)
                    u = rng.random(len(doy))
                    flag[u < 0.25] = 1                                   # cloud
                    flag[(u >= 0.25) & (u < 0.32)] = 2                   # shadow
                    winter = np.isin(month, [12, 1, 2])
                    flag[winter & (rng.random(len(doy)) < 0.25)] = 3     # snow
                    refl[flag == 1] = rng.uniform(3500, 6000, size=(int((flag == 1).sum()), len(BANDS)))
                    refl[flag == 2] *= 0.35
                    refl[flag == 3] = rng.uniform(6000, 9000, size=(int((flag == 3).sum()), len(BANDS)))
                    rows = np.column_stack([np.round(refl).astype(int), flag])
                    np.savetxt(cdir / f"{i}.csv", rows, fmt="%d", delimiter=",", header=",".join(BANDS + ["Flag"]),
                               comments="")
    return out
