"""TimeSen2Crop data loading.

Expected layout (as distributed, already split by tile)::

    <root>/
      train/    <tile>/dates.csv, <tile>/<crop_id 0..15>/<n>.csv   (13 tiles)
      validate/ 33TWN/...
      test/     33UVP/...

* ``dates.csv``: one acquisition date per row (YYYYMMDD), oldest first.
* ``<n>.csv``: one pixel. One row per acquisition date, columns = spectral
  bands (B2 … B12) followed by the pixel-condition flag
  (0 = clear, 1 = cloud, 2 = shadow, 3 = snow).

Each pixel is turned into a fixed-size time series following the protocol of
the dataset paper (Weikmann, Paris & Bruzzone, IEEE JSTARS 2021):
cloud/shadow/snow observations are discarded and the remaining *clear*
observations are aggregated into **12 monthly median composites**
(September 2017 → August 2018). Months without a clear observation are
marked missing. The result is an array of shape (12 months, n_bands).

IMPORTANT: the class label is the *crop folder*, never the Flag column.
"""
from __future__ import annotations

import csv
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# Class order as listed in the TimeSen2Crop description; folder "0" = first entry.
# Check this against Table I of the dataset paper for your copy of the data.
CROP_NAMES = [
    "Legumes", "Grassland", "Maize", "Potato", "Sunflower", "Soy", "Winter Barley", "Winter Caraway",
    "Rye", "Rapeseed", "Beet", "Spring Cereals", "Winter Wheat", "Winter Triticale",
    "Permanent Plantation", "Other Crops",
]
N_MONTHS = 12
FIRST_MONTH = 9  # agronomic year starts in September
MONTH_LABELS = ["Sep", "Oct", "Nov", "Dec", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug"]
CLEAR = 0
SPLITS = ("train", "validate", "test")


# ---------------------------------------------------------------------------
# low-level readers
# ---------------------------------------------------------------------------
def read_dates(tile_dir: Path) -> np.ndarray:
    """Return acquisition dates of a tile as int YYYYMMDD, in file order."""
    with open(tile_dir / "dates.csv", newline="") as f:
        rows = [r for r in csv.reader(f) if r]
    vals = []
    for r in rows:
        tok = r[-1].strip()
        if tok.isdigit() and len(tok) == 8:
            vals.append(int(tok))
    if not vals:
        raise ValueError(f"no YYYYMMDD dates found in {tile_dir / 'dates.csv'}")
    return np.asarray(vals, dtype=np.int64)


def month_index(dates: np.ndarray) -> np.ndarray:
    """YYYYMMDD → position in the agronomic year (Sep = 0 … Aug = 11)."""
    months = (dates // 100) % 100
    return (months - FIRST_MONTH) % 12


def read_header(csv_path: Path) -> List[str]:
    with open(csv_path, newline="") as f:
        return [h.strip() for h in next(csv.reader(f))]


def select_bands(header: Sequence[str]) -> Tuple[List[int], List[str], int]:
    """Pick spectral band columns and the flag column from a pixel-CSV header.

    The flag is the last column. B8 is dropped when B8A is also present
    (the dataset paper uses 9 bands: B2–B7, B8A, B11, B12).
    """
    names = list(header[:-1])
    keep = [i for i, n in enumerate(names) if not (n.upper() == "B8" and "B8A" in (x.upper() for x in names))]
    return keep, [names[i] for i in keep], len(header) - 1


def nir_red_columns(band_names: Sequence[str]) -> Tuple[int, int]:
    """Index of the NIR (B8A, else B8) and red (B4) band within ``band_names``."""
    up = [b.upper() for b in band_names]
    nir = up.index("B8A") if "B8A" in up else up.index("B8")
    return nir, up.index("B4")


def composite(values: np.ndarray, flags: np.ndarray, months: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Monthly median of clear observations.

    values: (T, B) reflectances, flags: (T,), months: (T,) in 0..11.
    Returns (12, B) composite (NaN where a month has no clear observation)
    and (12,) number of clear observations per month.
    """
    out = np.full((N_MONTHS, values.shape[1]), np.nan, dtype=np.float32)
    counts = np.zeros(N_MONTHS, dtype=np.int16)
    clear = flags == CLEAR
    for m in range(N_MONTHS):
        sel = clear & (months == m)
        n = int(sel.sum())
        if n:
            out[m] = np.median(values[sel], axis=0)
            counts[m] = n
    return out, counts


# ---------------------------------------------------------------------------
# tile / split loading
# ---------------------------------------------------------------------------
# Official spatial split of the dataset paper: 1 validation tile, 1 test tile, the rest for training.
VALIDATION_TILES = ("33TWN",)
TEST_TILES = ("33UVP",)


def tiles_for_split(root: os.PathLike, split: str) -> List[Path]:
    """Tile folders of a split.

    Works with both layouts:
      * already split:  <root>/{train,validate,test}/<tile>/
      * as downloaded:  <root>/<tile>/   → official split by tile name; folders whose
        name contains "2019" (the extra agronomic year) are left out.
    """
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}")
    root = Path(root)
    split_dir = root / split
    if split_dir.is_dir():
        tiles = sorted(d for d in split_dir.iterdir() if d.is_dir() and (d / "dates.csv").exists())
    else:
        all_tiles = sorted(d for d in root.iterdir() if d.is_dir() and (d / "dates.csv").exists()
                           and "2019" not in d.name)
        wanted = {"validate": VALIDATION_TILES, "test": TEST_TILES}.get(split)
        if wanted is None:
            tiles = [d for d in all_tiles if d.name not in VALIDATION_TILES + TEST_TILES]
        else:
            tiles = [d for d in all_tiles if d.name in wanted]
    if not tiles:
        raise FileNotFoundError(f"no tile folders with dates.csv for split {split!r} under {root} – expected "
                                f"<root>/{{train,validate,test}}/<tile>/ or <root>/<tile>/")
    return tiles


@dataclass
class SplitData:
    X: np.ndarray        # (N, 12, B) float32, NaN = month without clear observation
    y: np.ndarray        # (N,) int64 crop id
    tile: np.ndarray     # (N,) tile name
    counts: np.ndarray   # (N, 12) clear observations per month
    bands: List[str]

    def __len__(self) -> int:
        return len(self.y)


def _load_crop_dir(args) -> Tuple[np.ndarray, np.ndarray, int]:
    crop_dir, keep, flag_col, months, max_pixels, seed = args
    files = sorted(p for p in Path(crop_dir).glob("*.csv"))
    if max_pixels and len(files) > max_pixels:
        rng = np.random.default_rng(seed)
        files = [files[i] for i in sorted(rng.choice(len(files), max_pixels, replace=False))]
    X, C, skipped = [], [], 0
    for p in files:
        try:
            arr = np.loadtxt(p, delimiter=",", skiprows=1, ndmin=2, dtype=np.float32)
        except ValueError:
            skipped += 1
            continue
        if arr.shape[0] != len(months):  # pixel series must match the tile's dates
            skipped += 1
            continue
        comp, cnt = composite(arr[:, keep], arr[:, flag_col].astype(np.int16), months)
        X.append(comp)
        C.append(cnt)
    if not X:
        return np.empty((0, N_MONTHS, len(keep)), np.float32), np.empty((0, N_MONTHS), np.int16), skipped
    return np.stack(X), np.stack(C), skipped


def load_split(root: os.PathLike, split: str, max_per_class: Optional[int] = None, workers: int = 0,
               seed: int = 0, min_clear_months: int = 1, verbose: bool = True) -> SplitData:
    """Load and composite every tile of one split (``train`` / ``validate`` / ``test``)."""
    tiles = tiles_for_split(root, split)

    jobs, meta = [], []
    band_names: Optional[List[str]] = None
    for tile in tiles:
        months = month_index(read_dates(tile))
        for crop_dir in sorted((d for d in tile.iterdir() if d.is_dir() and d.name.isdigit()), key=lambda d: int(d.name)):
            first = next(crop_dir.glob("*.csv"), None)
            if first is None:
                continue
            keep, names, flag_col = select_bands(read_header(first))
            if band_names is None:
                band_names = names
            elif names != band_names:
                raise ValueError(f"band columns differ in {crop_dir}: {names} vs {band_names}")
            jobs.append((str(crop_dir), keep, flag_col, months, max_per_class, seed + int(crop_dir.name)))
            meta.append((tile.name, int(crop_dir.name)))

    if workers and workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            results = list(ex.map(_load_crop_dir, jobs, chunksize=1))
    else:
        results = [_load_crop_dir(j) for j in jobs]

    Xs, ys, ts, cs, skipped = [], [], [], [], 0
    for (tile_name, crop), (X, C, sk) in zip(meta, results):
        skipped += sk
        if len(X):
            Xs.append(X)
            cs.append(C)
            ys.append(np.full(len(X), crop, dtype=np.int64))
            ts.append(np.full(len(X), tile_name, dtype=object))
    X = np.concatenate(Xs)
    C = np.concatenate(cs)
    y = np.concatenate(ys)
    t = np.concatenate(ts)
    ok = (C > 0).sum(axis=1) >= min_clear_months
    if verbose:
        print(f"[{split}] {len(tiles)} tile(s), {ok.sum():,} pixels kept, "
              f"{(~ok).sum():,} without enough clear months, {skipped:,} unreadable/mismatched files")
    return SplitData(X[ok], y[ok], t[ok], C[ok], band_names or [])


def save_cache(data: SplitData, path: os.PathLike) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, X=data.X, y=data.y, tile=data.tile.astype(str), counts=data.counts,
                        bands=np.array(data.bands))


def load_cache(path: os.PathLike) -> SplitData:
    z = np.load(path, allow_pickle=False)
    return SplitData(z["X"], z["y"], z["tile"], z["counts"], [str(b) for b in z["bands"]])


# ---------------------------------------------------------------------------
# normalisation – statistics come from the TRAINING split only
# ---------------------------------------------------------------------------
@dataclass
class Normalizer:
    mean: np.ndarray  # (B,)
    std: np.ndarray   # (B,)

    @classmethod
    def fit(cls, X_train: np.ndarray) -> "Normalizer":
        flat = X_train.reshape(-1, X_train.shape[-1])
        return cls(np.nanmean(flat, axis=0).astype(np.float32),
                   (np.nanstd(flat, axis=0) + 1e-6).astype(np.float32))

    def transform(self, X: np.ndarray, add_mask: bool = True) -> np.ndarray:
        """Standardise; missing months become 0 (the training mean). With
        ``add_mask`` an extra channel tells the model which months were observed."""
        Z = (X - self.mean) / self.std
        missing = np.isnan(Z).any(axis=-1, keepdims=True)
        Z = np.where(np.isnan(Z), 0.0, Z).astype(np.float32)
        if add_mask:
            Z = np.concatenate([Z, (~missing).astype(np.float32)], axis=-1)
        return Z

    def to_dict(self) -> Dict[str, list]:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}


def class_weights(y: np.ndarray, n_classes: int) -> np.ndarray:
    """'Balanced' weights n / (k · count_c); classes absent from training get weight 0."""
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    present = counts > 0
    w = np.zeros(n_classes)
    w[present] = counts.sum() / (present.sum() * counts[present])
    return w.astype(np.float32)
