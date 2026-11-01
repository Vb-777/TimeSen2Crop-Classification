import json

import numpy as np
import pytest

from cropclf.data import (CROP_NAMES, Normalizer, class_weights, composite, load_split, month_index, select_bands)
from cropclf.ndvi import density_class, monthly_ndvi, ndvi
from cropclf.synth import write_dataset
from cropclf.train import evaluate, prepare, load_prepared, train_deep, train_rf

# distinct crops: maize (summer), grassland (all year), winter barley, rapeseed
EASY = [2, 1, 6, 9]


@pytest.fixture(scope="module")
def easy_root(tmp_path_factory):
    return write_dataset(tmp_path_factory.mktemp("ts2c"), pixels_per_class=40, classes=EASY, seed=1)


# ---------------------------------------------------------------- data ------
def test_month_index_starts_in_september():
    d = np.array([20170901, 20171231, 20180101, 20180831])
    assert month_index(d).tolist() == [0, 3, 4, 11]


def test_select_bands_drops_b8_when_b8a_present():
    keep, names, flag = select_bands(["B2", "B3", "B4", "B8", "B8A", "B11", "Flag"])
    assert names == ["B2", "B3", "B4", "B8A", "B11"] and flag == 6
    _, names2, _ = select_bands(["B2", "B4", "B8", "Flag"])
    assert names2 == ["B2", "B4", "B8"]           # keep B8 if it is the only NIR band


def test_composite_uses_only_clear_observations():
    values = np.array([[1.0], [100.0], [3.0], [5.0], [7.0]])
    flags = np.array([0, 1, 0, 3, 0])             # 100 is a cloud, 5 is snow
    months = np.array([0, 0, 0, 1, 2])
    comp, counts = composite(values, flags, months)
    assert comp[0, 0] == 2.0                      # median of 1 and 3, cloud ignored
    assert np.isnan(comp[1, 0])                   # only snow in month 1 → missing
    assert comp[2, 0] == 7.0
    assert counts[:3].tolist() == [2, 0, 1]


def test_labels_come_from_crop_folders_not_flags(easy_root):
    tr = load_split(easy_root, "train", verbose=False)
    assert sorted(set(tr.y.tolist())) == sorted(EASY)
    assert tr.X.shape[1:] == (12, 9)
    assert set(tr.tile) == {"32TNT", "33TUM", "33UWP"}


def test_normalizer_is_fit_on_train_only_and_handles_missing(easy_root):
    tr = load_split(easy_root, "train", verbose=False)
    te = load_split(easy_root, "test", verbose=False)
    n = Normalizer.fit(tr.X)
    Ztr = n.transform(tr.X, add_mask=False)
    obs = ~np.isnan(tr.X)
    assert abs(Ztr[obs].mean()) < 0.05            # standardised on train
    Zte = n.transform(te.X)
    assert not np.isnan(Zte).any() and Zte.shape[-1] == 10   # + observed-mask channel
    missing = np.isnan(te.X).any(-1)
    assert (Zte[..., -1][missing] == 0).all()


def test_class_weights_balance_and_ignore_absent_classes():
    w = class_weights(np.array([0, 0, 0, 1]), 3)
    assert w[2] == 0 and w[1] == pytest.approx(3 * w[0])


# ------------------------------------------------------------- metrics ------
def test_evaluate_metrics():
    m = evaluate(np.array([0, 0, 1, 1]), np.array([0, 1, 1, 1]))
    assert m["overall_accuracy"] == 0.75
    assert m["per_class"][CROP_NAMES[1]]["recall"] == 1.0
    assert len(m["confusion_matrix"]) == len(CROP_NAMES)


# -------------------------------------------------------------- models ------
@pytest.mark.parametrize("name", ["tempcnn", "lstm", "transformer"])
def test_model_output_shape(name):
    import torch
    from cropclf.models import build
    m = build(name, 10, 16).eval()
    assert m(torch.randn(5, 12, 10)).shape == (5, 16)


def test_training_learns_and_test_split_is_separate(easy_root, tmp_path):
    prepare(easy_root, tmp_path / "cache", workers=0)
    tr, va, te = load_prepared(tmp_path / "cache")
    r = train_deep("tempcnn", tr, va, te, epochs=15, batch=64, seed=0, verbose=False, device="cpu")
    assert r["test"]["n"] == len(te) and set(te.tile) == {"33UVP"}
    assert r["test"]["overall_accuracy"] > 0.8    # 4 distinct crops → well above 25 % chance
    assert 1 <= r["best_epoch"] <= 15
    rf = train_rf(tr, va, te, trees=50)
    assert rf["test"]["overall_accuracy"] > 0.8
    json.dumps({k: v for k, v in r.items() if k != "_model"})   # results are JSON-serialisable


# ---------------------------------------------------------------- NDVI ------
def test_ndvi_formula_and_classes():
    assert ndvi(np.array([0.5]), np.array([0.1]))[0] == pytest.approx(0.4 / 0.6)
    assert np.isnan(ndvi(np.array([0.0]), np.array([0.0]))[0])
    assert density_class(0.75) == "Dense, vigorous vegetation"
    assert density_class(0.1) == "Bare soil or very sparse"
    assert density_class(float("nan")) is None


def test_monthly_ndvi_phenology(easy_root):
    stats = monthly_ndvi(easy_root, "train")
    assert stats["Maize"]["peak_month"] in ("Jul", "Aug")           # summer crop
    assert stats["Winter Barley"]["peak_month"] in ("Apr", "May")   # winter cereal
    assert stats["Grassland"]["pixels"] == 3 * 40


def test_works_with_downloaded_layout_split_by_tile_name(tmp_path):
    """Zenodo layout <root>/<tile>/… : 33TWN → validate, 33UVP → test, 2019 folder ignored."""
    from cropclf.data import tiles_for_split
    write_dataset(tmp_path, tiles={"": ["32TNT", "33TUM", "33TWN", "33UVP", "2019_33UVP"]},
                  pixels_per_class=2, classes=[0], seed=3)
    names = lambda s: [t.name for t in tiles_for_split(tmp_path, s)]  # noqa: E731
    assert names("train") == ["32TNT", "33TUM"]
    assert names("validate") == ["33TWN"] and names("test") == ["33UVP"]
    assert len(load_split(tmp_path, "test", verbose=False)) == 2
