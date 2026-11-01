"""Training and evaluation.

Protocol (no leakage):
  1. normalisation statistics and class weights are computed on the TRAIN tiles only;
  2. models are trained on the train tiles; the epoch with the best accuracy on the
     VALIDATION tile (33TWN) is kept (early stopping);
  3. the TEST tile (33UVP) is evaluated exactly once, with that saved model.
"""
from __future__ import annotations

import copy
import json
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from .data import CROP_NAMES, Normalizer, SplitData, class_weights, load_cache, load_split, save_cache

MODELS = ("tempcnn", "transformer", "lstm", "weighted-lstm", "rf")


# ---------------------------------------------------------------------------
def prepare(root, cache_dir, max_per_class: Optional[int] = None, workers: int = 4, seed: int = 0) -> None:
    """Read the raw CSVs once and store compact .npz files per split."""
    cache_dir = Path(cache_dir)
    for split in ("train", "validate", "test"):
        t0 = time.time()
        data = load_split(root, split, max_per_class=max_per_class, workers=workers, seed=seed)
        save_cache(data, cache_dir / f"{split}.npz")
        print(f"  cached {split}: {data.X.shape} in {time.time() - t0:.0f}s → {cache_dir / f'{split}.npz'}")


def load_prepared(cache_dir):
    cache_dir = Path(cache_dir)
    return tuple(load_cache(cache_dir / f"{s}.npz") for s in ("train", "validate", "test"))


# ---------------------------------------------------------------------------
def evaluate(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = len(CROP_NAMES)) -> Dict:
    from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score, \
        precision_recall_fscore_support

    labels = np.unique(y_true)  # macro scores over classes present in this split
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    return {
        "n": int(len(y_true)),
        "overall_accuracy": float(accuracy_score(y_true, y_pred)),
        "kappa": float(cohen_kappa_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "per_class": {CROP_NAMES[c]: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]),
                                      "support": int(s[i])} for i, c in enumerate(labels)},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(range(n_classes))).tolist(),
    }


def _torch_device(device: str):
    import torch
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


def _predict(model, X: np.ndarray, device, batch: int = 4096) -> np.ndarray:
    import torch
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = torch.from_numpy(X[i:i + batch]).to(device)
            out.append(model(xb).argmax(1).cpu().numpy())
    return np.concatenate(out) if out else np.empty(0, dtype=np.int64)


def train_deep(name: str, tr: SplitData, va: SplitData, te: SplitData, epochs: int = 40, batch: int = 512,
               lr: float = 1e-3, patience: int = 8, seed: int = 0, device: str = "auto",
               verbose: bool = True) -> Dict:
    import torch
    from torch import nn

    from .models import build, count_parameters

    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = _torch_device(device)
    n_classes = len(CROP_NAMES)
    norm = Normalizer.fit(tr.X)
    Xtr, Xva, Xte = norm.transform(tr.X), norm.transform(va.X), norm.transform(te.X)

    model = build(name, Xtr.shape[-1], n_classes, Xtr.shape[1]).to(dev)
    weighted = name == "weighted-lstm"
    w = torch.tensor(class_weights(tr.y, n_classes), device=dev) if weighted else None
    loss_fn = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=3)

    Xt, yt = torch.from_numpy(Xtr), torch.from_numpy(tr.y)
    g = torch.Generator().manual_seed(seed)
    history, best_acc, best_state, best_epoch, bad = [], -1.0, None, 0, 0
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(Xt), generator=g)
        total, n = 0.0, 0
        for i in range(0, len(perm), batch):
            idx = perm[i:i + batch]
            if len(idx) < 2:  # BatchNorm needs > 1 sample
                continue
            xb, yb = Xt[idx].to(dev), yt[idx].to(dev)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
            n += len(idx)
        val_acc = float((_predict(model, Xva, dev) == va.y).mean())
        history.append({"epoch": epoch, "train_loss": total / max(n, 1), "val_accuracy": val_acc})
        sched.step(val_acc)
        if verbose:
            print(f"  [{name}] epoch {epoch:3d}  train loss {total / max(n, 1):.4f}  val acc {val_acc:.4f}")
        if val_acc > best_acc:
            best_acc, best_state, best_epoch, bad = val_acc, copy.deepcopy(model.state_dict()), epoch, 0
        else:
            bad += 1
            if bad >= patience:
                break
    train_time = time.time() - t0
    model.load_state_dict(best_state)

    result = {
        "model": name,
        "class_weighted_loss": weighted,
        "parameters": count_parameters(model),
        "best_epoch": best_epoch,
        "train_seconds": round(train_time, 1),
        "validation": evaluate(va.y, _predict(model, Xva, dev)),
        "test": evaluate(te.y, _predict(model, Xte, dev)),  # evaluated once, with the best-on-validation model
        "history": history,
        "normalizer": norm.to_dict(),
    }
    result["_model"] = model
    return result


def train_rf(tr: SplitData, va: SplitData, te: SplitData, seed: int = 0, trees: int = 300) -> Dict:
    from sklearn.ensemble import RandomForestClassifier

    norm = Normalizer.fit(tr.X)
    flat = lambda d: norm.transform(d.X).reshape(len(d.y), -1)  # noqa: E731
    t0 = time.time()
    rf = RandomForestClassifier(n_estimators=trees, n_jobs=-1, random_state=seed, class_weight="balanced_subsample")
    rf.fit(flat(tr), tr.y)
    return {
        "model": "rf",
        "class_weighted_loss": True,
        "parameters": int(sum(t.tree_.node_count for t in rf.estimators_)),
        "best_epoch": None,
        "train_seconds": round(time.time() - t0, 1),
        "validation": evaluate(va.y, rf.predict(flat(va))),
        "test": evaluate(te.y, rf.predict(flat(te))),
        "history": [],
    }


# ---------------------------------------------------------------------------
def save_result(result: Dict, out_dir) -> Path:
    out = Path(out_dir) / result["model"]
    out.mkdir(parents=True, exist_ok=True)
    model = result.pop("_model", None)
    if model is not None:
        import torch
        torch.save(model.state_dict(), out / "model.pt")
    (out / "metrics.json").write_text(json.dumps(result, indent=2))
    plot_result(result, out)
    return out


def plot_result(result: Dict, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = np.array(result["test"]["confusion_matrix"], dtype=float)
    present = cm.sum(axis=1) > 0
    norm = np.divide(cm, cm.sum(axis=1, keepdims=True), out=np.zeros_like(cm), where=cm.sum(axis=1, keepdims=True) > 0)
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(norm[present], cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_yticks(range(int(present.sum())), [CROP_NAMES[i] for i in np.where(present)[0]], fontsize=8)
    ax.set_xticks(range(len(CROP_NAMES)), CROP_NAMES, rotation=60, ha="right", fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True (test tile)")
    ax.set_title(f"{result['model']}: test OA {result['test']['overall_accuracy']:.3f}, "
                 f"macro-F1 {result['test']['macro_f1']:.3f}")
    fig.colorbar(im, ax=ax, fraction=0.04, label="row-normalised")
    fig.tight_layout()
    fig.savefig(out / "confusion_matrix.png", dpi=120)
    plt.close(fig)

    if result["history"]:
        h = result["history"]
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.6))
        a1.plot([e["epoch"] for e in h], [e["train_loss"] for e in h], color="#2a78d6")
        a1.set_title("Training loss")
        a1.set_xlabel("epoch")
        a2.plot([e["epoch"] for e in h], [e["val_accuracy"] for e in h], color="#2a78d6")
        a2.axvline(result["best_epoch"], color="#7a7974", ls="--", lw=1)
        a2.set_title("Validation-tile accuracy (dashed: kept epoch)")
        a2.set_xlabel("epoch")
        for a in (a1, a2):
            a.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(out / "training_curves.png", dpi=120)
        plt.close(fig)


def write_summary(results_dir, results, note: str = "") -> Path:
    """Markdown summary of all models, with the dataset paper's numbers for reference."""
    rows = sorted(results, key=lambda r: -r["test"]["overall_accuracy"])
    lines = ["# Results on the TimeSen2Crop test tile", "", note, "",
             "| Model | Test OA | Kappa | Macro-F1 | Weighted-F1 | Val OA | Best epoch | Train time |",
             "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        t, v = r["test"], r["validation"]
        lines.append(f"| {r['model']} | {t['overall_accuracy']:.2%} | {t['kappa']:.3f} | {t['macro_f1']:.2%} | "
                     f"{t['weighted_f1']:.2%} | {v['overall_accuracy']:.2%} | {r['best_epoch'] or '–'} | "
                     f"{r['train_seconds']:.0f} s |")
    lines += ["", f"Test pixels: {rows[0]['test']['n']:,} · validation pixels: {rows[0]['validation']['n']:,}", "",
              "## Per-class F1 on the test tile", "",
              "| Crop | Support | " + " | ".join(r["model"] for r in rows) + " |",
              "|---|---|" + "---|" * len(rows)]
    for crop, d in rows[0]["test"]["per_class"].items():
        lines.append(f"| {crop} | {d['support']:,} | " +
                     " | ".join(f"{r['test']['per_class'][crop]['f1']:.2f}" for r in rows) + " |")
    lines += ["", "Reference (Weikmann et al., IEEE JSTARS 2021, same tiles, 12 monthly composites): test OA "
              "TempCNN 81.71 %, LSTM 83.44 %, Transformer 84.44 %, weighted LSTM 85.39 %."]
    path = Path(results_dir) / "RESULTS.md"
    path.write_text("\n".join(lines) + "\n")
    return path
