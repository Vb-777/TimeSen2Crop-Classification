"""Command-line interface:  python -m cropclf <command> --help

  synth      write a small synthetic dataset in TimeSen2Crop format (for a quick demo)
  prepare    read the raw CSVs once → cache/{train,validate,test}.npz
  train      train + evaluate one model
  benchmark  train + evaluate all models and write results/RESULTS.md
  ndvi       monthly NDVI phenology per crop → results/ndvi/
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

from .train import MODELS, load_prepared, prepare, save_result, train_deep, train_rf, write_summary


def _run_model(name, tr, va, te, args):
    if name == "rf":
        return train_rf(tr, va, te, seed=args.seed)
    return train_deep(name, tr, va, te, epochs=args.epochs, batch=args.batch, lr=args.lr, patience=args.patience,
                      seed=args.seed, device=args.device, verbose=not args.quiet)


def main(argv=None) -> int:
    warnings.filterwarnings("ignore", message="All-NaN slice")
    warnings.filterwarnings("ignore", message="Mean of empty slice")
    ap = argparse.ArgumentParser(prog="cropclf", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("synth", help="write a synthetic TimeSen2Crop-format dataset")
    s.add_argument("--out", type=Path, default=Path("data/synthetic"))
    s.add_argument("--pixels-per-class", type=int, default=60)
    s.add_argument("--seed", type=int, default=0)

    p = sub.add_parser("prepare", help="composite the raw CSVs and cache them as .npz")
    p.add_argument("--root", type=Path, required=True, help="folder containing train/, validate/, test/")
    p.add_argument("--cache", type=Path, default=Path("cache"))
    p.add_argument("--max-per-class", type=int, default=None,
                   help="sample at most N pixels per crop per tile (quick experiments)")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)

    for name in ("train", "benchmark"):
        t = sub.add_parser(name, help=("train one model" if name == "train" else "train all models"))
        if name == "train":
            t.add_argument("--model", choices=MODELS, required=True)
        else:
            t.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
        t.add_argument("--cache", type=Path, default=Path("cache"))
        t.add_argument("--out", type=Path, default=Path("results"))
        t.add_argument("--epochs", type=int, default=40)
        t.add_argument("--batch", type=int, default=512)
        t.add_argument("--lr", type=float, default=1e-3)
        t.add_argument("--patience", type=int, default=8, help="early-stopping patience (epochs)")
        t.add_argument("--seed", type=int, default=0)
        t.add_argument("--device", default="auto", help="auto | cpu | cuda | mps")
        t.add_argument("--quiet", action="store_true")
        t.add_argument("--note", default="", help="text placed at the top of RESULTS.md")

    n = sub.add_parser("ndvi", help="monthly NDVI phenology per crop")
    n.add_argument("--root", type=Path, required=True)
    n.add_argument("--split", default="test", choices=("train", "validate", "test"))
    n.add_argument("--max-per-class", type=int, default=2000)
    n.add_argument("--out", type=Path, default=Path("results/ndvi"))

    args = ap.parse_args(argv)

    if args.cmd == "synth":
        from .synth import write_dataset
        print("wrote", write_dataset(args.out, pixels_per_class=args.pixels_per_class, seed=args.seed))
    elif args.cmd == "prepare":
        prepare(args.root, args.cache, args.max_per_class, args.workers, args.seed)
    elif args.cmd in ("train", "benchmark"):
        tr, va, te = load_prepared(args.cache)
        print(f"train {len(tr):,} px ({len(set(tr.tile))} tiles) · validate {len(va):,} px "
              f"({', '.join(sorted(set(va.tile)))}) · test {len(te):,} px ({', '.join(sorted(set(te.tile)))})")
        names = [args.model] if args.cmd == "train" else args.models
        results = []
        for name in names:
            r = _run_model(name, tr, va, te, args)
            out = save_result(r, args.out)
            t = r["test"]
            print(f"{name:14} test OA {t['overall_accuracy']:.4f}  kappa {t['kappa']:.4f}  "
                  f"macro-F1 {t['macro_f1']:.4f}  → {out}")
            results.append(r)
        if args.cmd == "benchmark":
            print("summary →", write_summary(args.out, results, args.note))
    elif args.cmd == "ndvi":
        from .ndvi import monthly_ndvi, write_ndvi_report
        stats = monthly_ndvi(args.root, args.split, args.max_per_class)
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "ndvi_stats.json").write_text(json.dumps(stats, indent=2))
        for p in write_ndvi_report(stats, args.out):
            print("wrote", p)
        for crop, d in stats.items():
            print(f"  {crop:22} peak {d['peak_month']}  " +
                  " ".join("  –" if x != x else f"{x:4.2f}" for x in d["median"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
