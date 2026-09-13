#!/usr/bin/env python3
"""Run the full Calibration Cliff study and write every table to disk.

    python experiments/run_study.py --preset smoke      # ~1 minute, checks wiring
    python experiments/run_study.py --preset main       # the study
    python experiments/run_study.py --data-root data    # real cohorts, same code

Everything downstream - figures, the paper's numbers - reads the parquet/CSV
files this writes, so a result in the manuscript can always be traced back to
the run that produced it.  The run manifest records the config, the package
versions and the elapsed time for exactly that reason.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from analysis import (  # noqa: E402
    collapse_map,
    conformal_table,
    decomposition_table,
    nstar_table,
    recalibration_table,
    transfer_table,
)
from pipeline import StudyConfig, run_pipeline  # noqa: E402

# 'smoke' is a wiring check, not a scientific configuration: its source
# calibration split is far too small to establish source calibration, so its
# unlabeled-correction numbers are not interpretable. Use medium or main.
PRESETS = {
    "smoke":  dict(n_source=1200, n_target=800,  ssl_epochs=2, model_size="tiny",
                   targets=("ptbxl_like", "korea_like")),
    "medium": dict(n_source=4000, n_target=2500, ssl_epochs=5, model_size="small"),
    "main":   dict(n_source=8000, n_target=5000, ssl_epochs=10, model_size="small"),
}


def load_scored(path: Path) -> tuple[dict, list[str]]:
    """Rebuild the scored-site structure from a persisted ``scored.npz``.

    Training and scoring are the expensive half; the tables are the cheap half
    and are the half that changes when an estimator is fixed.  Re-running the
    analysis from stored scores makes every table reproducible without a
    retrain, and means a corrected estimator can be applied to a run that
    already happened rather than forcing a rerun that would also change the
    model.
    """
    z = np.load(path, allow_pickle=True)
    names = [str(s) for s in z["label_names"]]
    sites = [str(s) for s in z["sites"]]
    out: dict[str, dict] = {}
    for k in sites:
        rec = {
            "scores": z[f"{k}_scores"],
            "labels": z[f"{k}_labels"],
            "mask": z[f"{k}_mask"],
            "n": int(len(z[f"{k}_scores"])),
        }
        # Patient identifiers are not persisted; the analysis only uses them to
        # draw calibration sets at the patient level, so each record becomes its
        # own patient here.  That makes the label budget a count of records
        # rather than of patients, which is the conservative direction, and the
        # manifest records that the tables came from stored scores.
        rec["patient_ids"] = np.arange(rec["n"])
        if f"{k}_country" in z:
            rec["country"] = str(z[f"{k}_country"])
        else:
            # Older archives predate the country field; recover it from the site
            # registry rather than leaving "??" in a published table.
            try:
                from ecgcal.sim.generator import SITE_LIBRARY

                rec["country"] = SITE_LIBRARY[k].country
            except (ImportError, KeyError):
                try:
                    from ecgcal.data.registry import get_cohort

                    rec["country"] = get_cohort(k).country
                except KeyError:
                    rec["country"] = "??"
        out[k] = rec
    if "source_ref_scores" in z:
        src = next((k for k in sites if k.startswith("mimic")), sites[0])
        out[src]["source_ref_scores"] = z["source_ref_scores"]
        out[src]["source_ref_labels"] = z["source_ref_labels"]
        out[src]["source_ref_mask"] = np.ones_like(z["source_ref_labels"], bool)
    for k in out:
        out[k].setdefault("is_source", "source_ref_scores" in out[k])
    return out, names


def _save(df: pd.DataFrame, out: Path, name: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f"{name}.csv", index=False)
    try:
        df.to_parquet(out / f"{name}.parquet")
    except Exception:
        pass
    print(f"  wrote {name}: {len(df)} rows")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", default="medium", choices=sorted(PRESETS))
    ap.add_argument("--arm", default="foundation", choices=["foundation", "supervised"])
    ap.add_argument("--source", default="mimic_like")
    ap.add_argument("--data-root", default=None, help="use real cohort bundles from this directory")
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eps", type=float, default=0.02, help="calibration budget for n*")
    ap.add_argument("--alpha", type=float, default=0.1, help="1-alpha is the required coverage")
    ap.add_argument("--skip", nargs="*", default=[], help="tables to skip: recalibration nstar conformal")
    ap.add_argument("--from-scored", type=Path, default=None,
                    help="rebuild every table from a previous run's scored.npz, without retraining")
    args = ap.parse_args(argv)

    cfg = StudyConfig(arm=args.arm, source=args.source, seed=args.seed,
                      data_root=args.data_root, **PRESETS[args.preset])
    out = Path(args.out or ROOT / "results" / f"{args.preset}_{args.arm}")
    t0 = time.time()

    print(f"=== Calibration Cliff: preset={args.preset} arm={args.arm} ===")
    if args.from_scored:
        src = args.from_scored
        src = src / "scored.npz" if src.is_dir() else src
        print(f"[scores] reusing {src} (no retraining)")
        scored, names = load_scored(src)
        run = {"model": type("M", (), {"train_info": {"reused_from": str(src)}})()}
        for k, v in scored.items():
            print(f"  {k:14s} n={v['n']:6d}  source={v['is_source']}")
    else:
        run = run_pipeline(cfg)
        scored, names = run["scored"], run["label_names"]

    # Persist the scored arrays: every figure and every re-analysis then runs
    # from stored scores instead of retraining, which is what makes a number in
    # the manuscript checkable without a GPU-hour.
    out.mkdir(parents=True, exist_ok=True)
    if args.from_scored:
        payload = None
    else:
        payload = {"label_names": np.array(names, dtype=object),
               "sites": np.array(list(scored), dtype=object)}
    if payload is not None:
        for k, v in scored.items():
            payload[f"{k}_scores"] = v["scores"]
            payload[f"{k}_labels"] = v["labels"]
            payload[f"{k}_mask"] = v["mask"]
            payload[f"{k}_country"] = np.array(v["country"], dtype=object)
            if v.get("is_source"):
                payload["source_ref_scores"] = v["source_ref_scores"]
                payload["source_ref_labels"] = v["source_ref_labels"]
        np.savez_compressed(out / "scored.npz", **payload)
        print(f"  wrote scored.npz ({len(scored)} sites)")

    print("\n[1/6] transfer table (discrimination vs calibration)")
    transfer = transfer_table(scored, names, seed=args.seed)
    _save(transfer, out, "transfer")

    print("[2/6] decomposition (Theorem 1)")
    decomp = decomposition_table(scored, names)
    _save(decomp, out, "decomposition")

    recal = pd.DataFrame()
    if "recalibration" not in args.skip:
        print("[3/6] recalibration sweep")
        recal = recalibration_table(scored, names, seed=args.seed)
        _save(recal, out, "recalibration")

    nstar = pd.DataFrame()
    if "nstar" not in args.skip:
        print("[4/6] sample size (Theorem 2)")
        nstar = nstar_table(scored, names, eps=args.eps, alpha=args.alpha, seed=args.seed)
        _save(nstar, out, "nstar")

    conf = pd.DataFrame()
    if "conformal" not in args.skip:
        print("[5/6] conformal coverage (Theorem 3)")
        conf = conformal_table(scored, names, alpha=args.alpha, seed=args.seed)
        _save(conf, out, "conformal")

    print("[6/6] collapse map")
    cmap = collapse_map(transfer, decomp)
    _save(cmap, out, "collapse_map")

    manifest = {
        "config": cfg.to_dict(),
        "preset": args.preset,
        "eps": args.eps,
        "alpha": args.alpha,
        "elapsed_seconds": time.time() - t0,
        "model": run["model"].train_info,
        "sites": {k: {"n": v["n"], "country": v["country"], "is_source": v["is_source"]}
                  for k, v in scored.items()},
        "label_names": names,
        "rows": {"transfer": len(transfer), "decomposition": len(decomp),
                 "recalibration": len(recal), "nstar": len(nstar), "conformal": len(conf)},
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "pandas": pd.__version__, "platform": platform.platform()},
        "data_source": "real_bundles" if cfg.data_root else "simulator",
        "tables_from": str(args.from_scored) if args.from_scored else "this run",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))

    print(f"\n=== done in {manifest['elapsed_seconds']:.1f}s -> {out} ===")
    if not cmap.empty:
        cols = [c for c in ["site", "country", "auroc_mean", "auroc_delta", "ece_mean",
                            "ece_ratio", "free_fraction", "realized_gain"] if c in cmap.columns]
        print("\nCollapse map:")
        print(cmap[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
