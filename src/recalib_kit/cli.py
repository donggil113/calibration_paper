"""Command-line interface: the deployment workflow, in the order it costs money.

    recalib audit    --target scores.csv --source src.csv     # free, unlabeled
    recalib budget   --target scores.csv --source src.csv     # how many labels?
    recalib fix      --target scores.csv --source src.csv     # apply a correction
    recalib export   --scores s.csv --edges edges.json        # federated payload

The order is the point.  ``audit`` needs only unlabeled target scores and tells
a site whether a free correction is licensed; ``budget`` prices the labeled work
if it is not; ``fix`` performs it.  A site should not reach ``fix`` without
having run ``audit``.

Input format is CSV with a ``score`` column and, where labels are needed, a
``label`` column in {0,1}.  Use ``--label-col``/``--score-col`` for other names.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def _read(path: Path, score_col: str, label_col: str | None, require_labels: bool):
    import pandas as pd

    df = pd.read_csv(path)
    if score_col not in df.columns:
        raise SystemExit(f"{path}: no column {score_col!r}; found {list(df.columns)[:12]}")
    s = df[score_col].to_numpy(float)
    y = None
    if label_col and label_col in df.columns:
        y = df[label_col].to_numpy(float)
    elif require_labels:
        raise SystemExit(f"{path}: no label column {label_col!r}, which this command needs")
    return s, y


def _labels_required(method: str) -> int:
    """How many labels a method needs, resolved on an instance."""
    from .recalibrate import METHODS

    cls = METHODS[method]
    try:
        return int(cls().n_labels_required)
    except TypeError:                       # needs pi_s at construction
        return int(cls(pi_s=0.5).n_labels_required)


def _say(as_json: bool, *lines: str) -> None:
    """Print human-facing prose, unless the caller asked for machine output.

    ``--json`` has to mean *only* JSON.  Interleaving guidance with the payload
    makes the output unparseable, which defeats the flag entirely - and the
    guidance is the part a script does not want.
    """
    if as_json:
        return
    for line in lines:
        print(line)


def _emit(obj: dict, as_json: bool) -> None:
    if as_json:
        print(json.dumps(obj, indent=2, default=float))
        return
    width = max(len(k) for k in obj)
    for k, v in obj.items():
        if isinstance(v, float):
            v = f"{v:.6g}"
        print(f"  {k:<{width}}  {v}")


def cmd_audit(args) -> int:
    """Unlabeled: is a free prior correction licensed at this site?"""
    from .decomposition import decompose_unlabeled_budget

    t, _ = _read(args.target, args.score_col, args.label_col, False)
    s, sy = _read(args.source, args.score_col, args.label_col, True)
    pi_s = float(sy.mean())
    b = decompose_unlabeled_budget(t, pi_s, s, sy, n_bins=args.bins)

    # Read as a veto, not a licence.  The test measures distance from the
    # label-shift cone, so it cannot see shift *along* the cone - which is the
    # direction cross-national transfer takes.  A rejection is informative;
    # acceptance is not permission.
    verdict = (
        "NOT REFUTED - label shift is not rejected on your unlabeled data.\n"
        "  This is NOT a licence to apply the unlabeled correction: the test cannot\n"
        "  see shift along the label-shift cone, which is the direction cross-national\n"
        "  shift takes. Decide with labels instead: `recalib fix` (cv_select)."
        if b["bbse_trustworthy"]
        else "REFUTED - this site's shift is not reducible to prevalence.\n"
             "  Do not apply an unlabeled correction here."
    )
    _say(args.json, "", verdict, "")
    _emit({
        "n_target_unlabeled": b["n_target_unlabeled"],
        "prevalence_source": b["pi_s"],
        "prevalence_target_estimated": b["pi_t_bbse"],
        "prevalence_ratio": b["prior_ratio"],
        "label_shift_test_p": b["gof_p_value"],
        "concept_shift_detected": b["concept_shift_detected"],
        "prior_estimate_degenerate": b["bbse_degenerate"],
        "min_concept_shift_gamma": b["gamma_min"],
        "D_label_recoverable_free": b["d_label_l1"],
    }, args.json)
    _say(args.json,
         "",
         "  Next step: `recalib budget` prices the labelled work, then",
         "  `recalib fix` (default --method cv_select) spends those labels first on",
         "  deciding whether your site needs recalibration at all. Fitting a map at a",
         "  site that does not need one made calibration worse in our transfer matrix.")
    return 0


def cmd_budget(args) -> int:
    """How many labeled target cases are needed for a stated calibration budget?"""
    from .recalibrate import prior_correction  # noqa: F401  (documents the dependency)
    from .sample_size import n_star_lower_bound, n_star_upper_plugin, residual_shift_gamma

    t, ty = _read(args.target, args.score_col, args.label_col, False)
    s, sy = _read(args.source, args.score_col, args.label_col, True)
    pi_s = float(sy.mean())

    gamma = args.gamma
    if gamma is None:
        if ty is None:
            raise SystemExit(
                "residual concept shift is unknown: supply --gamma, or give the target file a "
                "label column so it can be measured on a pilot sample"
            )
        gamma = residual_shift_gamma(t, ty, pi_s, float(ty.mean()))

    ub = n_star_upper_plugin(args.eps, args.alpha, t, gamma=gamma)
    lb = n_star_lower_bound(args.eps, 0.25, gamma=gamma)
    out = {
        "calibration_budget_eps": args.eps,
        "confidence_1_minus_alpha": 1 - args.alpha,
        "residual_concept_shift_gamma": gamma,
        "labels_needed_sufficient": ub["n_star"] if ub["feasible"] else "unreachable with a 1-parameter fit",
        "labels_needed_necessary": lb["n_star"],
        "mean_predictive_variance": ub.get("mean_pred_var"),
    }
    zero_suffices = gamma <= args.eps and lb["n_star"] == 0
    out["verdict"] = "zero labels suffice" if zero_suffices else "labels required"
    _say(args.json, "")
    if zero_suffices:
        _say(args.json,
             "ZERO LABELS SUFFICE - the residual shift is inside your calibration budget.", "")
    _emit(out, args.json)
    if args.json:
        return 0
    if zero_suffices:
        # The sufficient count above is what a *labelled* one-parameter fit would
        # cost; quoting it as a plan here would contradict the verdict.
        print("\n  The free correction alone meets the budget. The 'sufficient' figure")
        print("  above is what a labelled one-parameter fit would cost instead, and")
        print("  is the number to plan for only if you want that route anyway.")
    elif ub["feasible"] and isinstance(ub["n_star"], int) and ub["n_star"] > 0:
        print(f"\n  Plan for roughly {ub['n_star']} adjudicated target cases.")
    else:
        print("\n  A one-parameter recalibration cannot reach this budget at this site:")
        print("  the residual concept shift already exceeds it. Use a richer family")
        print("  (isotonic, scaling-binning) or revisit the model.")
    return 0


def cmd_fix(args) -> int:
    """Apply a recalibration and report what it did."""
    from .metrics import expected_calibration_error
    from .recalibrate import METHODS, fit_recalibrator

    if args.method not in METHODS:
        raise SystemExit(f"unknown --method {args.method!r}; choose from {sorted(METHODS)}")
    t, ty = _read(args.target, args.score_col, args.label_col, False)
    s, sy = _read(args.source, args.score_col, args.label_col, True)
    pi_s = float(sy.mean())

    # Resolve on an instance: one estimator's requirement depends on its own
    # configuration, so the attribute is a property there and reading it off the
    # class yields a descriptor rather than a number.
    needs_labels = _labels_required(args.method) > 0
    if needs_labels and ty is None:
        raise SystemExit(f"--method {args.method} needs target labels")

    rec = fit_recalibrator(args.method, t, ty, pi_s=pi_s, source_scores=s, source_labels=sy)
    fixed = rec.transform(t)

    out = {"method": args.method, "n_target": int(t.size),
           "mean_score_before": float(t.mean()), "mean_score_after": float(fixed.mean())}
    if hasattr(rec, "gate_"):
        out["gate_applied"] = rec.gate_.get("applied")
        out["gate_reason"] = rec.gate_.get("reason")
    if hasattr(rec, "selected_"):
        out["selected"] = rec.selected_
        out["cv_scores"] = {k: round(v, 5) for k, v in getattr(rec, "scores_", {}).items()}
    if hasattr(rec, "interval_") and rec.interval_:
        out["identified_interval"] = [rec.interval_.get("pi_lo"), rec.interval_.get("pi_hi")]
        out["prior_used"] = getattr(rec, "pi_t", None)
    if ty is not None:
        out["ece_before"] = expected_calibration_error(t, ty)
        out["ece_after"] = expected_calibration_error(fixed, ty)
    _say(args.json, "")
    _emit(out, args.json)

    if args.out:
        import pandas as pd

        pd.DataFrame({"score_raw": t, "score_calibrated": fixed}).to_csv(args.out, index=False)
        _say(args.json, f"\n  wrote {args.out}")
    return 0


def cmd_export(args) -> int:
    """Reduce a local cohort to the sufficient statistics, and audit the payload."""
    import sys as _sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from ecgcal.data.korea import audit_export, export_sufficient_statistics

    s, y = _read(args.scores, args.score_col, args.label_col, True)
    edges = np.asarray(json.loads(Path(args.edges).read_text()), float)
    stats = export_sufficient_statistics(
        s, y, edges, args.label, args.model_id, cohort=args.cohort
    )
    report = audit_export(stats, min_cell=args.min_cell)
    _say(args.json, "")
    _emit(report, args.json)
    if not report["safe"]:
        _say(args.json,
             "", "  REFUSING to write: merge the flagged bins with a neighbour and re-export.")
        return 2
    Path(args.out).write_text(stats.to_json())
    _say(args.json, f"\n  wrote {args.out} ({len(stats.to_json())} bytes) - safe to transfer")
    return 0


def main(argv=None) -> int:
    # Flags shared by every subcommand are declared on a parent parser as well
    # as on the top level, so they work in either position.  argparse otherwise
    # accepts them only *before* the subcommand, which is the opposite of the
    # order everyone types.
    # SUPPRESS rather than a concrete default: a subparser built from a parent
    # re-applies that parent's defaults after the top-level parse, so a flag
    # given *before* the subcommand would be silently overwritten by the
    # subparser's default. With SUPPRESS an absent flag sets no attribute at
    # all, the top-level value survives, and the real defaults are applied once,
    # below.
    COMMON_DEFAULTS = {"json": False, "score_col": "score", "label_col": "label", "bins": 15}
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--json", action="store_true", default=argparse.SUPPRESS,
                        help="machine-readable output (JSON only, no prose)")
    common.add_argument("--score-col", default=argparse.SUPPRESS)
    common.add_argument("--label-col", default=argparse.SUPPRESS)
    common.add_argument("--bins", type=int, default=argparse.SUPPRESS)

    ap = argparse.ArgumentParser(prog="recalib", description=__doc__,
                                 parents=[common],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("audit", parents=[common],
                       help="unlabeled: is a correction licensed here?")
    a.add_argument("--target", type=Path, required=True)
    a.add_argument("--source", type=Path, required=True)
    a.set_defaults(func=cmd_audit)

    b = sub.add_parser("budget", parents=[common],
                       help="how many labeled target cases are needed?")
    b.add_argument("--target", type=Path, required=True)
    b.add_argument("--source", type=Path, required=True)
    b.add_argument("--eps", type=float, default=0.02)
    b.add_argument("--alpha", type=float, default=0.1)
    b.add_argument("--gamma", type=float, default=None)
    b.set_defaults(func=cmd_budget)

    f = sub.add_parser("fix", parents=[common], help="decide on, and apply, a recalibration")
    f.add_argument("--target", type=Path, required=True)
    f.add_argument("--source", type=Path, required=True)
    f.add_argument("--method", default="cv_select")
    f.add_argument("--out", type=Path, default=None)
    f.set_defaults(func=cmd_fix)

    e = sub.add_parser("export", parents=[common],
                       help="federated: export sufficient statistics only")
    e.add_argument("--scores", type=Path, required=True)
    e.add_argument("--edges", type=Path, required=True, help="JSON list of source bin edges")
    e.add_argument("--label", required=True)
    e.add_argument("--model-id", required=True)
    e.add_argument("--cohort", default="local")
    e.add_argument("--min-cell", type=int, default=10)
    e.add_argument("--out", type=Path, default=Path("sufficient_statistics.json"))
    e.set_defaults(func=cmd_export)

    args = ap.parse_args(argv)
    for key, value in COMMON_DEFAULTS.items():
        if not hasattr(args, key):
            setattr(args, key, value)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
