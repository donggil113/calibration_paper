#!/usr/bin/env python3
"""Download and preprocess cohorts into model-ready bundles.

    python scripts/build_cohorts.py --plan                      # what would be fetched
    python scripts/build_cohorts.py --cohorts ptbxl georgia     # build the open ones
    python scripts/build_cohorts.py --cohorts mimic --raw /mnt/physionet/mimic-iv-ecg

Open cohorts are fetched directly.  Credentialed ones are never downloaded
anonymously: point ``--raw`` at a copy you obtained with your own PhysioNet
credentials and this builds from it.  Restricted cohorts are built inside their
holding institution and never appear here.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ecgcal.data.download import AccessDenied, describe_access, plan_download  # noqa: E402
from ecgcal.data.preprocess import PreprocessConfig  # noqa: E402
from ecgcal.data.registry import get_cohort, load_cohorts  # noqa: E402

BUILDERS = {
    "ptbxl": ("ecgcal.data.ptbxl", "build", {}),
    "chapman": ("ecgcal.data.physionet2021", "build", {"cohort": "chapman"}),
    "ningbo": ("ecgcal.data.physionet2021", "build", {"cohort": "ningbo"}),
    "cpsc": ("ecgcal.data.physionet2021", "build", {"cohort": "cpsc"}),
    "georgia": ("ecgcal.data.physionet2021", "build", {"cohort": "georgia"}),
    "code15": ("ecgcal.data.code15", "build", {}),
    "mimic": ("ecgcal.data.mimic_ecg", "build", {}),
    "echonext": ("ecgcal.data.echonext", "build", {}),
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cohorts", nargs="*", default=None)
    ap.add_argument("--raw", type=Path, default=Path("data/raw"),
                    help="directory holding the downloaded archives")
    ap.add_argument("--out", type=Path, default=Path("data/bundles"))
    ap.add_argument("--tier", default="core6")
    ap.add_argument("--fs", type=int, default=250)
    ap.add_argument("--limit", type=int, default=None, help="build only the first N records (smoke test)")
    ap.add_argument("--plan", action="store_true", help="report what would be built, then stop")
    args = ap.parse_args(argv)

    keys = args.cohorts or [k for k, c in load_cohorts().items() if c.access == "open"]

    if args.plan:
        plan = plan_download(keys, args.raw)
        print(f"data root: {plan['root']}\n")
        for k, v in plan["cohorts"].items():
            mark = "present" if v["present"] else "MISSING"
            print(f"  {k:10s} {v['access']:13s} {str(v['size_gb']) + ' GB':>9s}  {mark}")
        print(f"\ntotal to fetch: ~{plan['total_gb']:.1f} GB")
        if plan["blocked"]:
            print(f"\nnot downloadable without credentials or an agreement: {', '.join(plan['blocked'])}")
            for k in plan["blocked"]:
                print("\n" + describe_access(k))
        return 0

    cfg = PreprocessConfig(fs_out=args.fs)
    failures = []
    for key in keys:
        c = get_cohort(key)
        if c.is_restricted:
            print(f"\n[{key}] restricted cohort: built inside its holding institution, not here.")
            print(describe_access(key))
            continue
        if key not in BUILDERS:
            print(f"\n[{key}] no builder registered", file=sys.stderr)
            failures.append(key)
            continue

        mod_name, fn_name, kw = BUILDERS[key]
        print(f"\n=== {key} ({c.display}, {c.country}) ===")
        raw = args.raw / key
        if not raw.exists():
            print(f"  raw data not found at {raw}")
            print(describe_access(key))
            failures.append(key)
            continue
        try:
            import importlib

            build = getattr(importlib.import_module(mod_name), fn_name)
            t0 = time.time()
            bundle = build(raw, args.out, tier=args.tier, cfg=cfg, limit=args.limit, **kw)
            print(f"  built {len(bundle)} records in {time.time() - t0:.0f}s")
            s = bundle.summary()
            print(f"  patients: {s['n_patients']}  signal: {s['signal_shape']}")
            for lab, prev in s["prevalence"].items():
                n_obs = s["n_observed"][lab]
                print(f"    {lab:6s} prevalence {prev:.4f}  observed in {n_obs}" if n_obs
                      else f"    {lab:6s} not annotated by this cohort (masked)")
        except AccessDenied as e:
            print(f"  ACCESS DENIED: {e}")
            failures.append(key)
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}")
            failures.append(key)

    if failures:
        print(f"\n{len(failures)} cohort(s) not built: {', '.join(failures)}")
        return 1
    print("\nall requested cohorts built")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
