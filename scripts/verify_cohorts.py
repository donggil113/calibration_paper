#!/usr/bin/env python3
"""Check built bundles against the registry, and fail loudly on a mismatch.

    python scripts/verify_cohorts.py --bundles data/bundles

Several of these datasets have had records withdrawn between versions - PTB-XL
went from 21,837 records and 18,885 patients in v1.0.1 to 21,799 and 18,869 in
v1.0.3 - so a silent count mismatch means a result table is describing a
different cohort than its caption claims.  This script exists so that failure is
loud and happens before the modelling, not after the paper.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ecgcal.data.base import bundle_exists, load_bundle  # noqa: E402
from ecgcal.data.registry import load_cohorts  # noqa: E402

TOLERANCE = 0.01          # fraction by which a count may differ before it is an error


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundles", type=Path, default=Path("data/bundles"))
    ap.add_argument("--cohorts", nargs="*", default=None)
    ap.add_argument("--strict", action="store_true", help="treat a missing bundle as a failure too")
    args = ap.parse_args(argv)

    registry = load_cohorts()
    keys = args.cohorts or list(registry)
    problems, checked = [], 0

    for key in keys:
        c = registry[key]
        if not bundle_exists(args.bundles, key):
            print(f"  {key:10s} not built")
            if args.strict:
                problems.append(f"{key}: no bundle at {args.bundles / key}")
            continue
        try:
            b = load_bundle(args.bundles, key)
        except RuntimeError as e:              # incomplete build
            print(f"  {key:10s} INCOMPLETE: {e}")
            problems.append(f"{key}: {e}")
            continue

        checked += 1
        n = len(b)
        declared = c.n_records
        note = ""
        if declared:
            drift = abs(n - declared) / declared
            if drift > TOLERANCE:
                note = f"  <-- registry says {declared:,} ({drift:.1%} off)"
                problems.append(
                    f"{key}: built {n:,} records but the registry declares {declared:,} "
                    f"for version {c.version}. Either the download is partial or the "
                    f"registry is pinned to a different release."
                )
        n_pat = b.summary()["n_patients"]
        print(f"  {key:10s} {n:>8,} records  {str(n_pat or '-'):>8} patients  "
              f"v{c.version}{note}")

        # every label must be either observed for most records or masked for all
        for lab, n_obs in b.labels.n_observed().items():
            if 0 < n_obs < 0.5 * n:
                problems.append(
                    f"{key}: label {lab} is observed for only {n_obs:,} of {n:,} records. "
                    "Partial observation is not handled by the masking convention: a label "
                    "is either scored by a cohort or it is not."
                )

    print(f"\nchecked {checked} bundle(s); {len(problems)} problem(s)")
    for p in problems:
        print(f"  ! {p}")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
