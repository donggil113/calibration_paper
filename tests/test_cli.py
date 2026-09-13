"""End-to-end tests for the recalib command line.

The CLI is the surface a deploying site actually touches, so its failures are
the ones least likely to be caught by anyone reading the library code.
"""

import json
import subprocess
import sys

import numpy as np
import pytest


@pytest.fixture(scope="module")
def workspace(tmp_path_factory):
    """A source/target pair on disk, plus source bin edges."""
    import pandas as pd

    from recalib_kit.metrics import bin_edges

    d = tmp_path_factory.mktemp("cli")
    rng = np.random.default_rng(0)

    def sample(n, pi, concept=0.0):
        y = (rng.random(n) < pi).astype(float)
        x = np.where(y == 1, rng.normal(1.5 - concept, 1, n), rng.normal(-1.5, 1, n))
        o = (0.30 / 0.70) * np.exp(3.0 * x)
        return o / (1 + o), y

    s, ys = sample(40000, 0.30)
    t, yt = sample(15000, 0.05)
    pd.DataFrame({"score": s, "label": ys}).to_csv(d / "src.csv", index=False)
    pd.DataFrame({"score": t, "label": yt}).to_csv(d / "tgt.csv", index=False)
    (d / "edges.json").write_text(json.dumps([float(e) for e in bin_edges(s, 15, "equal_mass")]))
    return d


def run(*args, cwd=None):
    return subprocess.run(
        [sys.executable, "-m", "recalib_kit.cli", *args],
        capture_output=True, text=True, cwd=cwd,
    )


@pytest.mark.parametrize("cmd", ["audit", "budget", "fix"])
@pytest.mark.parametrize("flag_first", [True, False])
def test_common_flags_work_in_either_position(workspace, cmd, flag_first):
    """Regression: --json was declared only on the top-level parser, so argparse
    accepted it before the subcommand and rejected it after - the opposite of
    the order everyone types."""
    base = [str(workspace / "tgt.csv"), str(workspace / "src.csv")]
    args = ["--json", cmd] if flag_first else [cmd]
    args += ["--target", base[0], "--source", base[1]]
    if not flag_first:
        args.append("--json")
    r = run(*args)
    assert r.returncode == 0, r.stderr[-800:]
    payload = r.stdout[r.stdout.index("{"):]
    assert isinstance(json.loads(payload), dict)


def test_audit_reports_the_unlabeled_evidence(workspace):
    r = run("audit", "--target", str(workspace / "tgt.csv"),
            "--source", str(workspace / "src.csv"), "--json")
    assert r.returncode == 0, r.stderr[-800:]
    d = json.loads(r.stdout[r.stdout.index("{"):])
    for key in ("label_shift_test_p", "prevalence_target_estimated", "concept_shift_detected"):
        assert key in d, sorted(d)


def test_fix_records_which_candidate_was_selected(workspace):
    r = run("fix", "--target", str(workspace / "tgt.csv"),
            "--source", str(workspace / "src.csv"), "--json")
    assert r.returncode == 0, r.stderr[-800:]
    d = json.loads(r.stdout[r.stdout.index("{"):])
    assert d["method"] == "cv_select"
    assert "selected" in d and "cv_scores" in d
    assert d["ece_after"] <= d["ece_before"] * 1.5


def test_export_refuses_a_payload_below_the_disclosure_threshold(workspace, tmp_path):
    import pandas as pd

    small = tmp_path / "small.csv"
    pd.read_csv(workspace / "tgt.csv").head(30).to_csv(small, index=False)
    r = run("export", "--scores", str(small), "--edges", str(workspace / "edges.json"),
            "--label", "AF", "--model-id", "m", "--out", str(tmp_path / "p.json"))
    assert r.returncode == 2, r.stdout[-500:]
    assert not (tmp_path / "p.json").exists()


def test_unknown_method_names_the_alternatives(workspace):
    r = run("fix", "--target", str(workspace / "tgt.csv"),
            "--source", str(workspace / "src.csv"), "--method", "not_a_method")
    assert r.returncode != 0
    assert "unknown" in (r.stdout + r.stderr).lower()
