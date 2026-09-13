"""Metric correctness, checked against closed forms and against sklearn."""

import numpy as np
import pytest

from recalib_kit.metrics import (
    auprc, auroc, bin_edges, binned_stats, bootstrap_ci, brier_decomposition,
    calibration_report, expected_calibration_error, maximum_calibration_error,
    reliability_curve, squared_calibration_error,
)


@pytest.fixture
def calibrated():
    rng = np.random.default_rng(0)
    p = rng.beta(1.2, 6, 40000)
    return p, (rng.random(40000) < p).astype(float)


def test_auroc_matches_sklearn():
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(1)
    for transform in (lambda s: s, lambda s: np.clip(s * 3, 0, 1), lambda s: np.round(s, 1)):
        p = rng.beta(1.2, 6, 5000)
        y = (rng.random(5000) < p).astype(float)
        s = transform(p)
        assert auroc(s, y) == pytest.approx(roc_auc_score(y, s), abs=1e-9)


def test_auprc_matches_sklearn_with_ties():
    from sklearn.metrics import average_precision_score

    rng = np.random.default_rng(2)
    p = rng.beta(1.2, 6, 5000)
    y = (rng.random(5000) < p).astype(float)
    s = np.round(p, 1)                      # heavy ties
    assert auprc(s, y) == pytest.approx(average_precision_score(y, s), abs=1e-9)


def test_debiasing_reduces_bias_on_calibrated_scores(calibrated):
    p, y = calibrated
    plugin = squared_calibration_error(p, y, debiased=False)
    deb = squared_calibration_error(p, y, debiased=True)
    assert deb < plugin                      # the correction removes upward bias
    assert abs(deb) < 1e-3                   # and lands near the true value of zero


def test_debiased_estimator_may_go_negative():
    """Not clipped at zero: clipping reintroduces bias exactly where we claim
    a model is well calibrated."""
    rng = np.random.default_rng(3)
    worst = 0.0
    for s in range(40):
        r = np.random.default_rng(s)
        p = r.beta(2, 8, 800)
        y = (r.random(800) < p).astype(float)
        worst = min(worst, squared_calibration_error(p, y, n_bins=20))
    assert worst < 0


def test_miscalibration_is_detected_while_auroc_is_unchanged(calibrated):
    p, y = calibrated
    bad = np.clip(p * 3, 0, 1)
    assert expected_calibration_error(bad, y) > 10 * expected_calibration_error(p, y)
    assert auroc(bad, y) == pytest.approx(auroc(p, y), abs=0.01)


def test_brier_decomposition_residual_is_within_bin_variance(calibrated):
    p, y = calibrated
    coarse = brier_decomposition(p, y, n_bins=5)["residual"]
    fine = brier_decomposition(p, y, n_bins=50)["residual"]
    assert abs(fine) < abs(coarse)           # residual is within-bin score variance


def test_equal_mass_bins_are_balanced():
    rng = np.random.default_rng(4)
    s = rng.beta(1, 30, 10000)               # heavily skewed toward zero
    st = binned_stats(s, np.zeros_like(s), n_bins=10, scheme="equal_mass")
    occupancy = st.count[st.count > 0]
    assert occupancy.max() / occupancy.min() < 3
    stw = binned_stats(s, np.zeros_like(s), n_bins=10, scheme="equal_width")
    assert stw.n_nonempty < st.n_nonempty    # equal width wastes bins on skewed scores


def test_bootstrap_ci_brackets_point_estimate(calibrated):
    p, y = calibrated
    pt, lo, hi = bootstrap_ci(expected_calibration_error, p[:4000], y[:4000], n_boot=120, seed=0)
    assert lo <= pt <= hi


def test_stratified_bootstrap_keeps_positives():
    rng = np.random.default_rng(5)
    y = np.zeros(2000)
    y[:12] = 1                                # 12 positives only
    p = rng.random(2000)
    _, lo, hi = bootstrap_ci(auroc, p, y, n_boot=200, seed=0, stratify=y)
    assert np.isfinite(lo) and np.isfinite(hi)


def test_reliability_curve_intervals_are_valid(calibrated):
    p, y = calibrated
    rc = reliability_curve(p, y, n_bins=10)
    assert np.all(rc["ci_low"] <= rc["mean_label"] + 1e-12)
    assert np.all(rc["mean_label"] <= rc["ci_high"] + 1e-12)


def test_empty_and_degenerate_inputs_do_not_crash():
    assert np.isnan(auroc(np.array([0.5]), np.array([1.0])))
    assert np.isnan(auprc(np.array([0.5, 0.2]), np.array([0.0, 0.0])))
    assert np.isnan(maximum_calibration_error(np.array([0.5]), np.array([1.0]), min_count=10))
    r = calibration_report(np.array([]), np.array([]))
    assert r.n == 0


def test_mismatched_shapes_raise():
    with pytest.raises(ValueError):
        binned_stats(np.zeros(10), np.zeros(9))
