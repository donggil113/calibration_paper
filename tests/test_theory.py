"""Tests for the three theorems, on constructions with known ground truth.

These are the tests that would catch a wrong claim in the paper, so they are
written against the theorem statements rather than against the implementation.
Each fixture builds a target distribution whose decomposition into label shift
and concept shift is known by construction, which real data can never provide.
"""

import numpy as np
import pytest

from recalib_kit.conformal import coverage_report, hybrid_transfer_conformal, split_conformal, weighted_conformal
from recalib_kit.decomposition import decompose
from recalib_kit.label_shift import (
    bbse, min_gamma, partial_identification, prior_correction,
    source_operator, target_histogram,
)
# aliased: pytest would otherwise collect the imported estimator as a test case
from recalib_kit.label_shift import test_label_shift_sufficiency as gof_test
from recalib_kit.metrics import bin_edges, expected_calibration_error
from recalib_kit.recalibrate import fit_recalibrator
from recalib_kit.sample_size import n_star_lower_bound, n_star_upper_plugin, residual_shift_gamma

PI_S = 0.30
BETA = 3.0


def sample(n, pi, rng, concept=0.0):
    """Latent-feature model: `concept` tilts the target's class-conditional law."""
    y = (rng.random(n) < pi).astype(float)
    x = np.where(y == 1, rng.normal(1.5 - concept, 1, n), rng.normal(-1.5, 1, n))
    return x, y


def bayes(x, pi=PI_S):
    """Bayes-optimal source scorer, hence exactly calibrated on the source."""
    odds = (pi / (1 - pi)) * np.exp(BETA * x)
    return odds / (1 + odds)


@pytest.fixture(scope="module")
def source():
    rng = np.random.default_rng(0)
    xs, ys = sample(300000, PI_S, rng)
    return bayes(xs), ys


# ---------------------------------------------------------------- Theorem 1
def test_source_model_starts_calibrated(source):
    """The decomposition's premise: no miscalibration before transfer."""
    s, y = source
    assert expected_calibration_error(s, y) < 0.005


def test_decomposition_is_an_exact_identity():
    rng = np.random.default_rng(1)
    for pi_t, c in [(0.05, 0.0), (0.30, 0.8), (0.05, 0.8), (0.30, 0.0), (0.60, 0.3)]:
        xt, yt = sample(60000, pi_t, rng, c)
        d = decompose(bayes(xt), yt, pi_s=PI_S, pi_t=pi_t)
        assert abs(d.residual) < 1e-12, f"identity broken at pi_t={pi_t}, concept={c}"
        assert d.total == pytest.approx(d.d_label + d.d_concept + d.interaction, abs=1e-12)


def test_pure_label_shift_is_entirely_free():
    rng = np.random.default_rng(2)
    xt, yt = sample(150000, 0.05, rng, concept=0.0)
    d = decompose(bayes(xt), yt, pi_s=PI_S, pi_t=0.05)
    assert d.free_fraction > 0.97
    assert abs(d.d_concept) < 0.02 * d.total
    assert d.realized_free_gain > 0.9              # and the free fix actually works


def test_pure_concept_shift_is_entirely_paid():
    rng = np.random.default_rng(3)
    xt, yt = sample(150000, PI_S, rng, concept=0.8)
    d = decompose(bayes(xt), yt, pi_s=PI_S, pi_t=PI_S)
    assert d.d_label == pytest.approx(0.0, abs=1e-12)   # T is the identity when priors match
    assert d.d_concept == pytest.approx(d.total, rel=1e-6)
    assert abs(d.realized_free_gain) < 0.01            # nothing is free here


def test_prior_correction_is_the_exact_fix_under_label_shift():
    rng = np.random.default_rng(4)
    xt, yt = sample(150000, 0.05, rng, concept=0.0)
    s = bayes(xt)
    before = expected_calibration_error(s, yt)
    after = expected_calibration_error(prior_correction(s, PI_S, 0.05), yt)
    assert before > 0.02 and after < 0.01 * before + 0.005


# ------------------------------------------------- Theorem 1: identification
def test_bbse_is_unbiased_under_label_shift(source):
    ss, sy = source
    edges = bin_edges(ss, 20, "equal_mass")
    A, pi_s_vec = source_operator(ss, sy, edges)
    errs = []
    for seed in range(12):
        rng = np.random.default_rng(100 + seed)
        xt, _ = sample(80000, 0.05, rng)
        errs.append(bbse(A, pi_s_vec, target_histogram(bayes(xt), edges)).prevalence - 0.05)
    assert abs(np.mean(errs)) < 0.004


def test_gof_test_has_correct_unconditional_size():
    """Size must be measured with the *source* sample redrawn each replicate.

    Holding one source operator fixed measures a size conditional on that
    operator's own error, which legitimately departs from nominal - an earlier
    version of this test did exactly that and read 0.10 against a nominal 0.05.
    """
    rejects = []
    for k in range(60):
        r = np.random.default_rng(20_000 + k)
        xs, ys = sample(60000, PI_S, r)
        ss_ = bayes(xs)
        e = bin_edges(ss_, 20, "equal_mass")
        A, pv = source_operator(ss_, ys, e)
        xt, _ = sample(40000, 0.05, r)                      # H0: pure label shift
        rejects.append(gof_test(A, pv, target_histogram(bayes(xt), e), 40000, 60000,
                                n_mc=200, seed=k)["reject_label_shift"])
    assert np.mean(rejects) < 0.15                          # 60 replicates: binomial slack


def test_gof_test_has_power_against_concept_shift(source):
    ss, sy = source
    edges = bin_edges(ss, 20, "equal_mass")
    A, pv = source_operator(ss, sy, edges)
    rng = np.random.default_rng(5)
    for concept in (0.3, 1.0):
        xt, _ = sample(80000, 0.05, rng, concept=concept)
        alt = gof_test(A, pv, target_histogram(bayes(xt), edges), 80000, len(ss), n_mc=200)
        assert alt["reject_label_shift"], f"missed concept shift at c={concept}"


def test_monte_carlo_null_is_less_anticonservative_than_chi2():
    """The asymptotic reference over-rejects; the parametric bootstrap does not."""
    mc, chi2 = [], []
    for k in range(60):
        r = np.random.default_rng(30_000 + k)
        xs, ys = sample(40000, PI_S, r)
        ss_ = bayes(xs)
        e = bin_edges(ss_, 20, "equal_mass")
        A, pv = source_operator(ss_, ys, e)
        xt, _ = sample(40000, 0.05, r)
        res = gof_test(A, pv, target_histogram(bayes(xt), e), 40000, 40000, n_mc=200, seed=k)
        mc.append(res["p_value"] < 0.05)
        chi2.append(res["p_value_chi2"] < 0.05)
    assert np.mean(mc) <= np.mean(chi2) + 1e-9


def test_partial_identification_is_sharp_and_honest(source):
    """The identified set covers the truth exactly when the assumed budget is
    at least the true shift, and is empty below the data's own floor."""
    ss, sy = source
    edges = bin_edges(ss, 20, "equal_mass")
    A, pv = source_operator(ss, sy, edges)
    rng = np.random.default_rng(6)
    pi_t = 0.05
    xt, _ = sample(120000, pi_t, rng, concept=2.0)
    q_t = target_histogram(bayes(xt), edges)

    w_true = np.array([(1 - pi_t) / (1 - PI_S), pi_t / PI_S])
    eta_true = float(np.abs(q_t - A @ w_true).sum())
    g_min = min_gamma(A, pv, q_t)

    assert g_min <= eta_true + 1e-9                       # a lower bound, never an estimate
    assert not partial_identification(A, pv, q_t, gamma=g_min * 0.5)["feasible"]
    generous = partial_identification(A, pv, q_t, gamma=eta_true * 1.2)
    assert generous["lo"] - 1e-9 <= pi_t <= generous["hi"] + 1e-9
    # and the interval widens monotonically with the assumed budget
    widths = [partial_identification(A, pv, q_t, gamma=g)["width"]
              for g in np.linspace(g_min, g_min + 0.15, 6)]
    assert all(b >= a - 1e-9 for a, b in zip(widths, widths[1:]))


# ---------------------------------------------------------------- Theorem 2
def test_lower_bound_is_a_step_at_the_budget_and_scales_as_eps_squared():
    assert n_star_lower_bound(0.02, 0.25, gamma=0.01)["n_star"] == 0      # no hard instance fits
    assert n_star_lower_bound(0.02, 0.25, gamma=0.05)["n_star"] > 0
    n1 = n_star_lower_bound(0.02, 0.25, gamma=0.2)["n_star"]
    n2 = n_star_lower_bound(0.01, 0.25, gamma=0.2)["n_star"]
    assert n2 / n1 == pytest.approx(4.0, rel=0.05)


def test_upper_bound_is_monotone_in_shift_and_scales_as_eps_squared():
    rng = np.random.default_rng(7)
    xt, _ = sample(40000, 0.05, rng, 0.4)
    s = bayes(xt)
    ns = [n_star_upper_plugin(0.02, 0.1, s, gamma=g)["n_star"] for g in (0.0, 0.004, 0.008, 0.012)]
    assert all(b >= a for a, b in zip(ns, ns[1:]))
    n_coarse = n_star_upper_plugin(0.04, 0.1, s)["n_star"]
    n_fine = n_star_upper_plugin(0.02, 0.1, s)["n_star"]
    assert n_fine / n_coarse == pytest.approx(4.0, rel=0.1)
    assert not n_star_upper_plugin(0.02, 0.1, s, gamma=0.03)["feasible"]


def test_upper_bound_respects_its_cauchy_schwarz_corollary():
    rng = np.random.default_rng(8)
    xt, _ = sample(40000, 0.05, rng, 0.4)
    r = n_star_upper_plugin(0.02, 0.1, bayes(xt))
    assert r["n_star"] <= r["n_star_cs_bound"]


def test_hybrid_beats_temperature_scaling_in_the_few_shot_regime(source):
    """The estimator claim: warm-starting at the free correction means the
    labels are spent on concept shift, not on undoing a known prior shift."""
    ss, sy = source
    rng = np.random.default_rng(9)
    xt, yt = sample(60000, 0.05, rng, concept=0.5)
    s = bayes(xt)
    cal, ev = slice(0, 300), slice(300, None)
    out = {}
    for m in ("identity", "temperature", "hybrid", "prior_correction"):
        rec = fit_recalibrator(m, s[cal], yt[cal], pi_s=PI_S, source_scores=ss, source_labels=sy)
        out[m] = expected_calibration_error(rec.transform(s[ev]), yt[ev])
    assert out["hybrid"] < out["temperature"]
    assert out["hybrid"] < out["identity"]
    assert out["prior_correction"] < out["identity"]      # free correction alone already helps


def test_discrimination_survives_what_calibration_does_not(source):
    """The paper's central dissociation, stated as a test."""
    from recalib_kit.metrics import auroc

    ss, sy = source
    rng = np.random.default_rng(10)
    xt, yt = sample(80000, 0.05, rng, concept=0.0)
    s = bayes(xt)
    assert auroc(s, yt) > 0.9
    assert expected_calibration_error(s, yt) > 10 * expected_calibration_error(ss, sy)


# ---------------------------------------------------------------- Theorem 3
def _coverage_over_draws(make, y_of, n_draws=120, seed=0):
    rng = np.random.default_rng(seed)
    return float(np.mean([coverage_report(*make(rng)) ["coverage"] for _ in range(n_draws)]))


def test_conformal_validity_and_the_concept_shift_penalty(source):
    ss, sy = source
    rng = np.random.default_rng(11)
    alpha = 0.1

    for concept, expect_source_valid in [(0.0, True), (0.6, False)]:
        xt, yt = sample(60000, 0.05, rng, concept)
        s = bayes(xt)
        covs = {"split": [], "weighted": [], "hybrid": []}
        for _ in range(80):
            perm = rng.permutation(len(s))
            cal, ev = perm[:60], perm[60:20060]
            covs["split"].append(coverage_report(split_conformal(s[cal], yt[cal], s[ev], alpha), yt[ev])["coverage"])
            covs["weighted"].append(coverage_report(weighted_conformal(ss, sy, s[ev], PI_S, 0.05, alpha), yt[ev])["coverage"])
            covs["hybrid"].append(coverage_report(
                hybrid_transfer_conformal(s[cal], yt[cal], ss, sy, s[ev], PI_S, 0.05, alpha=alpha, gamma=0.05),
                yt[ev])["coverage"])
        m = {k: float(np.mean(v)) for k, v in covs.items()}
        # target-only split conformal is valid regardless of shift
        assert m["split"] >= 1 - alpha - 0.01
        # unlabeled reweighted-source conformal is valid only under pure label shift
        assert (m["weighted"] >= 1 - alpha - 0.005) == expect_source_valid
        # inflating the level restores the guarantee in both regimes
        assert m["hybrid"] >= 1 - alpha


def test_unlabeled_conformal_is_far_more_stable_under_label_shift(source):
    """Its practical advantage: the same guarantee with far less draw-to-draw
    variance, because it leans on a large source sample."""
    ss, sy = source
    rng = np.random.default_rng(12)
    xt, yt = sample(60000, 0.05, rng, 0.0)
    s = bayes(xt)
    sp, wt = [], []
    for _ in range(80):
        perm = rng.permutation(len(s))
        cal, ev = perm[:60], perm[60:20060]
        sp.append(coverage_report(split_conformal(s[cal], yt[cal], s[ev], 0.1), yt[ev])["coverage"])
        wt.append(coverage_report(weighted_conformal(ss, sy, s[ev], PI_S, 0.05, 0.1), yt[ev])["coverage"])
    assert np.std(wt) < 0.5 * np.std(sp)


# ------------------------------------------------- regressions on real defects
def test_realized_gain_is_undefined_when_already_calibrated():
    """Regression: the gain is a ratio with the baseline in the denominator.

    A site moving from ECE 0.0001 to 0.0006 - both zero for any practical
    purpose - reported a "gain" of -5, and averaging those across labels
    manufactured an apparent catastrophic failure of the prior correction at a
    site where it had done nothing at all.
    """
    rng = np.random.default_rng(40)
    p = rng.beta(1, 20, 40000)
    y = (rng.random(40000) < p).astype(float)
    pi = float(y.mean())
    d = decompose(p, y, pi_s=pi, pi_t=pi)
    assert d.ece_before < d.GAIN_FLOOR
    assert np.isnan(d.realized_free_gain)


def test_pooled_ratio_is_not_dominated_by_a_near_zero_denominator():
    """Regression: collapse_map pools ratios instead of averaging them."""
    import pandas as pd
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
    from analysis import collapse_map

    transfer = pd.DataFrame([
        # source: one label is essentially perfectly calibrated
        dict(site="src", country="US", is_source=True, label="A", ece=0.00010,
             auroc=0.90, prevalence_ratio=1.0),
        dict(site="src", country="US", is_source=True, label="B", ece=0.05000,
             auroc=0.90, prevalence_ratio=1.0),
        # target: both labels barely move in absolute terms
        dict(site="tgt", country="DE", is_source=False, label="A", ece=0.00060,
             auroc=0.89, prevalence_ratio=1.0),
        dict(site="tgt", country="DE", is_source=False, label="B", ece=0.05500,
             auroc=0.89, prevalence_ratio=1.0),
    ])
    cm = collapse_map(transfer, pd.DataFrame())
    pooled = float(cm.ece_ratio.iloc[0])
    mean_of_ratios = np.mean([0.00060 / 0.00010, 0.05500 / 0.05000])
    assert pooled == pytest.approx((0.0006 + 0.055) / (0.0001 + 0.05), rel=1e-9)
    assert pooled < 1.2 < mean_of_ratios          # the averaged version reads 3.5x


def test_degenerate_prior_estimates_are_flagged():
    """A prior pinned at the simplex boundary is a failed solve, not a number."""
    from recalib_kit.decomposition import decompose_unlabeled_budget

    rng = np.random.default_rng(41)
    xs, ys = sample(60000, PI_S, rng)
    ss = bayes(xs)
    # a target whose scores sit far from anything the source operator can span
    budget = decompose_unlabeled_budget(np.full(5000, 1e-6), PI_S, ss, ys)
    assert budget["bbse_degenerate"] is True
    assert budget["bbse_trustworthy"] is False


def test_gof_test_separates_trustworthy_from_untrustworthy_priors():
    """The deployable claim: an unlabeled test tells a site when to believe BBSE."""
    from recalib_kit.decomposition import decompose_unlabeled_budget

    rng = np.random.default_rng(42)
    xs, ys = sample(200000, PI_S, rng)
    ss = bayes(xs)
    errs = {True: [], False: []}
    for concept in (0.0, 0.0, 0.6, 1.0):
        xt, yt = sample(40000, 0.05, rng, concept)
        b = decompose_unlabeled_budget(bayes(xt), PI_S, ss, ys)
        rel = abs(b["pi_t_bbse"] - float(yt.mean())) / float(yt.mean())
        errs[bool(b["bbse_trustworthy"])].append(rel)
    assert errs[True], "no site was judged trustworthy; the test is too aggressive"
    assert max(errs[True]) < min(errs[False]) if errs[False] else True
