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


def test_rejection_is_informative_even_though_acceptance_is_not(source):
    """The asymmetry the paper actually claims.

    A rejection reliably marks a site whose prior estimate is bad.  Acceptance
    does *not* guarantee a good one: the test measures distance from the
    label-shift cone, so concept shift that moves the target histogram along the
    cone is invisible to it - and by Theorem 1 that component is unidentified
    for any unlabeled procedure.  An earlier version of this test asserted clean
    separation in both directions; the transfer matrix refuted it, with one pair
    returning a prevalence estimate of 0.85 against a truth of 0.02 at p=0.23.
    """
    from recalib_kit.decomposition import decompose_unlabeled_budget

    ss, sy = source
    rng = np.random.default_rng(42)
    rejected_errs = []
    for concept in (0.8, 1.2, 1.6):
        xt, yt = sample(40000, 0.05, rng, concept)
        b = decompose_unlabeled_budget(bayes(xt), PI_S, ss, sy)
        if b["concept_shift_detected"]:
            rejected_errs.append(abs(np.log(max(b["pi_t_bbse"], 1e-6) / float(yt.mean()))))
    assert rejected_errs, "the test never fired on a clear concept shift"
    assert max(rejected_errs) > 0.2, "a rejection should mark a materially wrong prior"


def test_plausibility_bound_catches_what_the_test_cannot(source):
    """The blunt guard that covers the structural blind spot."""
    from recalib_kit.decomposition import decompose_unlabeled_budget

    ss, sy = source
    rng = np.random.default_rng(43)
    # A target whose scores sit almost entirely at the top of the range implies
    # an absurd prevalence, which the cone-distance statistic need not reject.
    xt, _ = sample(20000, 0.95, rng)
    b = decompose_unlabeled_budget(bayes(xt), PI_S, ss, sy, max_prior_ratio=3.0)
    assert b["prior_ratio_implausible"] is True
    assert b["bbse_trustworthy"] is False


# ----------------------------------------- unlabeled correction, gated/minimax
def _unlabeled_arms(pi_t, concept, seed, source):
    """ECE of each unlabeled arm on one simulated target site."""
    from recalib_kit.metrics import expected_calibration_error as ece

    ss, sy = source
    rng = np.random.default_rng(seed)
    xt, yt = sample(40000, pi_t, rng, concept)
    s = bayes(xt)
    kw = dict(pi_s=PI_S, source_scores=ss, source_labels=sy, target_unlabeled=s)
    out = {"raw": ece(s, yt)}
    for m in ("prior_correction", "prior_correction_gated", "prior_correction_minimax"):
        out[m] = ece(fit_recalibrator(m, s, None, **kw).transform(s), yt)
    return out


def test_gate_closes_under_concept_shift_and_can_open_under_label_shift():
    """The gate's behaviour, measured with the source operator redrawn.

    Holding one source operator fixed measures a size conditional on that
    operator's error, and in deployment that is the relevant number - which is
    why the gate is *not* the recommended default (see MinimaxPriorCorrection).
    Here we assert the aggregate behaviour the estimator is for: it essentially
    always closes under real concept shift, and opens for a clear majority of
    pure-label-shift sites.
    """
    opens = {0.0: 0, 1.2: 0}
    reps = 12
    for concept in (0.0, 1.2):
        for k in range(reps):
            r = np.random.default_rng(70 + k)
            xs, ys = sample(80000, PI_S, r)        # source redrawn each replicate
            ss_ = bayes(xs)
            xt, _ = sample(30000, 0.05, r, concept)
            g = fit_recalibrator("prior_correction_gated", bayes(xt), None,
                                 pi_s=PI_S, source_scores=ss_, source_labels=ys)
            opens[concept] += bool(g.gate_["applied"])
    assert opens[1.2] == 0, "gate opened under a concept shift it must block"
    assert opens[0.0] >= reps // 2, "gate too rarely opens under pure label shift"


def test_minimax_correction_never_underperforms_shipping_unchanged(source):
    """The property the logit-midpoint version failed.

    That version took the midpoint of the identified set in logit space.  With
    an interval touching zero the midpoint was dragged to a near-zero prior and
    over-corrected, ending worse than leaving the model alone.  Optimising the
    real loss instead of the logit surrogate removes the failure mode.
    """
    for pi_t, concept, seed in [(0.05, 0.0, 60), (0.05, 0.5, 61), (0.05, 1.2, 62),
                                (0.05, 2.0, 63), (0.30, 0.0, 64)]:
        r = _unlabeled_arms(pi_t, concept, seed, source)
        assert r["prior_correction_minimax"] <= r["raw"] + 1e-3, (pi_t, concept, r)


def test_minimax_beats_the_gate_when_concept_shift_is_mild(source):
    r = _unlabeled_arms(0.05, 0.5, 65, source)
    assert r["prior_correction_minimax"] < r["prior_correction_gated"]


def test_identified_set_is_a_point_when_no_correction_is_warranted(source):
    """A site with no shift must be left alone by every unlabeled arm."""
    r = _unlabeled_arms(PI_S, 0.0, 66, source)
    for k in ("prior_correction", "prior_correction_gated", "prior_correction_minimax"):
        assert abs(r[k] - r["raw"]) < 2e-3, (k, r)


def test_dispatcher_wires_unlabeled_estimators_by_capability(source):
    """Regression: dispatch used a hardcoded name list, so every estimator added
    after it silently fell through to the labeled path and raised at first use."""
    from recalib_kit.recalibrate import METHODS

    ss, sy = source
    rng = np.random.default_rng(67)
    xt, yt = sample(20000, 0.05, rng, 0.3)
    s = bayes(xt)
    for name in METHODS:
        obj = fit_recalibrator(name, s[:400], yt[:400], pi_s=PI_S,
                               source_scores=ss, source_labels=sy)
        assert obj.transform(s[400:]).shape == s[400:].shape, name


def test_zero_label_methods_do_not_vary_with_the_label_budget(source):
    """Regression: a zero-label method must be invariant to the calibration set.

    The recalibration sweep passed the small labeled calibration subset to
    ``fit_unlabeled``, so the prior was estimated from a handful of scores
    instead of the site's full (free) score vector.  The result was a figure in
    which prior correction appeared to *improve* as labels were added - the
    exact opposite of the claim a zero-label method exists to support.
    """
    from recalib_kit.metrics import expected_calibration_error as ece

    ss, sy = source
    rng = np.random.default_rng(80)
    xt, yt = sample(20000, 0.05, rng)
    s = bayes(xt)
    vals = []
    for n in (1, 10, 100, 500):
        rec = fit_recalibrator("prior_correction", s[:n], yt[:n], pi_s=PI_S,
                               source_scores=ss, source_labels=sy, target_unlabeled=s)
        vals.append(ece(rec.transform(s), yt))
    assert max(vals) - min(vals) < 1e-9, vals


def test_empirical_n_star_gives_the_free_method_the_whole_site(source):
    """The same invariance, through the n* machinery rather than the sweep."""
    from recalib_kit.sample_size import empirical_n_star

    ss, sy = source
    rng = np.random.default_rng(81)
    xt, yt = sample(20000, 0.05, rng)
    s = bayes(xt)
    res = empirical_n_star(s, yt, eps=0.02, alpha=0.1, method="prior_correction",
                           n_repeats=8, pi_s=PI_S, source_scores=ss, source_labels=sy)
    cov = res.coverage[np.isfinite(res.coverage)]
    assert cov.max() - cov.min() < 1e-9, "coverage moved with a budget the method never spends"


# ------------------------------- measuring the prevalence beats inferring it
def test_prior_correction_with_a_known_prevalence_removes_most_of_the_cliff(source):
    """Theorem 1(c) in its operational form: the label-shift half is real."""
    from recalib_kit.metrics import expected_calibration_error as ece

    rng = np.random.default_rng(90)
    xt, yt = sample(60000, 0.05, rng, concept=0.3)
    s = bayes(xt)
    raw = ece(s, yt)
    fixed = ece(prior_correction(s, PI_S, float(yt.mean())), yt)
    assert fixed < 0.5 * raw


def test_a_measured_prevalence_beats_an_inferred_one_under_concept_shift(source):
    """The finding that motivates PrevalenceCorrection.

    Unlabeled prior estimation fails where concept shift moves the score
    distribution along the label-shift cone.  Measuring the prevalence on a
    small labelled sample sidesteps the identification problem entirely,
    because a proportion needs no model to be identified.
    """
    from recalib_kit.metrics import expected_calibration_error as ece

    ss, sy = source
    rng = np.random.default_rng(91)
    xt, yt = sample(60000, 0.05, rng, concept=1.2)
    s = bayes(xt)
    kw = dict(pi_s=PI_S, source_scores=ss, source_labels=sy, target_unlabeled=s)
    unlabeled = ece(fit_recalibrator("prior_correction", s, None, **kw).transform(s), yt)
    measured = ece(fit_recalibrator("prevalence_correction", s[:200], yt[:200], pi_s=PI_S).transform(s), yt)
    assert measured < unlabeled


def test_prevalence_correction_refuses_a_sample_with_no_positives():
    """A calibration draw with no positive case would collapse every score."""
    from recalib_kit.metrics import expected_calibration_error as ece

    rng = np.random.default_rng(92)
    xt, yt = sample(20000, 0.02, rng)
    s = bayes(xt)
    neg = np.flatnonzero(yt == 0)[:40]
    rec = fit_recalibrator("prevalence_correction", s[neg], yt[neg], pi_s=PI_S)
    assert rec.pi_t is None
    assert ece(rec.transform(s), yt) == pytest.approx(ece(s, yt), abs=1e-12)


# ------------------------------------- recalibrate only where there is a cliff
def _cliff_site(pi, shift, n=30000, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < pi).astype(float)
    x = np.where(y == 1, rng.normal(1.5, 1, n), rng.normal(-1.5, 1, n))
    o = (pi / (1 - pi)) * np.exp(3.0 * x) * shift
    return o / (1 + o), y


def test_recalibration_hurts_a_site_that_has_no_cliff():
    """The result that motivates cross-validated selection.

    Averaged over sites, isotonic regression beat shipping unchanged at every
    budget. Per site and label it was *worse* in a majority of pairs: the
    average was carried by the single worst-calibrated site, while at sites
    already acceptably calibrated the fitted map added more estimation noise
    than it removed bias.
    """
    from recalib_kit.metrics import expected_calibration_error as ece

    s, y = _cliff_site(0.05, 1.0, seed=100)          # already calibrated
    cal, ev = slice(0, 50), slice(50, None)
    raw = ece(s[ev], y[ev])
    iso = ece(fit_recalibrator("isotonic", s[cal], y[cal]).transform(s[ev]), y[ev])
    assert iso > 2 * raw, "expected recalibration to hurt a well-calibrated site"


def test_cv_selection_leaves_a_well_calibrated_site_alone():
    from recalib_kit.metrics import expected_calibration_error as ece

    s, y = _cliff_site(0.05, 1.0, seed=101)
    cal, ev = slice(0, 50), slice(50, None)
    rec = fit_recalibrator("cv_select", s[cal], y[cal])
    assert rec.selected_ == "identity", rec.scores_
    assert ece(rec.transform(s[ev]), y[ev]) == pytest.approx(ece(s[ev], y[ev]), abs=1e-12)


def test_cv_selection_acts_where_there_is_a_real_cliff():
    from recalib_kit.metrics import expected_calibration_error as ece

    s, y = _cliff_site(0.05, 8.0, seed=102)
    cal, ev = slice(0, 100), slice(100, None)
    rec = fit_recalibrator("cv_select", s[cal], y[cal])
    assert rec.selected_ != "identity", rec.scores_
    assert ece(rec.transform(s[ev]), y[ev]) < ece(s[ev], y[ev])


def test_cv_selection_never_badly_underperforms_shipping_unchanged():
    """Across cliff sizes, selection should not be the worst option anywhere."""
    from recalib_kit.metrics import expected_calibration_error as ece

    for k, shift in enumerate((1.0, 1.3, 2.0, 4.0, 8.0)):
        s, y = _cliff_site(0.05, shift, seed=110 + k)
        cal, ev = slice(0, 80), slice(80, None)
        raw = ece(s[ev], y[ev])
        cv = ece(fit_recalibrator("cv_select", s[cal], y[cal]).transform(s[ev]), y[ev])
        assert cv <= raw * 1.5 + 5e-3, (shift, raw, cv)
