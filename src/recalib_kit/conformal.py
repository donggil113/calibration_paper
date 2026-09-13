r"""Theorem 3: conformal prediction sets that stay valid under concept shift.

Recalibrating a probability is one guarantee; committing to a *decision* is
another.  A deploying hospital wants a set-valued output - "this ECG is
confidently negative", "this one is confidently positive", "this one needs a
human" - with a coverage guarantee it can put in a protocol.  Split conformal
gives exactly that, but its guarantee rests on exchangeability between the
calibration data and the test data, and cross-national transfer breaks it.

Three regimes, in order of what they assume:

* :func:`split_conformal` on **target** data: finite-sample valid with no
  distributional assumption at all, but it needs :math:`n \ge 1/\alpha - 1`
  target labels merely to produce a non-trivial set, and its width is noisy at
  the sample sizes a new site can afford.
* :func:`weighted_conformal` on **reweighted source** data: needs no target
  labels, exactly valid under pure label shift, and off by the concept-shift
  budget :math:`\gamma` otherwise.
* :func:`hybrid_transfer_conformal`: the estimator of Theorem 3.  Mixes the two
  and *pays for the bias it incurs* by inflating the nominal level, so the
  target coverage guarantee is restored:

  .. math::
      \text{run at } \alpha' = \alpha - (1-\lambda)\,\gamma
      \;\Longrightarrow\;
      \Pr_t\big[Y \in C(X)\big] \ge 1-\alpha ,

  whenever :math:`\alpha' > 0`.  Setting :math:`\lambda = 1` recovers
  target-only conformal and its exact guarantee; :math:`\lambda = 0` recovers
  the label-shift-corrected source version.  Intermediate :math:`\lambda`
  buys width from the large source sample at a coverage price that is known in
  advance rather than discovered in deployment.

:math:`\gamma` is an assumption, not an estimate - the same honesty that
governs the partial-identification machinery applies here.
:func:`gamma_sensitivity` sweeps it, and that sweep is what belongs in a
deployment dossier.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "nonconformity_binary",
    "conformal_quantile",
    "ConformalSets",
    "split_conformal",
    "weighted_conformal",
    "hybrid_transfer_conformal",
    "mondrian_conformal",
    "coverage_report",
    "gamma_sensitivity",
    "min_n_for_level",
]


def min_n_for_level(alpha: float) -> int:
    r"""Smallest calibration set giving a non-trivial split-conformal set.

    The conformal quantile is the :math:`\lceil (n+1)(1-\alpha)\rceil`-th order
    statistic, which exceeds :math:`n` - and so degenerates to "predict
    everything" - unless :math:`n \ge 1/\alpha - 1`.  At the 90% level that is
    9 labeled ECGs; at 99% it is 99.  Worth stating plainly because it is a
    hard floor no method can argue its way past.
    """
    return int(np.ceil(1.0 / alpha - 1.0))


def nonconformity_binary(scores: np.ndarray, labels: np.ndarray) -> np.ndarray:
    r"""Score :math:`s(x,y) = 1 - \hat p(y\mid x)`: small when the model agreed.

    Written as a branch rather than as ``1 - (1 - p)`` for the negative class.
    Those are equal in exact arithmetic and catastrophically different in
    floating point: for :math:`p` below the double-precision resolution of 1,
    :math:`1-p` rounds to exactly 1 and the round trip returns exactly 0 instead
    of :math:`p`.

    That is not a rounding nicety.  A model that separates a label cleanly
    produces exactly such scores, the whole calibration sample collapses to
    nonconformity 0, the conformal quantile becomes 0, and every prediction set
    computed against it is empty - while the set-construction code, which uses
    :math:`p` directly, disagrees with the calibration code about what the same
    quantity is.  Measured coverage fell to 0.02 on labels where the quantile
    itself still satisfied its guarantee at 0.96.
    """
    p = np.clip(np.asarray(scores, float).ravel(), 0.0, 1.0)
    y = np.asarray(labels, float).ravel()
    return np.where(y == 1, 1.0 - p, p)


def conformal_quantile(
    scores: np.ndarray, alpha: float, weights: np.ndarray | None = None
) -> float:
    r"""The :math:`(1-\alpha)` conformal quantile with the finite-sample correction.

    Unweighted, this is the :math:`\lceil (n+1)(1-\alpha)\rceil / n` empirical
    quantile - the ``+1`` is what makes the guarantee exact rather than
    asymptotic, and dropping it (as a plain ``np.quantile`` would) silently
    undercovers at small :math:`n`, which is the whole regime of interest here.
    Weighted, it is the weighted quantile with an extra atom of mass at
    :math:`+\infty` for the test point, following Tibshirani et al. (2019).
    """
    s = np.asarray(scores, float).ravel()
    n = s.size
    if n == 0:
        return float("inf")
    if weights is None:
        k = int(np.ceil((n + 1) * (1 - alpha)))
        if k > n:
            return float("inf")                     # too few points: abstain everywhere
        return float(np.sort(s)[k - 1])
    w = np.asarray(weights, float).ravel()
    if w.shape != s.shape:
        raise ValueError("weights must match scores")
    w = np.maximum(w, 0.0)
    total = w.sum()
    if total <= 0:
        return float("inf")
    order = np.argsort(s)
    s_sorted, w_sorted = s[order], w[order]
    # the test point contributes an atom at +inf with the mean weight
    w_inf = total / n
    cum = np.cumsum(w_sorted) / (total + w_inf)
    idx = np.searchsorted(cum, 1 - alpha, side="left")
    if idx >= n:
        return float("inf")
    return float(s_sorted[idx])


@dataclass
class ConformalSets:
    """Prediction sets for a binary label, as two boolean membership columns."""

    include_0: np.ndarray
    include_1: np.ndarray
    q_hat: float
    alpha_nominal: float
    alpha_run: float
    method: str
    n_cal: int

    @property
    def size(self) -> np.ndarray:
        """Number of labels in each set: 0 (empty), 1 (decisive) or 2 (abstain)."""
        return self.include_0.astype(int) + self.include_1.astype(int)

    @property
    def abstain_rate(self) -> float:
        return float(np.mean(self.size == 2))

    @property
    def empty_rate(self) -> float:
        return float(np.mean(self.size == 0))

    @property
    def decisive_rate(self) -> float:
        return float(np.mean(self.size == 1))

    @property
    def review_rate(self) -> float:
        """Share of cases a protocol would route to a human.

        An empty set is as much a referral as a two-label set: it says no label
        is plausible under the calibration distribution, which in a clinical
        workflow is a reason to look, not a reason to act.  Reporting only the
        abstention rate understates the human workload.
        """
        return float(np.mean(self.size != 1))

    def covers(self, labels: np.ndarray) -> np.ndarray:
        y = np.asarray(labels, float).ravel()
        return np.where(y == 1, self.include_1, self.include_0)


def _sets_from_q(test_scores: np.ndarray, q: float, **meta) -> ConformalSets:
    # s(x, 0) = 1 - phat(0|x) = p   and   s(x, 1) = 1 - phat(1|x) = 1 - p
    p = np.clip(np.asarray(test_scores, float).ravel(), 0.0, 1.0)
    return ConformalSets(include_0=(p <= q), include_1=((1.0 - p) <= q), **meta)


def split_conformal(
    cal_scores: np.ndarray,
    cal_labels: np.ndarray,
    test_scores: np.ndarray,
    alpha: float = 0.1,
) -> ConformalSets:
    """Exchangeable split conformal on target calibration data.

    The gold standard for validity - and the reason the paper reports it as the
    reference arm - but it converts the entire label budget into one quantile,
    so at :math:`n` of a few dozen its width is dominated by sampling noise.
    """
    s = nonconformity_binary(cal_scores, cal_labels)
    q = conformal_quantile(s, alpha)
    return _sets_from_q(
        test_scores, q, q_hat=q, alpha_nominal=alpha, alpha_run=alpha,
        method="split", n_cal=int(np.size(s)),
    )


def weighted_conformal(
    cal_scores: np.ndarray,
    cal_labels: np.ndarray,
    test_scores: np.ndarray,
    pi_s: float,
    pi_t: float,
    alpha: float = 0.1,
) -> ConformalSets:
    r"""Label-shift-weighted conformal using **source** calibration data.

    Each source calibration point is weighted by :math:`\pi_t(y_i)/\pi_s(y_i)`,
    which makes the reweighted source calibration sample exchangeable with the
    target one under pure label shift - and therefore needs no target labels at
    all.  Under concept shift its coverage is off by at most the budget
    :math:`\gamma`; :func:`hybrid_transfer_conformal` is the version that pays
    that debt explicitly.
    """
    y = np.asarray(cal_labels, float).ravel()
    s = nonconformity_binary(cal_scores, y)
    w = np.where(y == 1, pi_t / max(pi_s, 1e-12), (1 - pi_t) / max(1 - pi_s, 1e-12))
    q = conformal_quantile(s, alpha, weights=w)
    return _sets_from_q(
        test_scores, q, q_hat=q, alpha_nominal=alpha, alpha_run=alpha,
        method="weighted_label_shift", n_cal=int(s.size),
    )


def hybrid_transfer_conformal(
    target_cal_scores: np.ndarray,
    target_cal_labels: np.ndarray,
    source_cal_scores: np.ndarray,
    source_cal_labels: np.ndarray,
    test_scores: np.ndarray,
    pi_s: float,
    pi_t: float,
    alpha: float = 0.1,
    gamma: float = 0.0,
    lam: float | None = None,
) -> ConformalSets:
    r"""**Theorem 3.** Mix target and reweighted-source calibration scores, and
    pay the concept-shift bias by inflating the level.

    Total mass :math:`\lambda` is placed on the :math:`n` target calibration
    points and :math:`1-\lambda` on the reweighted source points.  The
    reweighted source half is exchangeable with the target only up to the
    concept-shift budget, contributing at most :math:`(1-\lambda)\gamma` of
    coverage loss, so the procedure is run at

    .. math:: \alpha' = \alpha - (1-\lambda)\gamma

    and the target coverage guarantee :math:`1-\alpha` is restored.  The
    trade-off is explicit: a larger :math:`\lambda` costs width (it leans on the
    small target sample) and buys back level.

    ``lam=None`` selects :math:`\lambda` by :func:`_default_lambda`, which
    spends only as much of the target sample as the budget requires.  If
    :math:`\alpha' \le 0` the requested level is unreachable at that
    :math:`\gamma` and the method falls back to target-only conformal, which is
    always valid - never to a silently invalid set.
    """
    ty = np.asarray(target_cal_labels, float).ravel()
    ts = nonconformity_binary(target_cal_scores, ty)
    sy = np.asarray(source_cal_labels, float).ravel()
    ss = nonconformity_binary(source_cal_scores, sy)
    n_t, n_s = ts.size, ss.size

    if lam is None:
        lam = _default_lambda(n_t, alpha, gamma)
    lam = float(np.clip(lam, 0.0, 1.0))
    alpha_run = alpha - (1.0 - lam) * gamma

    if alpha_run <= 0 or n_t == 0 and lam > 0:
        if n_t >= min_n_for_level(alpha):
            out = split_conformal(target_cal_scores, ty, test_scores, alpha)
            out.method = "hybrid->split_fallback"
            out.alpha_nominal = alpha
            return out
        lam, alpha_run = 0.0, max(alpha - gamma, 1e-6)

    sw = np.where(sy == 1, pi_t / max(pi_s, 1e-12), (1 - pi_t) / max(1 - pi_s, 1e-12))
    sw = sw / sw.sum() * (1.0 - lam) if sw.sum() > 0 else sw
    tw = np.full(n_t, lam / n_t) if n_t > 0 else np.zeros(0)

    s_all = np.concatenate([ts, ss])
    w_all = np.concatenate([tw, sw])
    q = conformal_quantile(s_all, alpha_run, weights=w_all)
    return _sets_from_q(
        test_scores, q, q_hat=q, alpha_nominal=alpha, alpha_run=alpha_run,
        method="hybrid_transfer", n_cal=n_t,
    )


def _default_lambda(n_t: int, alpha: float, gamma: float) -> float:
    r"""Spend the smallest target mass that keeps :math:`\alpha' > 0`.

    With :math:`\gamma \le \alpha/2` the source half is cheap and a small
    :math:`\lambda` suffices; as :math:`\gamma \to \alpha` the procedure is
    forced onto the target sample.  Guarded below by what the target sample can
    actually support, since a :math:`\lambda` larger than the target data can
    carry buys nothing but noise.
    """
    if gamma <= 0:
        return 0.0
    if n_t < min_n_for_level(alpha):
        return 0.0
    lam_needed = max(0.0, 1.0 - alpha / (2.0 * gamma)) if gamma > 0 else 0.0
    return float(np.clip(lam_needed, 0.0, 1.0))


def mondrian_conformal(
    cal_scores: np.ndarray,
    cal_labels: np.ndarray,
    cal_groups: np.ndarray,
    test_scores: np.ndarray,
    test_groups: np.ndarray,
    alpha: float = 0.1,
    min_group: int | None = None,
) -> ConformalSets:
    """Group-conditional (Mondrian) conformal: a separate quantile per stratum.

    Marginal coverage can hide a subgroup the model fails on entirely - a real
    risk here, since the sex and age composition of a screening cohort differs
    sharply from that of an emergency department.  Groups too small to support
    their own quantile fall back to the pooled one, and that fallback is
    recorded rather than hidden, because a group silently pooled is a group
    without a guarantee.
    """
    g_cal = np.asarray(cal_groups).ravel()
    g_test = np.asarray(test_groups).ravel()
    s = nonconformity_binary(cal_scores, cal_labels)
    p = np.clip(np.asarray(test_scores, float).ravel(), 0.0, 1.0)
    floor = min_group if min_group is not None else min_n_for_level(alpha)

    q_pooled = conformal_quantile(s, alpha)
    q_test = np.full(p.shape, q_pooled, dtype=float)
    pooled_groups = []
    for g in np.unique(g_test):
        sel = g_cal == g
        if sel.sum() >= floor:
            q_test[g_test == g] = conformal_quantile(s[sel], alpha)
        else:
            pooled_groups.append(g)

    out = ConformalSets(
        include_0=(p <= q_test), include_1=((1.0 - p) <= q_test),
        q_hat=float(np.median(q_test)), alpha_nominal=alpha, alpha_run=alpha,
        method="mondrian", n_cal=int(s.size),
    )
    out.pooled_groups = pooled_groups          # type: ignore[attr-defined]
    return out


def coverage_report(sets: ConformalSets, labels: np.ndarray) -> dict:
    """Empirical coverage and set-size profile."""
    cov = sets.covers(labels)
    y = np.asarray(labels, float).ravel()
    out = {
        "coverage": float(np.mean(cov)),
        "target": 1.0 - sets.alpha_nominal,
        "alpha_run": sets.alpha_run,
        "mean_set_size": float(np.mean(sets.size)),
        "abstain_rate": sets.abstain_rate,
        "decisive_rate": sets.decisive_rate,
        "empty_rate": sets.empty_rate,
        "review_rate": sets.review_rate,
        "q_hat": sets.q_hat,
        "method": sets.method,
        "n_cal": sets.n_cal,
    }
    for cls in (0, 1):
        sel = y == cls
        out[f"coverage_y{cls}"] = float(np.mean(cov[sel])) if sel.any() else float("nan")
    out["valid"] = bool(out["coverage"] >= out["target"])
    return out


def gamma_sensitivity(
    target_cal_scores, target_cal_labels, source_cal_scores, source_cal_labels,
    test_scores, test_labels, pi_s, pi_t, alpha=0.1, gammas=None,
) -> list[dict]:
    """Coverage and width of the hybrid procedure across concept-shift budgets."""
    gammas = np.linspace(0.0, alpha * 0.95, 8) if gammas is None else np.asarray(gammas, float)
    rows = []
    for g in gammas:
        cs = hybrid_transfer_conformal(
            target_cal_scores, target_cal_labels, source_cal_scores, source_cal_labels,
            test_scores, pi_s, pi_t, alpha=alpha, gamma=float(g),
        )
        r = coverage_report(cs, test_labels)
        r["gamma"] = float(g)
        rows.append(r)
    return rows
