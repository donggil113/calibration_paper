r"""Label-shift estimation, and its relaxation under partial concept shift.

Notation used throughout (and in the paper's Theorem 1):

* :math:`\pi_s, \pi_t` - source / target label priors.
* :math:`w_j = \pi_t(j)/\pi_s(j)` - importance weights.
* :math:`A \in \mathbb{R}^{M\times K}` - the *source operator*,
  :math:`A[m,j] = P_s(\text{bin}=m,\; y=j)`, estimated on held-out source data
  by binning the frozen model's scores into :math:`M` bins.
* :math:`q_t \in \Delta^{M-1}` - the target distribution over those same bins,
  estimated from **unlabeled** target data.

Classical BBSE (Lipton et al., 2018) assumes *pure* label shift,
:math:`P_t(x\mid y) = P_s(x\mid y)`, under which

.. math:: q_t = A w .

The premise of this study is that cross-national transfer violates that.  We
therefore work with the strictly weaker model

.. math:: q_t = A w + \eta, \qquad \mathbf{1}^\top \eta = 0,

where :math:`\eta` is a signed measure carrying the concept shift.  Without a
restriction on :math:`\eta` the pair :math:`(w, \eta)` is obviously not
identified - any :math:`w` can be rationalised.  This module implements the
three ways out, in increasing order of assumption strength:

1. :func:`partial_identification` - assume only :math:`\|\eta\|_1 \le \gamma`
   and return the *set* of priors consistent with the data (a linear program).
   Assumption-lean, gives intervals rather than points.
2. :func:`anchor_bbse` - assume an anchor region where concept shift is absent
   and identify :math:`w` from it alone.  Point-identified; this is the
   constructive half of Theorem 1.
3. :func:`bbse` / :func:`mlls` - assume :math:`\eta = 0`.  The classical
   estimators, recovered as the degenerate special case.

:func:`test_label_shift_sufficiency` closes the loop: with :math:`M > K` the
system is overdetermined, so pure label shift has a *testable implication*
that needs no target labels at all.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats
from scipy.optimize import linprog, nnls

__all__ = [
    "source_operator",
    "target_histogram",
    "LabelShiftEstimate",
    "bbse",
    "mlls",
    "anchor_bbse",
    "partial_identification",
    "min_gamma",
    "sensitivity_curve",
    "test_label_shift_sufficiency",
    "prior_correction",
    "importance_weights",
]


# --------------------------------------------------------------------------
# building the linear system
# --------------------------------------------------------------------------
def source_operator(
    scores: np.ndarray,
    labels: np.ndarray,
    edges: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Estimate :math:`A[m,j] = P_s(\text{bin}=m, y=j)` and the prior :math:`\pi_s`.

    Binary (one-vs-rest) form: ``scores`` is 1-D, ``labels`` in {0,1}, and the
    returned operator has ``K = 2`` columns ordered ``[y=0, y=1]``.

    The operator must be estimated on data the model did not train on.  Using
    training data makes ``A`` reflect a sharper score distribution than the
    model actually produces out of sample, which biases every downstream prior
    estimate toward the source prior - i.e. it hides exactly the effect we are
    measuring.
    """
    from .metrics import bin_assign

    scores = np.asarray(scores, float).ravel()
    labels = np.asarray(labels, float).ravel()
    m_idx = bin_assign(scores, edges)
    M = len(edges) - 1
    A = np.zeros((M, 2))
    n = max(len(scores), 1)
    for j in (0, 1):
        sel = labels == j
        A[:, j] = np.bincount(m_idx[sel], minlength=M) / n
    pi_s = np.array([1.0 - labels.mean(), labels.mean()])
    return A, pi_s


def target_histogram(scores: np.ndarray, edges: np.ndarray) -> np.ndarray:
    r"""Estimate :math:`q_t` from **unlabeled** target scores."""
    from .metrics import bin_assign

    scores = np.asarray(scores, float).ravel()
    M = len(edges) - 1
    counts = np.bincount(bin_assign(scores, edges), minlength=M)
    return counts / max(counts.sum(), 1)


# --------------------------------------------------------------------------
# estimates
# --------------------------------------------------------------------------
@dataclass
class LabelShiftEstimate:
    """Result of a prior estimator.

    ``pi_t`` is the estimated target prior, ``w`` the importance weights,
    ``eta`` the implied concept-shift residual on the bin simplex, and
    ``residual_l1`` its :math:`\\ell_1` norm - a direct, unlabeled read-out of
    how far the site is from pure label shift.
    """

    pi_t: np.ndarray
    w: np.ndarray
    eta: np.ndarray
    residual_l1: float
    method: str
    diagnostics: dict

    @property
    def prevalence(self) -> float:
        """Estimated target prevalence of the positive class."""
        return float(self.pi_t[1])


def _finish(A, pi_s, q_t, w, method, diagnostics) -> LabelShiftEstimate:
    w = np.maximum(np.asarray(w, float), 0.0)
    pi_t = pi_s * w
    total = pi_t.sum()
    if total <= 0:
        pi_t = pi_s.copy()
        w = np.ones_like(w)
    else:
        pi_t = pi_t / total
        w = np.divide(pi_t, pi_s, out=np.ones_like(pi_t), where=pi_s > 0)
    eta = q_t - A @ w
    return LabelShiftEstimate(
        pi_t=pi_t, w=w, eta=eta, residual_l1=float(np.abs(eta).sum()),
        method=method, diagnostics=diagnostics,
    )


def bbse(A: np.ndarray, pi_s: np.ndarray, q_t: np.ndarray, nonneg: bool = True) -> LabelShiftEstimate:
    r"""Black-box shift estimation: solve :math:`q_t = A w` in least squares.

    With ``M > K`` bins this is the *soft* / distribution-matching form, which
    is strictly more efficient than the original confusion-matrix version
    because it uses the whole score histogram rather than a thresholded label.
    ``nonneg`` enforces :math:`w \ge 0` (weights are ratios of probabilities),
    which matters at rare labels where the unconstrained solution routinely
    goes negative.
    """
    A = np.asarray(A, float)
    q_t = np.asarray(q_t, float).ravel()
    if nonneg:
        w, _ = nnls(A, q_t)
    else:
        w, *_ = np.linalg.lstsq(A, q_t, rcond=None)
    sv = np.linalg.svd(A, compute_uv=False)
    diag = {
        "sigma_min": float(sv[-1]),
        "cond": float(sv[0] / sv[-1]) if sv[-1] > 0 else float("inf"),
        "n_bins": int(A.shape[0]),
    }
    return _finish(A, np.asarray(pi_s, float), q_t, w, "bbse", diag)


def mlls(
    posteriors: np.ndarray,
    pi_s: np.ndarray,
    max_iter: int = 1000,
    tol: float = 1e-9,
) -> LabelShiftEstimate:
    r"""Maximum-likelihood label shift (EM of Saerens et al., 2002).

    ``posteriors`` are the source-model posteriors :math:`p_s(y\mid x)` on the
    unlabeled target sample, shape ``(n, K)``.

    Caveat carried through to the paper: MLLS is consistent only if those
    posteriors are *calibrated on the source domain*.  Feeding it the raw
    outputs of an uncalibrated network makes it confidently wrong, so the
    pipeline always temperature-scales on a source validation split first.
    """
    P = np.asarray(posteriors, float)
    if P.ndim == 1:
        P = np.column_stack([1 - P, P])
    pi_s = np.asarray(pi_s, float)
    pi = pi_s.copy()
    n_iter, delta = 0, float("inf")
    for n_iter in range(1, max_iter + 1):
        w = np.divide(pi, pi_s, out=np.ones_like(pi), where=pi_s > 0)
        num = P * w
        denom = num.sum(axis=1, keepdims=True)
        post = np.divide(num, denom, out=np.full_like(num, 1.0 / P.shape[1]), where=denom > 0)
        pi_new = post.mean(axis=0)
        delta = float(np.abs(pi_new - pi).max())
        pi = pi_new
        if delta < tol:
            break
    w = np.divide(pi, pi_s, out=np.ones_like(pi), where=pi_s > 0)
    est = LabelShiftEstimate(
        pi_t=pi, w=w, eta=np.zeros(1), residual_l1=float("nan"), method="mlls",
        diagnostics={"n_iter": n_iter, "delta": delta, "converged": delta < tol},
    )
    return est


def anchor_bbse(
    A: np.ndarray,
    pi_s: np.ndarray,
    q_t: np.ndarray,
    anchor: np.ndarray,
) -> LabelShiftEstimate:
    r"""Identify :math:`w` from an anchor region assumed free of concept shift.

    ``anchor`` is a boolean mask over bins marking where
    :math:`P_t(x\mid y) = P_s(x\mid y)` is credible.  Theorem 1 shows
    :math:`(w,\eta)` is identified as soon as the anchor rows
    :math:`A_{\mathcal{A}}` have full column rank - a condition that is
    *checkable from source data alone*, and strictly weaker than BBSE's
    requirement that concept shift be absent everywhere.

    Choosing the anchor is a modelling decision that must be defended, not a
    free parameter to tune.  In the ECG setting the defensible choice is the
    low-score region: sites differ in which abnormal morphologies they see, but
    a clearly normal tracing is a clearly normal tracing in any country.  The
    estimate's sensitivity to that choice is reported, never suppressed.
    """
    A = np.asarray(A, float)
    q_t = np.asarray(q_t, float).ravel()
    anchor = np.asarray(anchor, bool).ravel()
    if anchor.size != A.shape[0]:
        raise ValueError("anchor mask must have one entry per bin")
    if anchor.sum() < A.shape[1]:
        raise ValueError(
            f"anchor region has {int(anchor.sum())} bins but {A.shape[1]} classes: rank condition cannot hold"
        )
    A_a, q_a = A[anchor], q_t[anchor]
    rank = int(np.linalg.matrix_rank(A_a))
    sv = np.linalg.svd(A_a, compute_uv=False)
    w, _ = nnls(A_a, q_a)
    diag = {
        "anchor_bins": int(anchor.sum()),
        "anchor_rank": rank,
        "rank_deficient": rank < A.shape[1],
        "sigma_min_anchor": float(sv[-1]),
        "anchor_mass_source": float(A[anchor].sum()),
        "anchor_mass_target": float(q_t[anchor].sum()),
    }
    return _finish(A, np.asarray(pi_s, float), q_t, w, "anchor_bbse", diag)


def min_gamma(A: np.ndarray, pi_s: np.ndarray, q_t: np.ndarray) -> float:
    r"""Smallest concept-shift budget consistent with the observed histogram.

    .. math:: \gamma_{\min} = \min_{w\ge 0,\ \pi_s^\top w = 1} \|q_t - Aw\|_1 .

    Any :math:`\gamma < \gamma_{\min}` makes the identified set empty: the data
    refute a concept shift that small.  :math:`\gamma_{\min}` is therefore a
    *lower confidence bound on how non-label-shift a site is*, computable
    without a single target label, and it is the natural x-axis origin for the
    sensitivity sweep in :func:`sensitivity_curve`.

    It must not be mistaken for an estimate of the true :math:`\|\eta\|_1`,
    which is not identified - it is the most optimistic value compatible with
    the evidence.
    """
    A = np.asarray(A, float)
    q_t = np.asarray(q_t, float).ravel()
    pi_s = np.asarray(pi_s, float).ravel()
    M, K = A.shape
    obj = np.concatenate([np.zeros(K), np.ones(M)])
    ub_rows = np.vstack([np.hstack([A, -np.eye(M)]), np.hstack([-A, -np.eye(M)])])
    ub_vals = np.concatenate([q_t, -q_t])
    eq_rows = np.hstack([pi_s, np.zeros(M)]).reshape(1, -1)
    res = linprog(obj, A_ub=ub_rows, b_ub=ub_vals, A_eq=eq_rows, b_eq=np.array([1.0]),
                  bounds=[(0, None)] * (K + M), method="highs")
    return float(res.fun) if res.success else float("nan")


def partial_identification(
    A: np.ndarray,
    pi_s: np.ndarray,
    q_t: np.ndarray,
    gamma: float,
    class_index: int = 1,
) -> dict:
    r"""Sharp bounds on :math:`\pi_t` when concept shift is only bounded.

    Solves, by linear programming,

    .. math::
        \min / \max \; \pi_s(k) w_k
        \quad\text{s.t.}\quad w \ge 0,\;
        \|q_t - Aw\|_1 \le \gamma,\;
        \pi_s^\top w = 1 .

    The returned interval is *sharp*: every value inside it is attained by some
    admissible concept shift of size at most :math:`\gamma`, and no value
    outside it is.  This is the honest answer when no anchor region can be
    defended, and it is what the collapse map should display - a site whose
    interval is wide has not been shown to be safe, it has been shown to be
    unknowable from unlabeled data alone.

    ``gamma`` is a **sensitivity parameter, not an estimand**.  It cannot be
    read off the data: the residual left by BBSE is a *lower* bound on the true
    :math:`\|\eta\|_1`, because BBSE chooses :math:`w` to make that residual
    small.  Setting ``gamma`` to the BBSE residual therefore produces an
    interval that is too narrow and can exclude the truth.  Use
    :func:`min_gamma` for the smallest feasible value and
    :func:`sensitivity_curve` to sweep upward from it; report the sweep, not a
    single interval.

    Values below :math:`\gamma_{\min}` leave an empty feasible set; that is
    returned explicitly as ``feasible=False`` rather than as a silent NaN.
    """
    A = np.asarray(A, float)
    q_t = np.asarray(q_t, float).ravel()
    pi_s = np.asarray(pi_s, float).ravel()
    M, K = A.shape

    # Variables z = [w (K), u (M)] with u >= |q_t - A w| encoded linearly.
    n_var = K + M
    # A w - q_t - u <= 0  and  -(A w - q_t) - u <= 0
    ub_rows = np.vstack([
        np.hstack([A, -np.eye(M)]),
        np.hstack([-A, -np.eye(M)]),
    ])
    ub_vals = np.concatenate([q_t, -q_t])
    # sum(u) <= gamma
    ub_rows = np.vstack([ub_rows, np.hstack([np.zeros(K), np.ones(M)])])
    ub_vals = np.concatenate([ub_vals, [gamma]])
    # pi_s^T w == 1  (the corrected prior is a probability vector)
    eq_rows = np.hstack([pi_s, np.zeros(M)]).reshape(1, -1)
    eq_vals = np.array([1.0])

    obj = np.zeros(n_var)
    obj[class_index] = pi_s[class_index]
    bounds = [(0, None)] * n_var

    out = {}
    for sense, sign in (("lo", 1.0), ("hi", -1.0)):
        res = linprog(
            sign * obj, A_ub=ub_rows, b_ub=ub_vals, A_eq=eq_rows, b_eq=eq_vals,
            bounds=bounds, method="highs",
        )
        out[sense] = float(sign * res.fun) if res.success else float("nan")
        out[f"{sense}_status"] = res.status
    point = bbse(A, pi_s, q_t)
    feasible = bool(np.isfinite(out["lo"]) and np.isfinite(out["hi"]))
    out.update(
        {
            "gamma": float(gamma),
            "gamma_min": min_gamma(A, pi_s, q_t),
            "point": float(point.pi_t[class_index]),
            "width": (out["hi"] - out["lo"]) if feasible else float("nan"),
            "feasible": feasible,
            "point_identified": bool(feasible and out["hi"] - out["lo"] < 1e-8),
        }
    )
    return out


def sensitivity_curve(
    A: np.ndarray,
    pi_s: np.ndarray,
    q_t: np.ndarray,
    gammas: np.ndarray | None = None,
    class_index: int = 1,
    n_points: int = 12,
) -> dict:
    """Identified interval for the target prior as the concept-shift budget grows.

    This is the figure a regulator should be shown: not a point estimate of the
    target prevalence, but how fast the claim degrades as one relaxes the
    assumption that the two populations differ only in prevalence.  A site
    where the interval stays tight out to large ``gamma`` is safe to correct
    without labels; one where it fans out immediately is not.
    """
    A = np.asarray(A, float)
    g_min = min_gamma(A, pi_s, q_t)
    if gammas is None:
        base = g_min if np.isfinite(g_min) and g_min > 0 else 1e-3
        gammas = np.linspace(base, max(base * 8.0, base + 0.20), n_points)
    rows = [partial_identification(A, pi_s, q_t, float(g), class_index) for g in gammas]
    return {
        "gamma": np.asarray(gammas, float),
        "lo": np.array([r["lo"] for r in rows]),
        "hi": np.array([r["hi"] for r in rows]),
        "width": np.array([r["width"] for r in rows]),
        "gamma_min": g_min,
        "point": rows[0]["point"],
    }


# --------------------------------------------------------------------------
# testing the label-shift assumption without target labels
# --------------------------------------------------------------------------
def _pearson_statistic(
    A: np.ndarray, q_t: np.ndarray, n_t: int, n_s: int | None, min_expected: float
) -> tuple[float, int, np.ndarray]:
    """Pooled Pearson divergence between the target histogram and its best
    label-shift fit, with the per-bin source-estimation variance folded in.

    Returns ``(statistic, dof, fitted)``.
    """
    w, _ = nnls(A, q_t)
    fitted = np.maximum(A @ w, 0.0)
    if fitted.sum() > 0:
        fitted = fitted / fitted.sum()
    src_var = (A * (w**2)[None, :]).sum(axis=1) if n_s else np.zeros_like(fitted)

    keep_o, keep_e, keep_v = [], [], []
    acc_o = acc_e = acc_v = 0.0
    for o, e, v in zip(q_t, fitted, src_var):
        acc_o += o
        acc_e += e
        acc_v += v
        if acc_e * n_t >= min_expected:
            keep_o.append(acc_o)
            keep_e.append(acc_e)
            keep_v.append(acc_v)
            acc_o = acc_e = acc_v = 0.0
    if acc_e > 0 and keep_e:
        keep_o[-1] += acc_o
        keep_e[-1] += acc_e
        keep_v[-1] += acc_v
    obs, exp, svar = np.array(keep_o), np.array(keep_e), np.array(keep_v)
    dof = max(len(obs) - A.shape[1], 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        infl = (
            1.0 + (float(n_t) / float(n_s)) * np.where(exp > 0, svar / np.maximum(exp, 1e-300), 0.0)
            if n_s
            else np.ones_like(exp)
        )
        terms = np.where(exp > 0, (obs - exp) ** 2 / (exp * infl), 0.0)
    return float(n_t * terms.sum()), int(dof), fitted


def test_label_shift_sufficiency(
    A: np.ndarray,
    pi_s: np.ndarray,
    q_t: np.ndarray,
    n_t: int,
    n_s: int | None = None,
    min_expected: float = 5.0,
    n_mc: int = 500,
    seed: int = 0,
) -> dict:
    r"""Goodness-of-fit test of :math:`H_0:\; q_t = Aw` for some :math:`w \ge 0`.

    With :math:`M > K` bins the pure-label-shift model is overdetermined, so it
    has an implication testable on **unlabeled** target data alone.  The
    statistic is the minimum Pearson divergence

    .. math:: T = n_t \min_{w\ge 0} \sum_m \frac{(\hat q_t[m] - (Aw)[m])^2}{(Aw)[m]},

    which is asymptotically :math:`\chi^2_{M-K}` under :math:`H_0`.

    Operationally this is the most deployable thing in the package: a hospital
    can run it on its own unlabeled ECGs, before any chart review, and learn
    whether a free unlabeled prior correction will suffice or whether it must
    pay for labels.  A rejection says the calibration cliff at that site is not
    reducible to prevalence.

    ``n_s`` corrects for ``A`` itself being estimated.  That correction is
    applied **per bin**: the source contributes variance
    :math:`\sum_j w_j^2 A[m,j]/n_s`, which is weighted by the importance
    weights and so is far from uniform across the score range.  A single scalar
    factor understates it exactly where the reweighting is most aggressive.
    Bins with expected count below ``min_expected`` are pooled into their
    neighbour, as Pearson's test requires.

    ``n_mc`` replicates calibrate the null distribution by parametric
    bootstrap, which is the default because the chi-square reference is
    measurably anti-conservative at realistic sample sizes.  Set ``n_mc=0`` to
    fall back to the asymptotic reference; the chi-square p-value is returned
    alongside either way as ``p_value_chi2``.
    """
    A = np.asarray(A, float)
    q_t = np.asarray(q_t, float).ravel()
    est = bbse(A, pi_s, q_t)
    stat, dof, fitted = _pearson_statistic(A, q_t, n_t, n_s, min_expected)

    if n_mc and n_mc > 0:
        # Parametric bootstrap null.  The asymptotic chi-square reference is
        # anti-conservative here - measured size 0.09-0.12 against a nominal
        # 0.05 - because the multinomial cells of A covary and the fitted w
        # carries its own sampling error, neither of which the degrees-of-freedom
        # adjustment captures.  Simulating the null instead makes no asymptotic
        # approximation: draw a target histogram from the fitted label-shift
        # model and, when n_s is known, a fresh source operator from its own
        # multinomial, then recompute the statistic.
        #
        # This matters operationally rather than cosmetically: an inflated size
        # means flagging concept shift at sites that have none, and the decision
        # this test feeds is whether a hospital must pay for labels.
        rng = np.random.default_rng(seed)
        null = np.empty(int(n_mc))
        cell_p = (A / A.sum()).ravel()
        shape = A.shape
        for b in range(int(n_mc)):
            q_b = rng.multinomial(n_t, fitted) / float(n_t)
            A_b = (
                (rng.multinomial(int(n_s), cell_p).reshape(shape) / float(n_s))
                if n_s
                else A
            )
            null[b], _, _ = _pearson_statistic(A_b, q_b, n_t, n_s, min_expected)
        # +1 smoothing keeps the p-value from ever being exactly zero, which
        # would overstate the evidence beyond what n_mc replicates can support.
        p = float((1.0 + np.sum(null >= stat)) / (1.0 + len(null)))
        reference = "monte_carlo"
    else:
        p = float(stats.chi2.sf(stat, dof))
        null = np.empty(0)
        reference = "chi2_asymptotic"

    return {
        "statistic": stat,
        "dof": int(dof),
        "p_value": p,
        "reject_label_shift": bool(p < 0.05),
        "reference": reference,
        "n_mc": int(n_mc or 0),
        "p_value_chi2": float(stats.chi2.sf(stat, dof)),
        "bins_used": int(len(fitted)),
        "bins_original": int(A.shape[0]),
        "residual_l1": est.residual_l1,
        "pi_t_hat": float(est.pi_t[1]),
    }


# --------------------------------------------------------------------------
# applying a prior correction
# --------------------------------------------------------------------------
def importance_weights(pi_s: float | np.ndarray, pi_t: float | np.ndarray) -> np.ndarray:
    """Class weights :math:`w = \\pi_t/\\pi_s` for binary or K-class priors."""
    if np.isscalar(pi_s):
        pi_s = np.array([1.0 - float(pi_s), float(pi_s)])
    if np.isscalar(pi_t):
        pi_t = np.array([1.0 - float(pi_t), float(pi_t)])
    pi_s = np.asarray(pi_s, float)
    pi_t = np.asarray(pi_t, float)
    return np.divide(pi_t, pi_s, out=np.ones_like(pi_t), where=pi_s > 0)


def prior_correction(scores: np.ndarray, pi_s: float, pi_t: float, eps: float = 1e-12) -> np.ndarray:
    r"""The map :math:`T_{\pi_s\to\pi_t}` of Theorem 1.

    .. math::
        T(v) \;=\; \frac{w_1 v}{w_1 v + w_0 (1-v)},
        \qquad w_1 = \frac{\pi_t}{\pi_s},\; w_0 = \frac{1-\pi_t}{1-\pi_s}.

    Under pure label shift this map is *exactly* the recalibration that
    restores target calibration, and it needs no target labels - only
    :math:`\pi_t`, which :func:`bbse` gets from unlabeled data.  That is the
    entire content of the paper's "free half" of the cliff.  Everything
    :math:`T` fails to fix is, by construction, concept shift.
    """
    v = np.clip(np.asarray(scores, float), 0.0, 1.0)
    w1 = pi_t / max(pi_s, eps)
    w0 = (1.0 - pi_t) / max(1.0 - pi_s, eps)
    num = w1 * v
    den = num + w0 * (1.0 - v)
    return np.divide(num, den, out=np.full_like(v, pi_t), where=den > eps)
