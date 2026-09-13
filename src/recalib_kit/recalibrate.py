r"""Recalibrators, ordered by how many target labels they cost.

The study's operational claim is that the cost of making a transferred model
safe is not "collect a new training set" but a specific, small number of
labeled target ECGs - and that the number is small precisely because the
label-shift component is free.  That claim only means something if the free
part and the paid part are implemented as separate, composable objects, so
they are:

============================  ====================  ============================
estimator                     target labels needed  removes
============================  ====================  ============================
:class:`Identity`             0                     nothing (the status quo)
:class:`PrevalenceCorrection` n (one scalar)        :math:`D_{\text{label}}`, with the
                                                    prevalence measured rather
                                                    than inferred  **(best use of
                                                    a small budget)**
:class:`PriorCorrection`      0 (unlabeled only)    :math:`D_{\text{label}}`
:class:`TemperatureScaling`   n (1 parameter)       smooth global distortion
:class:`PlattScaling`         n (2 parameters)      affine logit distortion
:class:`Isotonic`             n (nonparametric)     any monotone distortion
:class:`ScalingBinning`       n                     as isotonic, with a
                                                    verifiable ECE guarantee
:class:`HybridPriorFewShot`   n (1 parameter)       both, warm-started by the
                                                    free correction
:class:`GatedPriorCorrection` 0 (unlabeled only)    :math:`D_{\text{label}}`, but
                                                    only where an unlabeled test
                                                    licenses it
:class:`MinimaxPriorCorrection` 0 (unlabeled only)  :math:`D_{\text{label}}`, damped
                                                    by how little the data pins
                                                    the prior  **(recommended)**
:class:`GatedHybrid`          n (1 parameter)       both, gated the same way
:class:`MinimaxHybrid`        n (1 parameter)       both, damped the same way
============================  ====================  ============================

:class:`MinimaxPriorCorrection` is the estimator we recommend for unlabeled
deployment.  The plain correction is not risk-free: where the label-shift
assumption fails, the prior it depends on can be wrong by an order of magnitude,
and applying it then makes calibration substantially worse than shipping the
model unchanged.  :class:`GatedPriorCorrection` blocks that case with a hard
yes/no, but inherits the test's conditional size and can shut on a site that
deserved the correction.  The minimax version replaces the decision with an
optimisation over the identified set and was never worse than shipping
unchanged across the regimes we tested.

:class:`HybridPriorFewShot` is the estimator Theorem 2 is about.  It applies
the unlabeled prior correction first and spends its labels only on the
*residual* concept shift, so its sample complexity scales with the size of
that residual rather than with the total shift.  When a site is close to pure
label shift it needs almost no labels; when it is not, it degrades gracefully
to plain temperature scaling.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit

from .label_shift import bbse, prior_correction, source_operator, target_histogram
from .metrics import bin_edges, bin_assign

__all__ = [
    "Recalibrator",
    "Identity",
    "PriorCorrection",
    "PrevalenceCorrection",
    "PrevalenceCorrection",
    "TemperatureScaling",
    "PlattScaling",
    "Isotonic",
    "ScalingBinning",
    "HybridPriorFewShot",
    "GatedPriorCorrection",
    "MinimaxPriorCorrection",
    "GatedHybrid",
    "MinimaxHybrid",
    "SelectByCrossValidation",
    "fit_recalibrator",
    "METHODS",
]

_EPS = 1e-6


def _safe_logit(p: np.ndarray) -> np.ndarray:
    return logit(np.clip(np.asarray(p, float), _EPS, 1 - _EPS))


def _nll(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(p, _EPS, 1 - _EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


class Recalibrator:
    """Base class: ``fit`` on a target calibration split, ``transform`` scores."""

    name = "base"
    n_labels_required = 0

    def fit(self, scores: np.ndarray, labels: np.ndarray | None = None, **kw) -> "Recalibrator":
        return self

    def transform(self, scores: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def fit_transform(self, scores, labels=None, **kw) -> np.ndarray:
        return self.fit(scores, labels, **kw).transform(scores)

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


class Identity(Recalibrator):
    """Ship the source model unchanged: the baseline the cliff is measured against."""

    name = "identity"

    def transform(self, scores: np.ndarray) -> np.ndarray:
        return np.clip(np.asarray(scores, float), 0.0, 1.0)


@dataclass
class PriorCorrection(Recalibrator):
    r"""The free correction: apply :math:`T_{\pi_s\to\pi_t}` with an unlabeled prior.

    ``pi_t`` may be supplied directly, or estimated by :meth:`fit_unlabeled`
    from target scores plus a held-out labeled *source* split.  No target
    labels are consumed in either case, which is what makes this deployable at
    a site that has not yet reviewed a single chart.
    """

    pi_s: float = 0.5
    pi_t: float | None = None
    name: str = "prior_correction"
    n_labels_required: int = 0

    def fit_unlabeled(
        self,
        target_scores: np.ndarray,
        source_scores: np.ndarray,
        source_labels: np.ndarray,
        n_bins: int = 15,
    ) -> "PriorCorrection":
        edges = bin_edges(np.asarray(source_scores, float), n_bins, "equal_mass")
        A, pi_s_vec = source_operator(source_scores, source_labels, edges)
        est = bbse(A, pi_s_vec, target_histogram(target_scores, edges))
        self.pi_s = float(pi_s_vec[1])
        self.pi_t = float(est.prevalence)
        return self

    def fit(self, scores, labels=None, **kw) -> "PriorCorrection":
        if self.pi_t is None:
            if labels is None:
                raise ValueError("PriorCorrection needs pi_t, or labels, or fit_unlabeled()")
            self.pi_t = float(np.asarray(labels, float).mean())
        return self

    def transform(self, scores: np.ndarray) -> np.ndarray:
        if self.pi_t is None:
            raise RuntimeError("PriorCorrection is not fitted")
        return prior_correction(scores, self.pi_s, self.pi_t)


class PrevalenceCorrection(Recalibrator):
    r"""Estimate the target prevalence **from labels**, then apply :math:`T`.

    The cheapest thing a site can buy, and empirically the best use of a very
    small label budget.  It spends its labels on a single scalar - the target
    prevalence - rather than on a whole recalibration map, so it converges at
    the rate of a proportion rather than of a function.

    This estimator exists because the unlabeled route to the same correction
    turned out not to work.  In our transfer matrix, applying :math:`T` with the
    *true* prevalence removes about 40% of the target calibration error, and at
    the worst-hit site nearly 60%.  Applying it with a prevalence estimated from
    unlabeled data by BBSE makes calibration roughly three times *worse*,
    because cross-national concept shift moves the score distribution along the
    label-shift cone, where Theorem 1 says the prevalence is not identified and
    the unlabeled goodness-of-fit test is structurally blind.

    The labelled version needs very little: in our matrix 25 adjudicated target
    cases already beat shipping unchanged, and 50 recover essentially the whole
    oracle benefit.  Above roughly 100 labels a nonparametric recalibration
    (:class:`Isotonic`, :class:`PlattScaling`) dominates, so this occupies a
    specific and narrow band of the budget - which is exactly the band a site
    starting a deployment is in.

    ``min_positives`` guards the degenerate draw: a calibration sample
    containing no positive case would set the prevalence to zero and collapse
    every score, so below that count the estimator falls back to the identity
    rather than acting on a prevalence it has not observed.
    """

    name = "prevalence_correction"

    def __init__(self, pi_s: float = 0.5, min_positives: int = 3):
        self.pi_s = pi_s
        self.min_positives = min_positives
        self.pi_t: float | None = None
        self.n_positive_: int = 0

    def fit(self, scores, labels=None, **kw) -> "PrevalenceCorrection":
        if labels is None:
            raise ValueError("prevalence_correction estimates the target prior from labels")
        y = np.asarray(labels, float).ravel()
        self.n_positive_ = int(y.sum())
        if y.size == 0 or self.n_positive_ < self.min_positives:
            self.pi_t = None                       # not enough evidence: do nothing
        else:
            self.pi_t = float(y.mean())
        return self

    def transform(self, scores):
        v = np.clip(np.asarray(scores, float), 0.0, 1.0)
        if self.pi_t is None:
            return v
        return prior_correction(v, self.pi_s, self.pi_t)

    # The only requirement in this module that depends on instance state, so it
    # stays a property.  Callers must read it from an INSTANCE, never from the
    # class: a property accessed on the class is a descriptor, not a number.
    @property
    def n_labels_required(self) -> int:
        return max(self.min_positives, 1)


class TemperatureScaling(Recalibrator):
    r"""Divide the logit by a single scalar :math:`T > 0`.

    One parameter, so it converges at the parametric :math:`n^{-1/2}` rate and
    is the right choice in the few-shot regime this paper cares about.  Its
    limitation is structural: it cannot change the *sign* of the miscalibration
    across the score range, so it cannot by itself repair a prior shift, which
    moves low and high scores in opposite directions.  That is the analytic
    reason the hybrid estimator exists.
    """

    name = "temperature"

    def __init__(self, bounds: tuple[float, float] = (0.05, 20.0)):
        self.bounds = bounds
        self.temperature_: float | None = None

    def fit(self, scores, labels=None, **kw) -> "TemperatureScaling":
        z = _safe_logit(scores)
        y = np.asarray(labels, float).ravel()
        if y.size == 0 or y.sum() in (0, y.size):
            self.temperature_ = 1.0      # degenerate split: refuse to move
            return self
        res = minimize_scalar(
            lambda t: _nll(expit(z / t), y), bounds=self.bounds, method="bounded"
        )
        self.temperature_ = float(res.x)
        return self

    n_labels_required = 1

    def transform(self, scores):
        if self.temperature_ is None:
            raise RuntimeError("not fitted")
        return expit(_safe_logit(scores) / self.temperature_)


class PlattScaling(Recalibrator):
    """Affine map on the logit, ``sigma(a*z + b)``: two parameters."""

    name = "platt"
    n_labels_required = 2

    def __init__(self):
        self.a_: float | None = None
        self.b_: float | None = None

    def fit(self, scores, labels=None, **kw) -> "PlattScaling":
        from scipy.optimize import minimize

        z = _safe_logit(scores)
        y = np.asarray(labels, float).ravel()
        if y.size == 0 or y.sum() in (0, y.size):
            self.a_, self.b_ = 1.0, 0.0
            return self
        res = minimize(
            lambda p: _nll(expit(p[0] * z + p[1]), y), x0=np.array([1.0, 0.0]), method="Nelder-Mead"
        )
        self.a_, self.b_ = float(res.x[0]), float(res.x[1])
        return self

    def transform(self, scores):
        if self.a_ is None:
            raise RuntimeError("not fitted")
        return expit(self.a_ * _safe_logit(scores) + self.b_)


class Isotonic(Recalibrator):
    """Nonparametric monotone fit (PAV).

    Flexible, but it overfits hard at the sample sizes a small site can afford
    and gives no finite-sample calibration guarantee - it is included as the
    strong-data ceiling, not as a recommendation for few-shot deployment.
    """

    name = "isotonic"
    n_labels_required = 50

    def __init__(self, out_of_bounds: str = "clip"):
        self.out_of_bounds = out_of_bounds
        self.model_ = None

    def fit(self, scores, labels=None, **kw) -> "Isotonic":
        from sklearn.isotonic import IsotonicRegression

        y = np.asarray(labels, float).ravel()
        self.model_ = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds=self.out_of_bounds)
        self.model_.fit(np.asarray(scores, float).ravel(), y)
        return self

    def transform(self, scores):
        if self.model_ is None:
            raise RuntimeError("not fitted")
        return np.clip(self.model_.predict(np.asarray(scores, float).ravel()), 0.0, 1.0)


class ScalingBinning(Recalibrator):
    """Scaling-binning of Kumar, Liang & Ma (2019).

    Fit a smooth parametric map on one half of the calibration split, then
    discretise it into equal-mass bins using the other half.  The discretisation
    is what makes the resulting calibration error *measurable* from a finite
    sample - a plain scaling map has calibration error that cannot be verified
    without additional assumptions.  For a deployment claim that a regulator is
    expected to check, verifiability is worth the small loss in sharpness.
    """

    name = "scaling_binning"
    n_labels_required = 100

    def __init__(self, n_bins: int = 15, base: Recalibrator | None = None, seed: int = 0):
        self.n_bins = n_bins
        self.base = base or PlattScaling()
        self.seed = seed
        self.edges_: np.ndarray | None = None
        self.values_: np.ndarray | None = None

    def fit(self, scores, labels=None, **kw) -> "ScalingBinning":
        s = np.asarray(scores, float).ravel()
        y = np.asarray(labels, float).ravel()
        rng = np.random.default_rng(self.seed)
        perm = rng.permutation(len(s))
        half = len(s) // 2
        i1, i2 = perm[:half], perm[half:]
        self.base.fit(s[i1], y[i1])
        g2 = self.base.transform(s[i2])
        self.edges_ = bin_edges(g2, self.n_bins, "equal_mass")
        idx = bin_assign(g2, self.edges_)
        nb = len(self.edges_) - 1
        cnt = np.bincount(idx, minlength=nb)
        tot = np.bincount(idx, weights=g2, minlength=nb)
        # bin value = mean of the *fitted function*, not of the labels: this is
        # the variance-reduction step that gives the estimator its guarantee.
        self.values_ = np.where(cnt > 0, tot / np.maximum(cnt, 1), 0.0)
        empty = cnt == 0
        if empty.any() and (~empty).any():
            self.values_[empty] = np.interp(
                np.flatnonzero(empty), np.flatnonzero(~empty), self.values_[~empty]
            )
        return self

    def transform(self, scores):
        if self.edges_ is None:
            raise RuntimeError("not fitted")
        g = self.base.transform(np.asarray(scores, float).ravel())
        return np.clip(self.values_[bin_assign(g, self.edges_)], 0.0, 1.0)


class HybridPriorFewShot(Recalibrator):
    r"""**The estimator of Theorem 2**: free prior correction, then paid residual fix.

    ``transform`` computes :math:`\sigma\big(\mathrm{logit}(T(v))/\hat\tau\big)`,
    where :math:`T` is the unlabeled prior correction and :math:`\hat\tau` is a
    single temperature fitted on :math:`n` labeled target examples.

    The point is where the labels go.  Plain temperature scaling spends them
    undoing the prior shift - a distortion whose size is already known from
    unlabeled data - and only then, with whatever precision is left, on the
    concept shift.  Warm-starting at :math:`T` means every label is spent on
    the part that genuinely requires them, so the required :math:`n` scales
    with the residual concept shift :math:`\Gamma` rather than with the total.
    That is the whole content of the sample-complexity bound.
    """

    name = "hybrid"

    def __init__(self, pi_s: float = 0.5, pi_t: float | None = None, bounds=(0.05, 20.0)):
        self.prior = PriorCorrection(pi_s=pi_s, pi_t=pi_t)
        self.temp = TemperatureScaling(bounds=bounds)

    def fit_unlabeled(self, target_scores, source_scores, source_labels, n_bins: int = 15):
        self.prior.fit_unlabeled(target_scores, source_scores, source_labels, n_bins)
        return self

    def fit(self, scores, labels=None, **kw) -> "HybridPriorFewShot":
        if self.prior.pi_t is None:
            raise RuntimeError("call fit_unlabeled() first: the prior must come from unlabeled data")
        self.temp.fit(self.prior.transform(scores), labels)
        return self

    n_labels_required = 1

    def transform(self, scores):
        return self.temp.transform(self.prior.transform(scores))

    @property
    def temperature_(self) -> float | None:
        return self.temp.temperature_


class GatedPriorCorrection(Recalibrator):
    r"""Prior correction that applies itself **only where it is licensed**.

    The free correction is not free of risk.  It depends on an estimate of
    :math:`\pi_t` obtained under the label-shift assumption, and where that
    assumption fails the estimate can be wrong by an order of magnitude - in
    which case applying :math:`T` makes calibration substantially *worse* than
    shipping the model unchanged.  In the transfer matrix reported here, the
    ungated correction raised mean ECE from 0.052 to 0.170.

    This estimator therefore runs the unlabeled goodness-of-fit test of
    Proposition 3 first, and applies the correction only when

    * the test does not reject pure label shift at ``level``, and
    * the prior estimate is not pinned at a simplex boundary (a failed solve,
      not a measurement).

    Otherwise it falls back to the identity, which is the honest default: a site
    that cannot verify the assumption has not earned the correction.  The
    decision and its evidence are recorded in :attr:`gate_`, so a deployment can
    audit why a site was or was not corrected.

    It converts the theory into a rule a hospital can follow without a
    statistician in the room: run the test on your own unlabeled ECGs; if it
    passes, take the free fix; if it fails, you need labels.

    **Caveat, and why this is not the default.**  The gate inherits the
    goodness-of-fit test's operating characteristics, and those are worse in
    deployment than their unconditional values suggest.  A deploying site uses
    one source operator, estimated once by whoever shipped the model, and its
    estimation error is then *common to every site*.  Conditional on one such
    operator we measured the gate staying shut on 2 of 5 pure-label-shift
    targets whose median null p-value was 0.14 - against an unconditional size
    of 0.035-0.065.  A closed gate is not catastrophic (the model ships
    unchanged) but it forgoes a correction that would have taken ECE from 0.052
    to 0.0007.  :class:`MinimaxPriorCorrection` avoids the binary decision
    altogether and is what we recommend; the gate is kept for settings where a
    hard, auditable yes/no is required by protocol.
    """

    name = "prior_correction_gated"
    n_labels_required = 0

    def __init__(self, pi_s: float = 0.5, level: float = 0.05, n_bins: int = 15,
                 n_mc: int = 300, max_prior_ratio: float = 20.0):
        self.pi_s = pi_s
        self.level = level
        self.n_bins = n_bins
        self.n_mc = n_mc
        self.max_prior_ratio = max_prior_ratio
        self.pi_t: float | None = None
        self.gate_: dict = {"applied": False, "reason": "not fitted"}

    def fit_unlabeled(
        self, target_scores, source_scores, source_labels, n_bins: int | None = None
    ) -> "GatedPriorCorrection":
        from .label_shift import (
            bbse, source_operator, target_histogram, test_label_shift_sufficiency,
        )

        nb = n_bins or self.n_bins
        edges = bin_edges(np.asarray(source_scores, float), nb, "equal_mass")
        A, pi_s_vec = source_operator(source_scores, source_labels, edges)
        q_t = target_histogram(target_scores, edges)
        est = bbse(A, pi_s_vec, q_t)
        gof = test_label_shift_sufficiency(
            A, pi_s_vec, q_t, n_t=int(np.size(target_scores)),
            n_s=int(np.size(source_scores)), n_mc=self.n_mc,
        )
        self.pi_s = float(pi_s_vec[1])
        pi_t = float(est.prevalence)
        degenerate = pi_t <= 1e-9 or pi_t >= 1 - 1e-9
        rejected = gof["p_value"] < self.level

        ratio = pi_t / max(self.pi_s, 1e-12)
        implausible = ratio > self.max_prior_ratio or ratio < 1.0 / self.max_prior_ratio

        if degenerate:
            reason = "prior estimate pinned at a simplex boundary: the solve failed"
        elif implausible:
            reason = (
                f"implied prevalence change of {ratio:.1f}x exceeds the plausibility "
                f"bound; the goodness-of-fit test cannot see shift along the "
                f"label-shift cone, so this check carries the case"
            )
        elif rejected:
            reason = f"label shift rejected on unlabeled data (p={gof['p_value']:.3g})"
        else:
            reason = f"label shift not rejected (p={gof['p_value']:.3g}); correction applied"

        self.gate_ = {
            "applied": bool(not degenerate and not implausible and not rejected),
            "prior_ratio": float(ratio),
            "implausible": bool(implausible),
            "reason": reason,
            "p_value": gof["p_value"],
            "pi_t_bbse": pi_t,
            "pi_s": self.pi_s,
            "degenerate": bool(degenerate),
        }
        self.pi_t = pi_t if self.gate_["applied"] else self.pi_s
        return self

    def fit(self, scores, labels=None, **kw) -> "GatedPriorCorrection":
        if self.pi_t is None:
            raise RuntimeError("call fit_unlabeled() first: the gate is an unlabeled decision")
        return self

    def transform(self, scores):
        if self.pi_t is None:
            raise RuntimeError("not fitted")
        if not self.gate_["applied"]:
            return np.clip(np.asarray(scores, float), 0.0, 1.0)
        return prior_correction(scores, self.pi_s, self.pi_t)


class MinimaxPriorCorrection(Recalibrator):
    r"""Prior correction at the minimax point of the identified set.

    The gate of :class:`GatedPriorCorrection` is all-or-nothing: it protects
    against the catastrophic case but forgoes the real gains available at a site
    with mild concept shift.  This estimator interpolates, and solves for the
    interpolation rather than tuning it.

    Under partial identification (Corollary 3) the data plus a concept-shift
    budget :math:`\gamma` pin :math:`\pi_t` only to an interval
    :math:`[\pi_{\mathrm{lo}}, \pi_{\mathrm{hi}}]`.  Acting requires choosing
    one value from it.  We choose the one minimising the worst case over the
    interval of

    .. math::
        \Lambda(\pi, \pi^\star)
        \;=\; \mathbb{E}_{\mu_t}\big| T_{\pi_s\to\pi}(V) - T_{\pi_s\to\pi^\star}(V) \big|,

    which upper-bounds the excess calibration error incurred by acting on
    :math:`\pi` when the truth is :math:`\pi^\star`, since
    :math:`|\mathrm{CE}_1(T_\pi) - \mathrm{CE}_1(T_{\pi^\star})| \le \Lambda`.
    Crucially :math:`\Lambda` depends on the target only through :math:`\mu_t`,
    the distribution of the *scores*, so the whole optimisation runs on
    unlabeled data.

    Because :math:`T` is monotone in the prior, the inner maximum is attained at
    an endpoint, so the problem reduces to

    .. math::
        \hat\pi \;=\; \arg\min_{\pi} \; \max\big\{\Lambda(\pi,\pi_{\mathrm{lo}}),\,
                                                  \Lambda(\pi,\pi_{\mathrm{hi}})\big\},

    a one-dimensional problem solved here by bisection on the difference of the
    two terms, which is monotone.

    A note on what this replaces.  An earlier version took the midpoint of the
    interval *in logit space*, on the reasoning that the prior correction is a
    logit shift and the loss is monotone in the size of that shift.  Both
    premises are true and the conclusion is still wrong: the loss is not
    *symmetric* in the logit gap, because once both candidate priors are small
    every corrected score is near zero and the calibration consequence
    saturates.  With an identified set touching zero, the logit midpoint was
    dragged to a near-zero prior and produced a violent over-correction - worse
    than leaving the model alone - from an interval whose actual content was
    "unknown".  Minimising the real loss instead of a surrogate removes the
    failure mode rather than patching it.
    """

    name = "prior_correction_minimax"
    n_labels_required = 0

    def __init__(self, pi_s: float = 0.5, gamma_multiplier: float = 2.0, n_bins: int = 15,
                 n_grid: int = 4096):
        self.pi_s = pi_s
        self.gamma_multiplier = gamma_multiplier
        self.n_bins = n_bins
        self.n_grid = n_grid
        self.pi_t: float | None = None
        self.interval_: dict = {}

    @staticmethod
    def _loss(v: np.ndarray, pi_s: float, pi_a: float, pi_b: float) -> float:
        return float(np.mean(np.abs(prior_correction(v, pi_s, pi_a) - prior_correction(v, pi_s, pi_b))))

    def fit_unlabeled(
        self, target_scores, source_scores, source_labels, n_bins: int | None = None
    ) -> "MinimaxPriorCorrection":
        from .label_shift import (
            min_gamma, partial_identification, source_operator, target_histogram,
        )

        nb = n_bins or self.n_bins
        v = np.clip(np.asarray(target_scores, float).ravel(), 0.0, 1.0)
        if v.size > self.n_grid:                       # subsample: the loss is an expectation
            rs = np.random.default_rng(0)
            v = v[rs.choice(v.size, self.n_grid, replace=False)]

        edges = bin_edges(np.asarray(source_scores, float), nb, "equal_mass")
        A, pi_s_vec = source_operator(source_scores, source_labels, edges)
        q_t = target_histogram(target_scores, edges)
        self.pi_s = float(pi_s_vec[1])

        g_min = min_gamma(A, pi_s_vec, q_t)
        gamma = (g_min if np.isfinite(g_min) else 0.0) * float(self.gamma_multiplier)
        pid = partial_identification(A, pi_s_vec, q_t, gamma=gamma)

        if not pid.get("feasible", False):
            self.pi_t = self.pi_s                      # no admissible prior: do nothing
            self.interval_ = {"feasible": False, "gamma": gamma, "gamma_min": g_min,
                              "reason": "identified set empty"}
            return self

        lo, hi = float(pid["lo"]), float(pid["hi"])
        if hi - lo < 1e-9:
            self.pi_t = float(np.clip(lo, 1e-9, 1 - 1e-9))
            self.interval_ = {"feasible": True, "gamma": gamma, "gamma_min": g_min,
                              "pi_lo": lo, "pi_hi": hi, "pi_point": pid["point"],
                              "worst_case_loss": 0.0, "point_identified": True}
            return self

        # Bisect on f(pi) = Lambda(pi, hi) - Lambda(pi, lo): increasing in pi at
        # the lo end and decreasing at the hi end, so the minimax point equalises.
        a, b = lo, hi
        for _ in range(60):
            mid = 0.5 * (a + b)
            if self._loss(v, self.pi_s, mid, hi) > self._loss(v, self.pi_s, mid, lo):
                a = mid                                 # too far from hi: move up
            else:
                b = mid
        pi_hat = float(np.clip(0.5 * (a + b), 1e-12, 1 - 1e-12))

        self.pi_t = pi_hat
        self.interval_ = {
            "feasible": True, "gamma": gamma, "gamma_min": g_min,
            "pi_lo": lo, "pi_hi": hi, "pi_point": pid["point"], "pi_minimax": pi_hat,
            "worst_case_loss": max(self._loss(v, self.pi_s, pi_hat, lo),
                                   self._loss(v, self.pi_s, pi_hat, hi)),
            "loss_of_doing_nothing": max(self._loss(v, self.pi_s, self.pi_s, lo),
                                         self._loss(v, self.pi_s, self.pi_s, hi)),
            "point_identified": False,
        }
        return self

    def fit(self, scores, labels=None, **kw) -> "MinimaxPriorCorrection":
        if self.pi_t is None:
            raise RuntimeError("call fit_unlabeled() first")
        return self

    def transform(self, scores):
        if self.pi_t is None:
            raise RuntimeError("not fitted")
        return prior_correction(scores, self.pi_s, self.pi_t)


class GatedHybrid(HybridPriorFewShot):
    """Hybrid recalibration whose free half is gated by the same unlabeled test.

    When the gate closes this degrades exactly to plain temperature scaling, so
    it is never worse than the labeled baseline; when the gate opens it inherits
    the sample-complexity advantage of Theorem 2.
    """

    name = "hybrid_gated"

    def __init__(self, pi_s: float = 0.5, level: float = 0.05, bounds=(0.05, 20.0), n_mc: int = 300):
        super().__init__(pi_s=pi_s, bounds=bounds)
        self.prior = GatedPriorCorrection(pi_s=pi_s, level=level, n_mc=n_mc)

    @property
    def gate_(self) -> dict:
        return self.prior.gate_


class MinimaxHybrid(HybridPriorFewShot):
    """Hybrid whose free half uses the minimax point of the identified set."""

    name = "hybrid_minimax"

    def __init__(self, pi_s: float = 0.5, gamma_multiplier: float = 2.0, bounds=(0.05, 20.0)):
        super().__init__(pi_s=pi_s, bounds=bounds)
        self.prior = MinimaxPriorCorrection(pi_s=pi_s, gamma_multiplier=gamma_multiplier)

    @property
    def interval_(self) -> dict:
        return self.prior.interval_


class SelectByCrossValidation(Recalibrator):
    r"""Choose among candidate recalibrators - including doing nothing - by cross-validation.

    This exists because of a result that inverts the obvious reading of a
    recalibration comparison.  Averaged over sites, isotonic regression beat
    shipping unchanged at every label budget we tested.  Per site and label, it
    was *worse* than shipping unchanged in 21 of 28 pairs at 25 labels and 14 of
    28 at 100: the average was carried by the single site with the largest
    cliff, while at sites whose calibration was already acceptable the fitted
    map added more estimation noise than it removed bias.

    Recalibration is therefore not something to apply indiscriminately.  It is a
    decision, and the same labels that would fit a map can decide whether to fit
    one.  This estimator holds out folds of the calibration set, scores every
    candidate - with :class:`Identity` always among them - on data it did not
    fit, and keeps the winner.  At a site with a real cliff it selects a map; at
    a site without one it selects the identity and leaves the model alone.

    ``one_se`` applies the one-standard-error rule: among candidates within one
    standard error of the best, prefer the simpler one (candidates are ordered
    simplest-first).  Ties therefore fall to doing nothing, which is the right
    default when the evidence does not distinguish the options.
    """

    name = "cv_select"

    def __init__(
        self,
        candidates: tuple[str, ...] = ("identity", "temperature", "platt", "isotonic"),
        n_folds: int = 5,
        n_bins: int = 10,
        one_se: bool = True,
        seed: int = 0,
        pi_s: float = 0.5,
    ):
        self.candidates = tuple(candidates)
        self.n_folds = n_folds
        self.n_bins = n_bins
        self.one_se = one_se
        self.seed = seed
        self.pi_s = pi_s
        self.selected_: str | None = None
        self.model_: Recalibrator | None = None
        self.scores_: dict[str, float] = {}

    def _build(self, name: str) -> Recalibrator:
        cls = METHODS[name]
        import inspect

        try:
            takes = "pi_s" in inspect.signature(cls).parameters
        except (TypeError, ValueError):
            takes = False
        return cls(pi_s=self.pi_s) if takes else cls()

    def fit(self, scores, labels=None, **kw) -> "SelectByCrossValidation":
        from .metrics import expected_calibration_error

        s = np.asarray(scores, float).ravel()
        y = np.asarray(labels, float).ravel()
        n = s.size
        rng = np.random.default_rng(self.seed)
        folds = np.array_split(rng.permutation(n), min(self.n_folds, max(n, 1)))

        per_fold: dict[str, list[float]] = {c: [] for c in self.candidates}
        for k in range(len(folds)):
            te = folds[k]
            tr = np.concatenate([folds[i] for i in range(len(folds)) if i != k]) if len(folds) > 1 else te
            if te.size < 5 or tr.size < 5:
                continue
            for name in self.candidates:
                try:
                    obj = self._build(name)
                    if obj.n_labels_required > 0:
                        obj.fit(s[tr], y[tr])
                    per_fold[name].append(
                        expected_calibration_error(obj.transform(s[te]), y[te], self.n_bins)
                    )
                except Exception:
                    per_fold[name].append(float("inf"))

        means, ses = {}, {}
        for name, vals in per_fold.items():
            v = np.array([x for x in vals if np.isfinite(x)])
            means[name] = float(v.mean()) if v.size else float("inf")
            ses[name] = float(v.std(ddof=1) / np.sqrt(v.size)) if v.size > 1 else 0.0
        self.scores_ = means

        best = min(means, key=lambda k: means[k])
        pick = best
        if self.one_se and np.isfinite(means[best]):
            cutoff = means[best] + ses[best]
            # candidates are ordered simplest-first, so this lands on the
            # simplest option that is not distinguishably worse
            for name in self.candidates:
                if means[name] <= cutoff:
                    pick = name
                    break
        self.selected_ = pick
        obj = self._build(pick)
        if obj.n_labels_required > 0:
            obj.fit(s, y)
        self.model_ = obj
        return self

    n_labels_required = 20

    def transform(self, scores):
        if self.model_ is None:
            raise RuntimeError("not fitted")
        return self.model_.transform(scores)


METHODS: dict[str, type[Recalibrator]] = {
    "identity": Identity,
    "prior_correction": PriorCorrection,
    "prevalence_correction": PrevalenceCorrection,
    "temperature": TemperatureScaling,
    "platt": PlattScaling,
    "isotonic": Isotonic,
    "scaling_binning": ScalingBinning,
    "hybrid": HybridPriorFewShot,
    "prior_correction_gated": GatedPriorCorrection,
    "prior_correction_minimax": MinimaxPriorCorrection,
    "hybrid_gated": GatedHybrid,
    "hybrid_minimax": MinimaxHybrid,
    "cv_select": SelectByCrossValidation,
}


def fit_recalibrator(
    method: str,
    scores: np.ndarray,
    labels: np.ndarray | None = None,
    *,
    pi_s: float | None = None,
    source_scores: np.ndarray | None = None,
    source_labels: np.ndarray | None = None,
    target_unlabeled: np.ndarray | None = None,
    **kw,
) -> Recalibrator:
    """Construct and fit a recalibrator by name, wiring up unlabeled estimation.

    ``scores``/``labels`` are the **labeled calibration set**.
    ``target_unlabeled`` is the site's full score vector, which costs nothing to
    collect and is what the unlabeled half should be estimated from; it defaults
    to ``scores`` only for convenience when the two coincide.

    Keeping them separate is not a nicety.  Passing the small calibration subset
    to ``fit_unlabeled`` makes a zero-label method appear to improve as labels
    are added - the exact opposite of the claim such a method exists to support -
    and it did so in an early version of the recalibration sweep.

    Dispatch is by *capability*, not by a hardcoded list of method names: any
    estimator exposing ``fit_unlabeled`` gets the source data, and any estimator
    whose ``n_labels_required`` is zero is returned without a labeled fit.  An
    earlier version matched names, and every new estimator silently fell through
    to the labeled path and raised at first use.
    """
    if method not in METHODS:
        raise KeyError(f"unknown method {method!r}; choose from {sorted(METHODS)}")
    cls = METHODS[method]
    # inspect.signature handles classes that define no __init__ of their own,
    # where cls.__init__ is object.__init__ and carries no code object.
    import inspect

    try:
        takes_pi_s = "pi_s" in inspect.signature(cls).parameters
    except (TypeError, ValueError):
        takes_pi_s = False
    obj = cls(pi_s=pi_s if pi_s is not None else 0.5, **kw) if takes_pi_s else cls(**kw)

    if hasattr(obj, "fit_unlabeled"):
        if source_scores is None or source_labels is None:
            raise ValueError(
                f"{method!r} estimates its prior from unlabeled target scores plus a "
                "labeled source sample; pass source_scores and source_labels"
            )
        obj.fit_unlabeled(
            scores if target_unlabeled is None else target_unlabeled,
            source_scores, source_labels,
        )

    if obj.n_labels_required == 0:
        return obj                       # nothing labeled to fit
    if labels is None:
        raise ValueError(f"{method!r} needs target labels")
    return obj.fit(scores, labels)
