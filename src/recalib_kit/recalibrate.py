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
:class:`PriorCorrection`      0 (unlabeled only)    :math:`D_{\text{label}}`
:class:`TemperatureScaling`   n (1 parameter)       smooth global distortion
:class:`PlattScaling`         n (2 parameters)      affine logit distortion
:class:`Isotonic`             n (nonparametric)     any monotone distortion
:class:`ScalingBinning`       n                     as isotonic, with a
                                                    verifiable ECE guarantee
:class:`HybridPriorFewShot`   n (1 parameter)       both, warm-started by the
                                                    free correction
============================  ====================  ============================

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
    "TemperatureScaling",
    "PlattScaling",
    "Isotonic",
    "ScalingBinning",
    "HybridPriorFewShot",
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

    def transform(self, scores):
        if self.temperature_ is None:
            raise RuntimeError("not fitted")
        return expit(_safe_logit(scores) / self.temperature_)

    @property
    def n_labels_required(self) -> int:
        return 1


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

    def transform(self, scores):
        return self.temp.transform(self.prior.transform(scores))

    @property
    def temperature_(self) -> float | None:
        return self.temp.temperature_

    @property
    def n_labels_required(self) -> int:
        return 1


METHODS: dict[str, type[Recalibrator]] = {
    "identity": Identity,
    "prior_correction": PriorCorrection,
    "temperature": TemperatureScaling,
    "platt": PlattScaling,
    "isotonic": Isotonic,
    "scaling_binning": ScalingBinning,
    "hybrid": HybridPriorFewShot,
}


def fit_recalibrator(
    method: str,
    scores: np.ndarray,
    labels: np.ndarray | None = None,
    *,
    pi_s: float | None = None,
    source_scores: np.ndarray | None = None,
    source_labels: np.ndarray | None = None,
    **kw,
) -> Recalibrator:
    """Construct and fit a recalibrator by name, wiring up unlabeled estimation."""
    if method not in METHODS:
        raise KeyError(f"unknown method {method!r}; choose from {sorted(METHODS)}")
    cls = METHODS[method]
    if method in ("prior_correction", "hybrid"):
        obj = cls(pi_s=pi_s if pi_s is not None else 0.5, **kw)
        if source_scores is not None and source_labels is not None:
            obj.fit_unlabeled(scores, source_scores, source_labels)
        if method == "hybrid":
            return obj.fit(scores, labels)
        # PriorCorrection is already usable once pi_t is known; it only needs a
        # fit() call to fall back on labels when no unlabeled estimate was made.
        return obj if obj.pi_t is not None else obj.fit(scores, labels)
    obj = cls(**kw)
    return obj.fit(scores, labels)
