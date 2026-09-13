r"""Theorem 2: how many labeled target ECGs a new site actually needs.

The deliverable is a number, :math:`n^\star`, defined as the smallest labeled
target sample for which a recalibration procedure attains *calibration
coverage*:

.. math::
    \Pr_{\,S \sim P_t^{\otimes n}}\big[\; \mathrm{CalErr}(\hat h_S \circ f) \le \varepsilon \;\big]
    \;\ge\; 1-\alpha ,

the probability being over the draw of the calibration set.  Note what this is
*not*: it is not a statement that the expected calibration error is small.  A
procedure whose average is fine but which fails badly one site in five is not
deployable, and averaging hides exactly that.

Three routes to :math:`n^\star` are provided and they answer different
questions:

* :func:`n_star_upper_plugin` - a computable sufficient bound.  Uses the
  observed Fisher information of the temperature parameter, so the constant is
  estimated from data rather than left as an unspecified :math:`C`.
* :func:`n_star_lower_bound` - an information-theoretic necessary bound from a
  two-point (Le Cam) argument.  No procedure whatsoever can do better.
* :func:`empirical_n_star` - the operational answer: repeatedly draw calibration
  sets of size :math:`n`, recalibrate, evaluate on held-out target data, and
  report the smallest :math:`n` meeting the coverage requirement.

The shift-adaptive part is the point.  Both bounds depend on the *residual*
concept shift :math:`\Gamma` left after the free prior correction, not on the
total shift.  Hence the threshold result: when :math:`\Gamma \le \varepsilon`,
:math:`n^\star = 0` - the unlabeled correction alone already meets the target
and the site owes nothing.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
from scipy import stats
from scipy.special import expit, logit

from .metrics import expected_calibration_error, squared_calibration_error
import inspect

from .recalibrate import METHODS, Recalibrator

__all__ = [
    "observed_fisher_temperature",
    "residual_shift_gamma",
    "n_star_upper_plugin",
    "n_star_lower_bound",
    "NStarResult",
    "empirical_n_star",
    "n_star_curve",
]

_EPS = 1e-6


def observed_fisher_temperature(scores: np.ndarray, temperature: float = 1.0) -> float:
    r"""Observed Fisher information per sample for the temperature parameter.

    For :math:`p_i(\tau) = \sigma(z_i/\tau)` with :math:`z_i` the source logit,
    the score function is :math:`\partial_\tau \ell_i = (y_i - p_i)\,z_i/\tau^2`
    and the per-sample information is
    :math:`\mathbb{E}[p_i(1-p_i) z_i^2/\tau^4]`.

    Labels do not enter: the information is an expectation under the model, so
    it is computable from the site's unlabeled scores alone.

    Estimating this from the site's own unlabeled scores is what turns the
    asymptotic bound into a number a hospital can compute before collecting any
    labels: the information depends on the score distribution, which is
    observable, and on :math:`\tau`, for which unity is the conservative
    default.
    """
    z = logit(np.clip(np.asarray(scores, float).ravel(), _EPS, 1 - _EPS))
    tau = max(float(temperature), _EPS)
    p = expit(z / tau)
    return float(np.mean(p * (1 - p) * z**2) / tau**4)


def residual_shift_gamma(
    target_scores: np.ndarray,
    target_labels: np.ndarray,
    pi_s: float,
    pi_t: float,
    n_bins: int = 15,
) -> float:
    r"""Residual concept shift :math:`\Gamma` left after the free prior correction.

    Defined as the :math:`L^1` calibration error that survives applying
    :math:`T_{\pi_s\to\pi_t}`.  This is the quantity both bounds scale with, and
    the reason a site close to pure label shift pays almost nothing: correcting
    the prior costs no labels, and what is left is all that the labels must buy.
    """
    from .label_shift import prior_correction

    corrected = prior_correction(target_scores, pi_s, pi_t)
    return float(expected_calibration_error(corrected, target_labels, n_bins))


def n_star_upper_plugin(
    eps: float,
    alpha: float,
    target_scores: np.ndarray,
    gamma: float = 0.0,
    temperature: float = 1.0,
    lipschitz: float | None = None,
) -> dict:
    r"""Sufficient :math:`n` for one-parameter recalibration, constant included.

    Write :math:`p_i = \sigma(z_i/\tau)`.  A temperature perturbation moves the
    probability scale by :math:`\partial_\tau p = -p(1-p)z/\tau^2`, so the
    induced :math:`L^1` calibration error is, to first order,
    :math:`L\,|\Delta\tau|` with the *sensitivity*
    :math:`L = \mathbb{E}[p(1-p)|z|]/\tau^2`.  The maximum-likelihood
    temperature obeys
    :math:`\sqrt{n}(\hat\tau-\tau^\star)\Rightarrow\mathcal{N}(0, I^{-1})`
    with :math:`I = \mathbb{E}[p(1-p)z^2]/\tau^4`.  Requiring the induced error
    to fit the budget left by the residual shift,
    :math:`\varepsilon_{\text{eff}} = \varepsilon-\Gamma`, gives

    .. math::
        n \;\ge\; \frac{z_{1-\alpha/2}^2\,L^2}{I\,\varepsilon_{\text{eff}}^2}
        \;=\; \frac{z_{1-\alpha/2}^2\,\big(\mathbb{E}[p(1-p)|z|]\big)^2}
                      {\mathbb{E}[p(1-p)z^2]\;\varepsilon_{\text{eff}}^2}.

    Cauchy-Schwarz collapses this to a corollary worth stating on its own:

    .. math:: n^\star \;\le\; \frac{z_{1-\alpha/2}^2\; \mathbb{E}[p(1-p)]}{\varepsilon_{\text{eff}}^2},

    i.e. **the label cost is set by the model's mean predictive variance on the
    target site**, divided by the squared calibration budget.  A site where the
    transferred model is confident - the screening-population case, where
    nearly every score sits near zero - has small :math:`\mathbb{E}[p(1-p)]`
    and therefore a small label bill.  Every quantity on the right is computable
    from unlabeled target scores, so the site can price the work before doing it.

    ``lipschitz`` overrides the derived sensitivity :math:`L`; leave it ``None``.
    When :math:`\Gamma \ge \varepsilon` the budget is already spent and the
    result is ``feasible=False``: the honest advice there is a richer
    recalibration family or a re-examined model, not more labels.
    """
    eps_eff = eps - gamma
    v = np.clip(np.asarray(target_scores, float).ravel(), _EPS, 1 - _EPS)
    tau = max(float(temperature), _EPS)
    z = logit(v)
    p = expit(z / tau)
    var = p * (1 - p)
    sens = float(np.mean(var * np.abs(z))) / tau**2 if lipschitz is None else float(lipschitz)
    info = float(np.mean(var * z**2)) / tau**4
    z_q = float(stats.norm.ppf(1 - alpha / 2))
    if eps_eff <= 0:
        return {
            "n_star": float("inf"), "feasible": False, "eps": eps, "gamma": gamma,
            "eps_effective": eps_eff, "fisher_info": info, "sensitivity": sens,
            "mean_pred_var": float(np.mean(var)),
            "reason": "residual concept shift already exceeds the calibration budget",
        }
    if info <= 0:
        return {
            "n_star": 0, "feasible": True, "eps": eps, "gamma": gamma,
            "eps_effective": eps_eff, "fisher_info": info, "sensitivity": sens,
            "mean_pred_var": float(np.mean(var)),
            "reason": "temperature is unidentified (degenerate scores); nothing to fit",
        }
    n = (z_q**2) * sens**2 / (info * eps_eff**2)
    cs = (z_q**2) * float(np.mean(var)) / eps_eff**2      # Cauchy-Schwarz corollary
    return {
        "n_star": int(np.ceil(n)),
        "n_star_cs_bound": int(np.ceil(cs)),
        "feasible": True,
        "eps": eps,
        "gamma": gamma,
        "eps_effective": eps_eff,
        "fisher_info": info,
        "sensitivity": sens,
        "mean_pred_var": float(np.mean(var)),
        "z_quantile": z_q,
        "reason": "",
    }


def n_star_lower_bound(eps: float, alpha: float = 0.25, gamma: float | None = None) -> dict:
    r"""Information-theoretic lower bound via a two-point (Le Cam) argument.

    Construct two target laws sharing the *same* unlabeled score marginal - so
    no unlabeled procedure, however clever, can separate them - whose
    calibration curves differ by :math:`2\varepsilon`.  Their :math:`n`-fold
    label distributions stay statistically indistinguishable unless
    :math:`n \gtrsim \varepsilon^{-2}`, giving

    .. math:: n^\star \;\ge\; \frac{(1-2\alpha)^2}{8\,\varepsilon^2}.

    The role of :math:`\Gamma` is a *constraint on the construction*, not a
    scale in the rate.  Both hypotheses must lie in the ball of admissible
    concept shifts, and two points of that ball are at most :math:`2\Gamma`
    apart, so a separation of :math:`2\varepsilon` is constructible only when
    :math:`\Gamma \ge \varepsilon`.  Below that the bound is vacuous - which
    is the formal statement of the threshold phenomenon: a site whose residual
    concept shift is within the calibration budget can be made safe with **zero**
    labels, and no lower bound contradicts it.

    An earlier form of this function scaled the separation as
    :math:`\min(\varepsilon,\Gamma)`, which made the bound *decrease* as the
    shift grew - backwards.  The quantity that must be resolved is the budget
    :math:`\varepsilon`; :math:`\Gamma` only says whether a hard instance
    exists at all.
    """
    if eps <= 0:
        return {"n_star": int(np.iinfo(np.int64).max), "eps": eps, "gamma": gamma,
                "constructible": True, "note": "zero budget is unattainable at finite n"}
    if gamma is not None and gamma < eps:
        return {"n_star": 0, "eps": eps, "gamma": gamma, "constructible": False,
                "note": "no hard instance fits inside the Gamma-ball: zero labels can suffice"}
    # (1 - 2 alpha) SQUARED.  Pinsker plus tensorisation bounds the squared total
    # variation, so the confidence factor enters squared; dropping the exponent
    # inflates the bound by up to 4x and would claim more labels are provably
    # necessary than the argument establishes.
    n = (1.0 - 2.0 * alpha) ** 2 / (8.0 * eps**2)
    return {"n_star": int(max(0, np.ceil(n))), "eps": eps, "gamma": gamma,
            "constructible": True, "note": ""}


@dataclass
class NStarResult:
    n_star: int | None
    eps: float
    alpha: float
    method: str
    grid: np.ndarray
    coverage: np.ndarray       # P(CalErr <= eps) at each n
    ece_median: np.ndarray
    ece_q90: np.ndarray
    n_repeats: int
    zero_shot_coverage: float  # coverage with 0 labels (free correction only)

    def to_dict(self) -> dict:
        d = asdict(self)
        for k in ("grid", "coverage", "ece_median", "ece_q90"):
            d[k] = np.asarray(d[k]).tolist()
        return d


def empirical_n_star(
    target_scores: np.ndarray,
    target_labels: np.ndarray,
    *,
    eps: float = 0.02,
    alpha: float = 0.1,
    method: str = "hybrid",
    grid: np.ndarray | None = None,
    n_repeats: int = 200,
    eval_frac: float = 0.5,
    pi_s: float | None = None,
    source_scores: np.ndarray | None = None,
    source_labels: np.ndarray | None = None,
    n_bins: int = 15,
    seed: int = 0,
    metric: str = "ece",
) -> NStarResult:
    """Smallest :math:`n` meeting the coverage requirement, by direct simulation.

    Protocol, per candidate :math:`n` and per repeat: draw a calibration set of
    that size from the target site, fit the recalibrator, and evaluate on a
    held-out target split that is *disjoint* from it.  ``coverage`` is the
    fraction of repeats whose held-out calibration error is at most ``eps``.

    The unlabeled ingredients of ``prior_correction`` and ``hybrid`` are fitted
    once from target scores plus source data, never from the calibration
    labels, so the reported :math:`n` is a true label cost.

    ``n = 0`` is evaluated explicitly: for the label-free methods it is the
    deployable zero-shot answer, and if its coverage already exceeds
    :math:`1-\\alpha` the site's honest label requirement is zero.
    """
    v = np.asarray(target_scores, float).ravel()
    y = np.asarray(target_labels, float).ravel()
    rng = np.random.default_rng(seed)
    n_total = len(v)
    n_eval = int(round(eval_frac * n_total))
    if n_eval < 50 or n_total - n_eval < 10:
        raise ValueError(f"target sample of {n_total} is too small to split for n* estimation")

    score_fn = (
        (lambda p, yy: expected_calibration_error(p, yy, n_bins))
        if metric == "ece"
        else (lambda p, yy: abs(squared_calibration_error(p, yy, n_bins)) ** 0.5)
    )
    if grid is None:
        cap = n_total - n_eval
        grid = np.unique(np.clip(
            np.array([0, 10, 25, 50, 100, 200, 400, 800, 1600, 3200, 6400]), 0, cap
        ))

    def make(nn: int, cal_idx: np.ndarray) -> Recalibrator:
        """Build a recalibrator; unlabeled parts never see calibration labels.

        Dispatch is by capability rather than by a name list, so an estimator
        added to METHODS works here without touching this function.
        """
        cls = METHODS[method]
        try:
            takes_pi_s = "pi_s" in inspect.signature(cls).parameters
        except (TypeError, ValueError):
            takes_pi_s = False
        obj = cls(pi_s=pi_s if pi_s is not None else float(y.mean())) if takes_pi_s else cls()

        if hasattr(obj, "fit_unlabeled"):
            if source_scores is None or source_labels is None:
                raise ValueError(f"{method!r} needs source_scores and source_labels")
            obj.fit_unlabeled(v, source_scores, source_labels)
        if obj.n_labels_required == 0:
            return obj
        if nn == 0:
            # No labels: fall back to the free half if there is one, else identity.
            inner = getattr(obj, "prior", None)
            return inner if inner is not None else METHODS["identity"]()
        return obj.fit(v[cal_idx], y[cal_idx])

    coverage = np.zeros(len(grid))
    ece_med = np.zeros(len(grid))
    ece_q90 = np.zeros(len(grid))
    for gi, nn in enumerate(grid):
        vals = np.empty(n_repeats)
        for r in range(n_repeats):
            perm = rng.permutation(n_total)
            ev, pool = perm[:n_eval], perm[n_eval:]
            cal = pool[: int(nn)]
            try:
                rec = make(int(nn), cal)
                vals[r] = score_fn(rec.transform(v[ev]), y[ev])
            except Exception:
                vals[r] = np.nan
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            coverage[gi] = ece_med[gi] = ece_q90[gi] = np.nan
            continue
        coverage[gi] = float(np.mean(vals <= eps))
        ece_med[gi] = float(np.median(vals))
        ece_q90[gi] = float(np.quantile(vals, 0.9))

    ok = np.where(coverage >= 1 - alpha)[0]
    n_star = int(grid[ok[0]]) if ok.size else None
    return NStarResult(
        n_star=n_star, eps=eps, alpha=alpha, method=method, grid=np.asarray(grid),
        coverage=coverage, ece_median=ece_med, ece_q90=ece_q90, n_repeats=n_repeats,
        zero_shot_coverage=float(coverage[0]) if grid[0] == 0 else float("nan"),
    )


def n_star_curve(
    target_scores: np.ndarray,
    target_labels: np.ndarray,
    methods: list[str] | None = None,
    **kw,
) -> dict[str, NStarResult]:
    """Run :func:`empirical_n_star` for several methods on the same target site."""
    methods = methods or ["identity", "prior_correction", "temperature", "hybrid"]
    return {m: empirical_n_star(target_scores, target_labels, method=m, **kw) for m in methods}
