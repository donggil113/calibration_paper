r"""Theorem 1: decomposing target calibration error into label- and concept-shift parts.

Let :math:`f` be a frozen source-trained model, :math:`\mu` the distribution of
its score :math:`v = f(X)` on the target site, and

.. math::
    g_t(v) = \mathbb{E}_{P_t}[\,Y \mid f(X) = v\,]

the target-site calibration curve.  Let :math:`T = T_{\pi_s\to\pi_t}` be the
prior-correction map (:func:`recalib_kit.label_shift.prior_correction`).  Write

.. math::
    \underbrace{g_t(v) - v}_{\text{total miscalibration}}
    = \underbrace{\big(T(v) - v\big)}_{\delta_{\text{label}}(v)}
    + \underbrace{\big(g_t(v) - T(v)\big)}_{\delta_{\text{concept}}(v)} .

Both pieces are exactly what their names claim:

* :math:`\delta_{\text{label}}` is a deterministic function of
  :math:`(\pi_s, \pi_t, v)` only.  Under **pure** label shift it equals the
  whole miscalibration, and :math:`\pi_t` is recoverable from unlabeled target
  data - so this component is correctable *for free*.
* :math:`\delta_{\text{concept}}` is whatever survives the best possible prior
  correction.  It is zero if and only if
  :math:`P_t(y\mid x) = T(P_s(y\mid x))` :math:`\mu`-a.e., i.e. iff the label
  shift model holds on the score sigma-algebra.  Estimating it requires target
  labels; no unlabeled procedure can.

In :math:`L^2(\mu)` the split is an exact Pythagorean identity,

.. math::
    \underbrace{\|g_t - \mathrm{id}\|^2_{\mu}}_{\mathrm{CalErr}_2^2}
    = \underbrace{\|\delta_{\text{label}}\|^2_{\mu}}_{D_{\text{label}}}
    + \underbrace{\|\delta_{\text{concept}}\|^2_{\mu}}_{D_{\text{concept}}}
    + \underbrace{2\langle \delta_{\text{label}}, \delta_{\text{concept}}\rangle_{\mu}}_{R},

which is why the paper reports squared calibration error as the primary
endpoint: the familiar :math:`L^1` ECE only obeys a triangle *inequality*
(:func:`decompose_l1` returns that bound together with the sign-alignment
condition under which it is tight).

The interaction term :math:`R` is not a nuisance.  :math:`R < 0` means the two
shifts partially cancel - a site can look deceptively well calibrated while
both components are individually large, and a naive prior correction there
makes calibration *worse*.  Detecting that case is a practical deliverable of
the decomposition, so :class:`Decomposition` carries it explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import ClassVar

import numpy as np

from .label_shift import (
    LabelShiftEstimate,
    bbse,
    prior_correction,
    source_operator,
    target_histogram,
)
from .metrics import bin_assign, bin_edges, binned_stats, expected_calibration_error

__all__ = ["Decomposition", "decompose", "decompose_l1", "decompose_unlabeled_budget"]


@dataclass
class Decomposition:
    """Result of the Theorem 1 decomposition on one site and one label."""

    total: float               # CalErr_2^2 on the target, debiased
    d_label: float             # ||delta_label||^2 : correctable without labels
    d_concept: float           # ||delta_concept||^2 : needs labels, debiased
    interaction: float         # R = 2<delta_label, delta_concept>
    residual: float            # total - (d_label + d_concept + interaction); ~0 by construction
    pi_s: float
    pi_t_used: float
    pi_t_source: str           # 'oracle' | 'bbse' | 'labeled'
    n_target: int
    ece_before: float          # L1 ECE of raw scores on target
    ece_after_prior_fix: float # L1 ECE after applying T (unlabeled correction only)
    n_bins_effective: int

    @property
    def label_shift_share(self) -> float:
        """Share of the squared calibration error attributable to label shift.

        This is the part that a **single number** - the target prevalence -
        would repair.  It is deliberately *not* called a "free" share: whether
        that number can be obtained without target labels is a separate
        question, and under cross-national concept shift the answer is no (see
        :func:`decompose_unlabeled_budget` and
        :class:`recalib_kit.recalibrate.PrevalenceCorrection`).

        Clipped to [0, 1] for display only; the raw components are kept intact
        so a negative interaction remains visible.
        """
        if self.total <= 0:
            return float("nan")
        return float(np.clip(self.d_label / self.total, 0.0, 1.0))

    @property
    def free_fraction(self) -> float:
        """Deprecated alias for :attr:`label_shift_share`.

        Kept so stored analyses remain readable.  The old name asserted that the
        share was recoverable without labels, which this study found to be false.
        """
        return self.label_shift_share

    #: Below this L1 ECE there is nothing to gain, so the gain ratio is undefined.
    #: Chosen well under any clinically meaningful miscalibration: a model whose
    #: ECE is 0.002 is calibrated for every practical purpose.  ClassVar, not a
    #: field: a dataclass turns a bare annotation into a per-result datum, and
    #: this threshold then appeared as a constructor argument and as a column in
    #: every exported table.
    GAIN_FLOOR: ClassVar[float] = 0.005

    @property
    def realized_free_gain(self) -> float:
        """Fraction of L1 ECE actually removed by the prior correction as applied.

        The operational counterpart of :attr:`label_shift_share`: what a hospital
        would measure after switching the correction on, rather than what the
        decomposition says is available in principle.  The two agree when the
        interaction term is small and diverge when it is not, which makes the
        gap a useful audit.

        Returns NaN when the site was already calibrated
        (``ece_before <= GAIN_FLOOR``).  This is not cosmetic: it is a ratio
        with the baseline in the denominator, so a site whose ECE moves from
        0.0001 to 0.0006 - both zero for any practical purpose - reports a
        "gain" of -5.  Averaging such values across labels produced an apparent
        catastrophic failure of the prior correction at a site where it had in
        fact done nothing at all, in either direction.
        """
        if not np.isfinite(self.ece_before) or self.ece_before <= self.GAIN_FLOOR:
            return float("nan")
        return float((self.ece_before - self.ece_after_prior_fix) / self.ece_before)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["label_shift_share"] = self.label_shift_share
        d["free_fraction"] = self.label_shift_share      # deprecated alias
        d["realized_free_gain"] = self.realized_free_gain
        return d


def decompose(
    target_scores: np.ndarray,
    target_labels: np.ndarray,
    pi_s: float,
    pi_t: float | None = None,
    n_bins: int = 15,
    scheme: str = "equal_mass",
    source_scores: np.ndarray | None = None,
    source_labels: np.ndarray | None = None,
    debias: bool = True,
) -> Decomposition:
    r"""Estimate the Theorem 1 decomposition on a labeled target sample.

    ``pi_t`` controls which regime is being measured:

    * a float - oracle prior, for simulation studies where the truth is known;
    * ``None`` with ``source_scores``/``source_labels`` given - :func:`bbse`
      estimates the prior from the target scores *without using their labels*,
      which is the deployable setting;
    * ``None`` with no source data - the prior is read off the target labels.
      This is an upper bound on what any unlabeled method can achieve and is
      labeled ``'labeled'`` so it is never mistaken for a deployable result.

    Target labels are used only to estimate :math:`g_t`; the label-shift
    component never touches them.
    """
    v = np.clip(np.asarray(target_scores, float).ravel(), 0.0, 1.0)
    y = np.asarray(target_labels, float).ravel()
    if v.shape != y.shape:
        raise ValueError(f"scores {v.shape} and labels {y.shape} must match")

    # --- choose the target prior ------------------------------------------
    if pi_t is not None:
        pi_t_used, src = float(pi_t), "oracle"
    elif source_scores is not None and source_labels is not None:
        edges_s = bin_edges(np.asarray(source_scores, float), n_bins, scheme)
        A, pi_s_vec = source_operator(source_scores, source_labels, edges_s)
        est: LabelShiftEstimate = bbse(A, pi_s_vec, target_histogram(v, edges_s))
        pi_t_used, src = est.prevalence, "bbse"
    else:
        pi_t_used, src = float(y.mean()), "labeled"

    # --- the two component functions, averaged WITHIN each target bin -----
    # T is nonlinear, so evaluating it at the bin-mean score (T(v_bar)) instead
    # of averaging it over the samples in the bin (mean_b T(v_i)) injects a
    # Jensen bias of order T''(v) Var_b(v).  That bias is systematic, it is
    # correlated with delta_label, and it therefore leaks into the interaction
    # term - large enough, at practical bin counts, to fake a cross-component
    # interaction under *pure* label shift.  Averaging the noise-free function
    # per sample removes it and makes the identity exact by construction.
    t_i = prior_correction(v, pi_s, pi_t_used)
    st = binned_stats(v, y, n_bins, scheme)
    idx = bin_assign(v, st.edges)
    nb = len(st.edges) - 1
    t_sum = np.bincount(idx, weights=t_i, minlength=nb)
    with np.errstate(invalid="ignore", divide="ignore"):
        t_mean = np.where(st.count > 0, t_sum / np.maximum(st.count, 1e-12), 0.0)

    mask = st.count > 0
    w = st.weight[mask]
    w = w / w.sum() if w.sum() > 0 else w
    v_bar = st.mean_score[mask]                 # mean_b v_i
    g_hat = st.mean_label[mask]                 # \hat g_t
    n_b = st.count[mask]
    t_bar = t_mean[mask]                        # mean_b T(v_i)

    d_lab_fn = t_bar - v_bar                    # noise-free given pi_t
    d_con_fn = g_hat - t_bar                    # carries the noise in \hat g_t

    # Debiasing: \hat g_t has per-bin binomial variance, which inflates any
    # squared quantity containing it.  d_label is free of it; the interaction
    # term is linear in \hat g_t and therefore already unbiased.  Only
    # d_concept and the total need correction.
    if debias:
        var = np.where(n_b > 1, g_hat * (1 - g_hat) / np.maximum(n_b - 1.0, 1e-12), 0.0)
    else:
        var = np.zeros_like(g_hat)

    d_label = float(np.sum(w * d_lab_fn**2))
    d_concept = float(np.sum(w * (d_con_fn**2 - var)))
    interaction = float(2.0 * np.sum(w * d_lab_fn * d_con_fn))
    total = float(np.sum(w * ((g_hat - v_bar) ** 2 - var)))

    return Decomposition(
        total=total,
        d_label=d_label,
        d_concept=d_concept,
        interaction=interaction,
        residual=total - (d_label + d_concept + interaction),
        pi_s=float(pi_s),
        pi_t_used=float(pi_t_used),
        pi_t_source=src,
        n_target=int(v.size),
        ece_before=expected_calibration_error(v, y, n_bins, scheme),
        ece_after_prior_fix=expected_calibration_error(
            prior_correction(v, pi_s, pi_t_used), y, n_bins, scheme
        ),
        n_bins_effective=int(mask.sum()),
    )


def decompose_l1(
    target_scores: np.ndarray,
    target_labels: np.ndarray,
    pi_s: float,
    pi_t: float,
    n_bins: int = 15,
    scheme: str = "equal_mass",
) -> dict:
    r"""The :math:`L^1` companion: a triangle bound plus its tightness condition.

    Returns ``ece_total``, the two component :math:`L^1` norms, the bound
    ``d_label_l1 + d_concept_l1``, and ``sign_agreement`` - the :math:`\mu`-mass
    on which the two components share a sign.  The triangle inequality is tight
    exactly when that mass is 1; values well below 1 flag the cancellation
    regime in which the site's apparent calibration is an accident of two
    errors pointing in opposite directions.
    """
    v = np.clip(np.asarray(target_scores, float).ravel(), 0.0, 1.0)
    y = np.asarray(target_labels, float).ravel()
    st = binned_stats(v, y, n_bins, scheme)
    t_i = prior_correction(v, pi_s, pi_t)
    idx = bin_assign(v, st.edges)
    nb = len(st.edges) - 1
    with np.errstate(invalid="ignore", divide="ignore"):
        t_mean = np.where(
            st.count > 0,
            np.bincount(idx, weights=t_i, minlength=nb) / np.maximum(st.count, 1e-12),
            0.0,
        )
    mask = st.count > 0
    w = st.weight[mask]
    w = w / w.sum() if w.sum() > 0 else w
    v_bar, g_hat, t_bar = st.mean_score[mask], st.mean_label[mask], t_mean[mask]
    d_lab, d_con = t_bar - v_bar, g_hat - t_bar
    same_sign = np.sign(d_lab) == np.sign(d_con)
    return {
        "ece_total": float(np.sum(w * np.abs(g_hat - v_bar))),
        "d_label_l1": float(np.sum(w * np.abs(d_lab))),
        "d_concept_l1": float(np.sum(w * np.abs(d_con))),
        "triangle_bound": float(np.sum(w * (np.abs(d_lab) + np.abs(d_con)))),
        "sign_agreement": float(np.sum(w[same_sign])),
    }


def decompose_unlabeled_budget(
    target_scores: np.ndarray,
    pi_s: float,
    source_scores: np.ndarray,
    source_labels: np.ndarray,
    n_bins: int = 15,
    scheme: str = "equal_mass",
    max_prior_ratio: float = 20.0,
) -> dict:
    r"""The part of the decomposition computable with **zero** target labels.

    Returns the BBSE prior estimate, :math:`D_{\text{label}}`, and the
    unlabeled goodness-of-fit evidence for concept shift.  A deploying site can
    run this before committing to any chart review, and it answers the only
    question that matters at that stage: *is a free correction going to be
    enough here, or do I have to buy labels?*
    """
    from .label_shift import min_gamma, test_label_shift_sufficiency

    v = np.clip(np.asarray(target_scores, float).ravel(), 0.0, 1.0)
    edges = bin_edges(np.asarray(source_scores, float), n_bins, scheme)
    A, pi_s_vec = source_operator(source_scores, source_labels, edges)
    q_t = target_histogram(v, edges)
    est = bbse(A, pi_s_vec, q_t)
    pi_t_hat = est.prevalence

    st = binned_stats(v, np.zeros_like(v), n_bins, scheme)
    t_i = prior_correction(v, pi_s, pi_t_hat)
    idx = bin_assign(v, st.edges)
    nb = len(st.edges) - 1
    with np.errstate(invalid="ignore", divide="ignore"):
        t_mean = np.where(
            st.count > 0,
            np.bincount(idx, weights=t_i, minlength=nb) / np.maximum(st.count, 1e-12),
            0.0,
        )
    mask = st.count > 0
    w = st.weight[mask]
    w = w / w.sum() if w.sum() > 0 else w
    d_lab = t_mean[mask] - st.mean_score[mask]

    gof = test_label_shift_sufficiency(A, pi_s_vec, q_t, n_t=int(v.size), n_s=int(len(source_scores)))
    # A prior estimate pinned at a simplex boundary is a failed solve, not a
    # measurement: nnls has hit w=0 and the "estimate" carries no information.
    degenerate = bool(pi_t_hat <= 1e-9 or pi_t_hat >= 1 - 1e-9)

    # Passing the goodness-of-fit test is necessary but NOT sufficient.  The test
    # can only see concept shift that moves the target histogram *off* the
    # label-shift cone; shift that moves it *along* the cone is invisible to it,
    # and by Theorem 1 is genuinely unidentified - no unlabeled procedure can do
    # better.  Empirically this is not a corner case: at one site the estimator
    # returned a prevalence of 0.85 against a truth of 0.02 with a p-value of
    # 0.23.  A crude plausibility bound catches that class of failure where the
    # test cannot, and it is stated as what it is - a sanity check on the
    # magnitude of the implied prior change, not an inference.
    ratio = pi_t_hat / max(pi_s, 1e-12)
    implausible = bool(ratio > max_prior_ratio or ratio < 1.0 / max_prior_ratio)

    return {
        "pi_s": float(pi_s),
        "pi_t_bbse": float(pi_t_hat),
        "prior_ratio_implausible": implausible,
        "bbse_degenerate": degenerate,
        "bbse_sigma_min": float(est.diagnostics.get("sigma_min", float("nan"))),
        "bbse_trustworthy": bool(
            not degenerate and not implausible and not gof["reject_label_shift"]
        ),
        "prior_ratio": float(pi_t_hat / max(pi_s, 1e-12)),
        "d_label": float(np.sum(w * d_lab**2)),
        "d_label_l1": float(np.sum(w * np.abs(d_lab))),
        "gamma_min": min_gamma(A, pi_s_vec, q_t),
        "gof_p_value": gof["p_value"],
        "concept_shift_detected": gof["reject_label_shift"],
        "n_target_unlabeled": int(v.size),
    }
