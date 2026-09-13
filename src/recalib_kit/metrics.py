"""Calibration and discrimination metrics.

The estimators here are the measurement instrument for the whole study, so two
properties matter more than convenience:

1. **Bias is handled explicitly.**  The plugin binned estimator of squared
   calibration error is upward biased by roughly ``B/n``; at the bin counts a
   small target site affords, that bias is the same order as the effect we are
   trying to measure.  :func:`squared_calibration_error` therefore offers the
   debiased estimator of Kumar, Liang & Ma (2019) alongside the plugin one, and
   the debiased value is what the paper reports.

2. **Binning scheme is a declared parameter, never a default that drifts.**
   Equal-width bins are unstable when the score distribution piles up near
   zero - exactly what happens when a hospital-trained model meets a screening
   population.  Equal-mass (quantile) binning is the default here for that
   reason.

All functions take 1-D score/label arrays for a single label (one-vs-rest);
multilabel aggregation lives in :func:`multilabel_report`.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Literal

import numpy as np

BinScheme = Literal["equal_mass", "equal_width"]

__all__ = [
    "bin_edges",
    "bin_assign",
    "BinnedStats",
    "binned_stats",
    "expected_calibration_error",
    "squared_calibration_error",
    "maximum_calibration_error",
    "brier_decomposition",
    "negative_log_likelihood",
    "auroc",
    "auprc",
    "reliability_curve",
    "bootstrap_ci",
    "CalibrationReport",
    "calibration_report",
    "multilabel_report",
]


# --------------------------------------------------------------------------
# binning
# --------------------------------------------------------------------------
def bin_edges(scores: np.ndarray, n_bins: int, scheme: BinScheme = "equal_mass") -> np.ndarray:
    """Return ``n_bins + 1`` monotone edges spanning [0, 1].

    Equal-mass edges are quantiles of the observed scores, deduplicated.  When
    scores are highly degenerate (many ties at 0) the number of usable bins can
    fall below ``n_bins``; that is reported rather than silently patched, since
    it is itself a symptom of the shift.
    """
    scores = np.asarray(scores, dtype=float)
    if scheme == "equal_width":
        return np.linspace(0.0, 1.0, n_bins + 1)
    if scheme != "equal_mass":
        raise ValueError(f"unknown bin scheme {scheme!r}")
    if scores.size == 0:
        return np.linspace(0.0, 1.0, n_bins + 1)
    qs = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(scores, qs)
    edges[0], edges[-1] = 0.0, 1.0
    edges = np.unique(edges)
    if edges.size < 2:
        edges = np.array([0.0, 1.0])
    return edges


def bin_assign(scores: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Map scores to bin indices in ``[0, len(edges) - 2]``."""
    scores = np.asarray(scores, dtype=float)
    idx = np.searchsorted(edges, scores, side="right") - 1
    return np.clip(idx, 0, len(edges) - 2)


@dataclass
class BinnedStats:
    """Per-bin sufficient statistics.

    These are the *only* quantities that need to cross an institutional
    boundary for every estimator in this package, which is what makes the
    federated protocol for the restricted Korean cohort possible.
    """

    edges: np.ndarray
    count: np.ndarray          # n_b
    mean_score: np.ndarray     # mean f(x) in bin
    mean_label: np.ndarray     # empirical P(y=1 | bin)
    n: int

    @property
    def weight(self) -> np.ndarray:
        return self.count / max(self.n, 1)

    @property
    def n_nonempty(self) -> int:
        return int((self.count > 0).sum())

    def to_dict(self) -> dict:
        d = asdict(self)
        return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in d.items()}


def binned_stats(
    scores: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
    scheme: BinScheme = "equal_mass",
    sample_weight: np.ndarray | None = None,
) -> BinnedStats:
    """Compute per-bin counts, mean score and empirical event rate.

    ``sample_weight`` supports importance-weighted calibration assessment,
    which is how a source-domain estimate is reweighted to a target prior.
    """
    scores = np.asarray(scores, dtype=float).ravel()
    labels = np.asarray(labels, dtype=float).ravel()
    if scores.shape != labels.shape:
        raise ValueError(f"scores {scores.shape} and labels {labels.shape} must match")
    w = np.ones_like(scores) if sample_weight is None else np.asarray(sample_weight, float).ravel()
    if w.shape != scores.shape:
        raise ValueError("sample_weight must match scores")

    edges = bin_edges(scores, n_bins, scheme)
    nb = len(edges) - 1
    idx = bin_assign(scores, edges)

    count = np.bincount(idx, weights=w, minlength=nb)
    s_sum = np.bincount(idx, weights=w * scores, minlength=nb)
    y_sum = np.bincount(idx, weights=w * labels, minlength=nb)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_score = np.where(count > 0, s_sum / np.maximum(count, 1e-12), 0.0)
        mean_label = np.where(count > 0, y_sum / np.maximum(count, 1e-12), 0.0)
    return BinnedStats(edges, count, mean_score, mean_label, n=int(scores.size))


# --------------------------------------------------------------------------
# calibration error
# --------------------------------------------------------------------------
def expected_calibration_error(
    scores: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
    scheme: BinScheme = "equal_mass",
    sample_weight: np.ndarray | None = None,
) -> float:
    r"""L1 expected calibration error :math:`E|\,E[Y\mid f(X)] - f(X)\,|`."""
    st = binned_stats(scores, labels, n_bins, scheme, sample_weight)
    gap = np.abs(st.mean_label - st.mean_score)
    return float(np.sum(st.weight * gap))


def squared_calibration_error(
    scores: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
    scheme: BinScheme = "equal_mass",
    debiased: bool = True,
    sample_weight: np.ndarray | None = None,
) -> float:
    r"""Squared (L2) calibration error :math:`E\,(E[Y\mid f(X)] - f(X))^2`.

    This is the *calibration* term of the Brier decomposition and the quantity
    Theorem 1 decomposes exactly, because an :math:`L^2` norm obeys a
    Pythagorean identity while an :math:`L^1` norm only obeys a triangle
    inequality.

    With ``debiased=True`` the per-bin binomial variance ``p(1-p)/(n_b - 1)``
    is subtracted, giving an approximately unbiased estimate that can be
    slightly negative when the true calibration error is near zero.  It is
    *not* clipped at zero: clipping reintroduces positive bias exactly in the
    regime where we claim a model is well calibrated.
    """
    st = binned_stats(scores, labels, n_bins, scheme, sample_weight)
    gap2 = (st.mean_label - st.mean_score) ** 2
    est = float(np.sum(st.weight * gap2))
    if not debiased:
        return est
    with np.errstate(invalid="ignore", divide="ignore"):
        var = np.where(
            st.count > 1,
            st.mean_label * (1.0 - st.mean_label) / np.maximum(st.count - 1.0, 1e-12),
            0.0,
        )
    return float(est - np.sum(st.weight * var))


def maximum_calibration_error(
    scores: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
    scheme: BinScheme = "equal_mass",
    min_count: int = 10,
) -> float:
    """Worst-case gap over bins holding at least ``min_count`` samples."""
    st = binned_stats(scores, labels, n_bins, scheme)
    mask = st.count >= min_count
    if not mask.any():
        return float("nan")
    return float(np.max(np.abs(st.mean_label - st.mean_score)[mask]))


def brier_decomposition(
    scores: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
    scheme: BinScheme = "equal_mass",
) -> dict[str, float]:
    """Murphy decomposition: ``brier = reliability - resolution + uncertainty``.

    Reported together with the calibration numbers because the paper's claim is
    a *dissociation*: reliability blows up while resolution (the discrimination
    content) is preserved.  Showing both halves of the same decomposition is
    what makes that claim falsifiable rather than rhetorical.
    """
    scores = np.asarray(scores, float).ravel()
    labels = np.asarray(labels, float).ravel()
    st = binned_stats(scores, labels, n_bins, scheme)
    base = float(labels.mean()) if labels.size else 0.0
    reliability = float(np.sum(st.weight * (st.mean_score - st.mean_label) ** 2))
    resolution = float(np.sum(st.weight * (st.mean_label - base) ** 2))
    uncertainty = base * (1.0 - base)
    brier = float(np.mean((scores - labels) ** 2)) if scores.size else float("nan")
    return {
        "brier": brier,
        "reliability": reliability,
        "resolution": resolution,
        "uncertainty": uncertainty,
        "residual": brier - (reliability - resolution + uncertainty),
    }


def negative_log_likelihood(scores: np.ndarray, labels: np.ndarray, eps: float = 1e-7) -> float:
    s = np.clip(np.asarray(scores, float).ravel(), eps, 1 - eps)
    y = np.asarray(labels, float).ravel()
    return float(-np.mean(y * np.log(s) + (1 - y) * np.log(1 - s)))


# --------------------------------------------------------------------------
# discrimination
# --------------------------------------------------------------------------
def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Rank-based AUROC with exact tie handling (Mann-Whitney U)."""
    s = np.asarray(scores, float).ravel()
    y = np.asarray(labels, float).ravel()
    pos, neg = y == 1, y == 0
    n_pos, n_neg = int(pos.sum()), int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    sorted_s = s[order]
    i = 0
    while i < len(sorted_s):                       # average ranks within ties
        j = i
        while j + 1 < len(sorted_s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[i : j + 1] = 0.5 * (i + j) + 1.0
        i = j + 1
    r = np.empty_like(ranks)
    r[order] = ranks
    return float((r[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def auprc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Average precision, step-wise with exact tie grouping.

    Scores that tie cannot be ordered by any threshold rule, so precision is
    evaluated once per distinct score rather than once per sample.  Ignoring
    this inflates AP whenever the score distribution saturates - which is the
    normal situation for a hospital-trained model applied to a screening
    cohort, where a large mass of predictions pins to the same value.
    """
    s = np.asarray(scores, float).ravel()
    y = np.asarray(labels, float).ravel()
    n_pos = float(y.sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    s, y = s[order], y[order]
    # index of the last sample of each tie group
    last = np.r_[np.nonzero(np.diff(s))[0], len(s) - 1]
    tp = np.cumsum(y)[last]
    n_pred = (last + 1).astype(float)
    precision = tp / n_pred
    recall = tp / n_pos
    d_recall = np.diff(np.r_[0.0, recall])
    return float(np.sum(precision * d_recall))


# --------------------------------------------------------------------------
# curves + uncertainty
# --------------------------------------------------------------------------
def reliability_curve(
    scores: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
    scheme: BinScheme = "equal_mass",
    min_count: int = 1,
) -> dict[str, np.ndarray]:
    """Points for a reliability diagram plus a per-bin Wilson interval."""
    st = binned_stats(scores, labels, n_bins, scheme)
    mask = st.count >= min_count
    p = st.mean_label[mask]
    n = st.count[mask]
    z = 1.959963984540054
    denom = 1 + z**2 / np.maximum(n, 1)
    centre = (p + z**2 / (2 * np.maximum(n, 1))) / denom
    half = z * np.sqrt(np.maximum(p * (1 - p) / np.maximum(n, 1) + z**2 / (4 * np.maximum(n, 1) ** 2), 0)) / denom
    return {
        "mean_score": st.mean_score[mask],
        "mean_label": p,
        "count": n,
        "ci_low": np.clip(centre - half, 0, 1),
        "ci_high": np.clip(centre + half, 0, 1),
    }


def bootstrap_ci(
    fn,
    *arrays: np.ndarray,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
    stratify: np.ndarray | None = None,
) -> tuple[float, float, float]:
    """Percentile bootstrap ``(point, lo, hi)`` for any metric ``fn(*arrays)``.

    ``stratify`` resamples within strata, which keeps the number of positives
    stable.  For rare labels in a screening cohort an unstratified bootstrap
    routinely draws replicates with zero positives, and the resulting interval
    is an artefact of that, not of the estimator.
    """
    arrays = [np.asarray(a) for a in arrays]
    n = len(arrays[0])
    rng = np.random.default_rng(seed)
    point = float(fn(*arrays))

    if stratify is None:
        idx_pool = [np.arange(n)]
    else:
        stratify = np.asarray(stratify).ravel()
        idx_pool = [np.where(stratify == v)[0] for v in np.unique(stratify)]

    vals = np.empty(n_boot)
    for b in range(n_boot):
        pick = np.concatenate([rng.choice(ix, size=len(ix), replace=True) for ix in idx_pool])
        try:
            vals[b] = fn(*[a[pick] for a in arrays])
        except Exception:
            vals[b] = np.nan
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return point, float("nan"), float("nan")
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return point, float(lo), float(hi)


# --------------------------------------------------------------------------
# reports
# --------------------------------------------------------------------------
@dataclass
class CalibrationReport:
    n: int
    prevalence: float
    mean_score: float
    ece: float
    ece2: float
    ece2_plugin: float
    mce: float
    nll: float
    brier: float
    reliability: float
    resolution: float
    auroc: float
    auprc: float
    n_bins_effective: int
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d.update(d.pop("extra"))
        return d


def calibration_report(
    scores: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
    scheme: BinScheme = "equal_mass",
) -> CalibrationReport:
    """Full single-label report: calibration *and* discrimination side by side."""
    scores = np.asarray(scores, float).ravel()
    labels = np.asarray(labels, float).ravel()
    st = binned_stats(scores, labels, n_bins, scheme)
    bd = brier_decomposition(scores, labels, n_bins, scheme)
    return CalibrationReport(
        n=int(scores.size),
        prevalence=float(labels.mean()) if labels.size else float("nan"),
        mean_score=float(scores.mean()) if scores.size else float("nan"),
        ece=expected_calibration_error(scores, labels, n_bins, scheme),
        ece2=squared_calibration_error(scores, labels, n_bins, scheme, debiased=True),
        ece2_plugin=squared_calibration_error(scores, labels, n_bins, scheme, debiased=False),
        mce=maximum_calibration_error(scores, labels, n_bins, scheme),
        nll=negative_log_likelihood(scores, labels),
        brier=bd["brier"],
        reliability=bd["reliability"],
        resolution=bd["resolution"],
        auroc=auroc(scores, labels),
        auprc=auprc(scores, labels),
        n_bins_effective=st.n_nonempty,
    )


def multilabel_report(
    scores: np.ndarray,
    labels: np.ndarray,
    label_names: list[str],
    n_bins: int = 15,
    scheme: BinScheme = "equal_mass",
) -> "object":
    """Per-label reports stacked into a DataFrame (pandas imported lazily)."""
    import pandas as pd

    scores = np.atleast_2d(np.asarray(scores, float))
    labels = np.atleast_2d(np.asarray(labels, float))
    if scores.shape != labels.shape:
        raise ValueError(f"scores {scores.shape} != labels {labels.shape}")
    if scores.shape[1] != len(label_names):
        raise ValueError(f"{scores.shape[1]} columns but {len(label_names)} label names")
    rows = []
    for j, name in enumerate(label_names):
        rep = calibration_report(scores[:, j], labels[:, j], n_bins, scheme).to_dict()
        rep["label"] = name
        rows.append(rep)
    df = pd.DataFrame(rows)
    return df[["label"] + [c for c in df.columns if c != "label"]]
