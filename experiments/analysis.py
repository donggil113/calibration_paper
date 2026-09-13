"""Turning scored sites into the paper's tables.

One row per (site, label) throughout.  Labels are analysed one-vs-rest and
never macro-averaged before the calibration analysis: averaging ECE across
labels of very different prevalence produces a number dominated by the common
labels and hides precisely the rare-label collapse that matters clinically.
Aggregation, where it happens, happens last and is reported alongside the
per-label detail.

Every table carries the sample size and the number of positives it was computed
from, because a calibration estimate from a site with eleven positive cases is
not evidence of anything and the reader must be able to see that.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from recalib_kit.conformal import (
    coverage_report,
    hybrid_transfer_conformal,
    split_conformal,
    weighted_conformal,
)
from recalib_kit.decomposition import decompose, decompose_l1, decompose_unlabeled_budget
from recalib_kit.label_shift import bbse, min_gamma, source_operator, target_histogram, test_label_shift_sufficiency
from recalib_kit.metrics import bin_edges, brier_decomposition, calibration_report, bootstrap_ci, expected_calibration_error
from recalib_kit.recalibrate import fit_recalibrator
from recalib_kit.sample_size import empirical_n_star, n_star_lower_bound, n_star_upper_plugin, residual_shift_gamma

__all__ = ["MIN_POSITIVES", "transfer_table", "decomposition_table", "recalibration_table",
           "nstar_table", "conformal_table", "collapse_map"]

# Below this many positive cases a per-label calibration estimate is reported
# but flagged: with fewer, the binned estimator's variance exceeds any effect
# size the study is looking for.
MIN_POSITIVES = 25


def _source_ref(scored: dict, j: int) -> tuple[np.ndarray, np.ndarray]:
    """The held-out labeled *source* sample used to build label-shift operators."""
    src = next(v for v in scored.values() if v.get("is_source"))
    m = src["source_ref_mask"][:, j]
    return src["source_ref_scores"][m, j], src["source_ref_labels"][m, j]


def _iter_site_label(scored: dict, label_names: list[str]):
    for site, v in scored.items():
        for j, name in enumerate(label_names):
            sel = v["mask"][:, j]
            s, y = v["scores"][sel, j], v["labels"][sel, j].astype(float)
            if sel.sum() < 50 or y.sum() == 0:
                continue
            yield site, v, j, name, s, y


def transfer_table(scored: dict, label_names: list[str], n_boot: int = 400, seed: int = 0) -> pd.DataFrame:
    """Discrimination and calibration side by side, per site and label.

    The two columns that carry the paper are ``auroc`` and ``ece``: the claim is
    that the first is stable across the transfer matrix while the second is not.
    Bootstrap intervals are stratified on the label so replicates cannot lose
    all the positives.
    """
    src_prev = {}
    src = next(v for v in scored.values() if v.get("is_source"))
    for j, name in enumerate(label_names):
        m = src["source_ref_mask"][:, j]
        src_prev[name] = float(src["source_ref_labels"][m, j].mean())

    rows = []
    for site, v, j, name, s, y in _iter_site_label(scored, label_names):
        rep = calibration_report(s, y)
        bd = brier_decomposition(s, y)
        _, a_lo, a_hi = bootstrap_ci(lambda ss, yy: calibration_report(ss, yy).auroc, s, y,
                                     n_boot=n_boot, seed=seed, stratify=y)
        _, e_lo, e_hi = bootstrap_ci(lambda ss, yy: expected_calibration_error(ss, yy), s, y,
                                     n_boot=n_boot, seed=seed, stratify=y)
        rows.append({
            "site": site, "country": v["country"], "is_source": v["is_source"], "label": name,
            "n": rep.n, "n_positive": int(y.sum()),
            "prevalence": rep.prevalence, "prevalence_source": src_prev[name],
            "prevalence_ratio": rep.prevalence / max(src_prev[name], 1e-9),
            "mean_score": rep.mean_score,
            "auroc": rep.auroc, "auroc_lo": a_lo, "auroc_hi": a_hi,
            "auprc": rep.auprc,
            "ece": rep.ece, "ece_lo": e_lo, "ece_hi": e_hi,
            "ece2": rep.ece2, "mce": rep.mce, "nll": rep.nll,
            "brier": bd["brier"], "reliability": bd["reliability"], "resolution": bd["resolution"],
            "reliable_estimate": bool(y.sum() >= MIN_POSITIVES),
        })
    return pd.DataFrame(rows)


def decomposition_table(scored: dict, label_names: list[str]) -> pd.DataFrame:
    """Theorem 1 applied at every site and label.

    Three prior regimes are reported side by side because their difference *is*
    a result: ``oracle`` uses the true target prevalence, ``bbse`` estimates it
    from unlabeled target scores, and the gap between the two free fractions is
    what a site actually loses by not knowing its own prevalence.
    """
    rows = []
    for site, v, j, name, s, y in _iter_site_label(scored, label_names):
        if v["is_source"]:
            continue
        ss, sy = _source_ref(scored, j)
        if ss.size < 100 or sy.sum() < 10:
            continue
        pi_s = float(sy.mean())

        d_or = decompose(s, y, pi_s=pi_s, pi_t=float(y.mean()))
        d_bb = decompose(s, y, pi_s=pi_s, pi_t=None, source_scores=ss, source_labels=sy)
        l1 = decompose_l1(s, y, pi_s=pi_s, pi_t=float(y.mean()))
        budget = decompose_unlabeled_budget(s, pi_s, ss, sy)

        rows.append({
            "site": site, "country": v["country"], "label": name,
            "n": int(s.size), "n_positive": int(y.sum()),
            "pi_s": pi_s, "pi_t_true": float(y.mean()), "pi_t_bbse": budget["pi_t_bbse"],
            "prior_ratio": float(y.mean()) / max(pi_s, 1e-9),
            "total": d_or.total, "d_label": d_or.d_label, "d_concept": d_or.d_concept,
            "interaction": d_or.interaction, "residual": d_or.residual,
            "free_fraction": d_or.free_fraction,
            "free_fraction_bbse": d_bb.free_fraction,
            "realized_gain": d_or.realized_free_gain,
            "realized_gain_bbse": d_bb.realized_free_gain,
            "ece_before": d_or.ece_before, "ece_after_prior_fix": d_or.ece_after_prior_fix,
            "ece_l1": l1["ece_total"], "d_label_l1": l1["d_label_l1"], "d_concept_l1": l1["d_concept_l1"],
            "sign_agreement": l1["sign_agreement"],
            "gamma_min": budget["gamma_min"], "gof_p": budget["gof_p_value"],
            "concept_shift_detected": budget["concept_shift_detected"],
            "bbse_degenerate": budget["bbse_degenerate"],
            "bbse_sigma_min": budget["bbse_sigma_min"],
            "bbse_trustworthy": budget["bbse_trustworthy"],
            "reliable_estimate": bool(y.sum() >= MIN_POSITIVES),
        })
    return pd.DataFrame(rows)


def recalibration_table(
    scored: dict,
    label_names: list[str],
    n_cal_grid: tuple[int, ...] = (0, 25, 50, 100, 200, 400, 800),
    methods: tuple[str, ...] = (
        "identity", "prior_correction", "prior_correction_gated", "prior_correction_minimax",
        "temperature", "platt", "isotonic", "hybrid", "hybrid_minimax",
    ),
    n_repeats: int = 20,
    seed: int = 0,
) -> pd.DataFrame:
    """Post-hoc ECE per method as the target label budget grows.

    Calibration sets are drawn at the **patient** level and evaluated on the
    disjoint remainder, so the reported budget is in patients chart-reviewed -
    the unit a hospital plans in - rather than in records.
    """
    from ecgcal.data.splits import calibration_split

    rows = []
    for site, v, j, name, s, y in _iter_site_label(scored, label_names):
        if v["is_source"]:
            continue
        ss, sy = _source_ref(scored, j)
        if ss.size < 100 or sy.sum() < 10:
            continue
        pi_s = float(sy.mean())
        pid = v["patient_ids"][v["mask"][:, j]]

        for n_cal in n_cal_grid:
            for method in methods:
                vals = []
                reps = 1 if (n_cal == 0 and method in ("identity", "prior_correction")) else n_repeats
                for r in range(reps):
                    cal, ev = calibration_split(len(s), max(n_cal, 1), seed=seed + r, patient_ids=pid)
                    if len(ev) < 100:
                        continue
                    try:
                        if n_cal == 0 and not unlabeled:
                            rec = fit_recalibrator("identity", s[ev][:1], y[ev][:1])
                        else:
                            rec = fit_recalibrator(
                                method, s[cal], y[cal], pi_s=pi_s,
                                source_scores=ss, source_labels=sy,
                                # the unlabeled half sees the whole site, which
                                # costs nothing; only the labeled half is rationed
                                target_unlabeled=s,
                            )
                        vals.append(expected_calibration_error(rec.transform(s[ev]), y[ev]))
                    except Exception:
                        continue
                if not vals:
                    continue
                rows.append({
                    "site": site, "country": v["country"], "label": name,
                    "method": method, "n_cal": n_cal, "n_repeats": len(vals),
                    "ece_mean": float(np.mean(vals)), "ece_median": float(np.median(vals)),
                    "ece_q90": float(np.quantile(vals, 0.9)),
                    "n_positive": int(y.sum()),
                    "reliable_estimate": bool(y.sum() >= MIN_POSITIVES),
                })
    return pd.DataFrame(rows)


def nstar_table(
    scored: dict,
    label_names: list[str],
    eps: float = 0.02,
    alpha: float = 0.1,
    methods: tuple[str, ...] = (
        "prior_correction", "prior_correction_minimax", "temperature", "hybrid", "hybrid_minimax",
    ),
    n_repeats: int = 60,
    seed: int = 0,
) -> pd.DataFrame:
    """Theorem 2 at every site: the label bill, empirical and analytic.

    ``n_star_empirical`` is the deliverable number - the smallest calibration
    set meeting the coverage requirement in simulation.  The analytic bounds
    bracket it and are reported alongside so a site can sanity-check the
    empirical value rather than take it on faith.
    """
    rows = []
    for site, v, j, name, s, y in _iter_site_label(scored, label_names):
        if v["is_source"]:
            continue
        ss, sy = _source_ref(scored, j)
        if ss.size < 100 or sy.sum() < 10 or s.size < 400:
            continue
        pi_s = float(sy.mean())
        pi_t = float(y.mean())
        gamma = residual_shift_gamma(s, y, pi_s, pi_t)
        ub = n_star_upper_plugin(eps, alpha, s, gamma=gamma)
        lb = n_star_lower_bound(eps, 0.25, gamma=gamma)

        for method in methods:
            try:
                res = empirical_n_star(
                    s, y, eps=eps, alpha=alpha, method=method, n_repeats=n_repeats,
                    pi_s=pi_s, source_scores=ss, source_labels=sy, seed=seed,
                )
            except Exception:
                continue
            rows.append({
                "site": site, "country": v["country"], "label": name, "method": method,
                "n_star_empirical": res.n_star,
                "zero_shot_coverage": res.zero_shot_coverage,
                "gamma": gamma, "eps": eps, "alpha": alpha,
                "n_star_upper": ub["n_star"] if ub["feasible"] else np.inf,
                "n_star_lower": lb["n_star"],
                "upper_feasible": ub["feasible"],
                "mean_pred_var": ub.get("mean_pred_var", np.nan),
                "n_positive": int(y.sum()), "n": int(s.size),
                "reliable_estimate": bool(y.sum() >= MIN_POSITIVES),
            })
    return pd.DataFrame(rows)


def conformal_table(
    scored: dict,
    label_names: list[str],
    alpha: float = 0.1,
    n_cal: int = 100,
    gammas: tuple[float, ...] = (0.0, 0.02, 0.05),
    n_repeats: int = 50,
    seed: int = 0,
) -> pd.DataFrame:
    """Theorem 3 at every site: coverage over repeated calibration draws.

    A single draw says nothing - the realised coverage of a valid procedure is
    spread by roughly sqrt(alpha(1-alpha)/n) around its nominal level - so
    coverage is averaged over draws and the 5th percentile is reported too,
    since a protocol has to hold on a bad day as well as on average.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for site, v, j, name, s, y in _iter_site_label(scored, label_names):
        if v["is_source"]:
            continue
        ss, sy = _source_ref(scored, j)
        if ss.size < 100 or sy.sum() < 10 or s.size < 500:
            continue
        pi_s, pi_t = float(sy.mean()), float(y.mean())

        acc: dict[str, list] = {}
        for _ in range(n_repeats):
            perm = rng.permutation(len(s))
            cal, ev = perm[:n_cal], perm[n_cal:]
            cand = {
                "split_target": split_conformal(s[cal], y[cal], s[ev], alpha),
                "weighted_source": weighted_conformal(ss, sy, s[ev], pi_s, pi_t, alpha),
            }
            for g in gammas:
                cand[f"hybrid_g{g:g}"] = hybrid_transfer_conformal(
                    s[cal], y[cal], ss, sy, s[ev], pi_s, pi_t, alpha=alpha, gamma=g
                )
            for k, cs in cand.items():
                r = coverage_report(cs, y[ev])
                acc.setdefault(k, []).append((r["coverage"], r["review_rate"], r["mean_set_size"]))

        for k, vals in acc.items():
            a = np.array(vals)
            rows.append({
                "site": site, "country": v["country"], "label": name, "method": k,
                "alpha": alpha, "n_cal": n_cal,
                "coverage_mean": float(a[:, 0].mean()),
                "coverage_p05": float(np.quantile(a[:, 0], 0.05)),
                "coverage_sd": float(a[:, 0].std()),
                "review_rate": float(a[:, 1].mean()),
                "mean_set_size": float(a[:, 2].mean()),
                "valid": bool(a[:, 0].mean() >= 1 - alpha),
                "n_positive": int(y.sum()),
                "reliable_estimate": bool(y.sum() >= MIN_POSITIVES),
            })
    return pd.DataFrame(rows)


def collapse_map(transfer: pd.DataFrame, decomposition: pd.DataFrame) -> pd.DataFrame:
    """Per-country summary: how far calibration falls, and how much is free.

    Ratios are **pooled**, not averaged.  ``ece_ratio`` is
    ``sum(target ECE) / sum(source ECE)`` over labels rather than the mean of
    per-label ratios, and the same for the realised gain.  A mean of ratios is
    dominated by labels whose source ECE is near zero, where the ratio is an
    arbitrarily large number divided by noise; pooling weights each label by how
    much miscalibration it actually contributes, which is also the quantity a
    deploying site cares about.
    """
    src = transfer[transfer.is_source]
    src_ece = src.groupby("label")["ece"].mean().to_dict()
    src_auroc = src.groupby("label")["auroc"].mean().to_dict()

    t = transfer[~transfer.is_source].copy()
    t["src_ece"] = t["label"].map(src_ece)
    t["auroc_delta"] = t["auroc"] - t["label"].map(src_auroc)

    rows = []
    for (site, country), g in t.groupby(["site", "country"]):
        denom = g["src_ece"].sum()
        rows.append({
            "site": site, "country": country, "n_labels": len(g),
            "auroc_mean": g["auroc"].mean(),
            "auroc_delta": g["auroc_delta"].mean(),
            "ece_mean": g["ece"].mean(),
            "ece_ratio": (g["ece"].sum() / denom) if denom > 0 else np.nan,
            "prevalence_ratio": g["prevalence_ratio"].median(),
        })
    agg = pd.DataFrame(rows)

    if not decomposition.empty:
        d = decomposition[decomposition.reliable_estimate]
        drows = []
        for site, g in d.groupby("site"):
            tot = g["total"].sum()
            before, after = g["ece_before"].sum(), g["ece_after_prior_fix"].sum()
            drows.append({
                "site": site,
                # pooled, so a label with negligible miscalibration cannot
                # dominate through a near-zero denominator
                "free_fraction": (g["d_label"].sum() / tot) if tot > 0 else np.nan,
                "realized_gain": ((before - after) / before) if before > 0 else np.nan,
                "interaction_share": (g["interaction"].sum() / tot) if tot > 0 else np.nan,
                "gamma_min": g["gamma_min"].mean(),
                "concept_shift_frac": g["concept_shift_detected"].mean(),
                "bbse_trustworthy_frac": (
                    g["bbse_trustworthy"].mean() if "bbse_trustworthy" in g else np.nan
                ),
                "n_labels_decomposed": len(g),
            })
        agg = agg.merge(pd.DataFrame(drows), on="site", how="left")
    return agg.sort_values("ece_ratio", ascending=False).reset_index(drop=True)
