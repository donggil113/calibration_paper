#!/usr/bin/env python3
"""Build the paper's figures from a results directory.

    python experiments/make_figures.py results/main_foundation

Every figure reads only the CSVs written by run_study.py, so a figure can always
be regenerated from a stored result without rerunning the model.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "src"))

import matplotlib.pyplot as plt  # noqa: E402

from figstyle import (  # noqa: E402
    CATEGORICAL, GRID, INK, INK_MUTED, INK_SECONDARY, SITE_LABEL, SITE_ORDER,
    apply_style, despine, sequential_cmap, site_color, title,
)


def _order_sites(sites) -> list[str]:
    known = [s for s in SITE_ORDER if s in set(sites)]
    return known + sorted(set(sites) - set(known))


def _lab(site: str) -> str:
    return SITE_LABEL.get(site, site)


# ---------------------------------------------------------------- Figure 1
def fig_dissociation(transfer: pd.DataFrame, out: Path) -> Path:
    """The headline: discrimination holds across sites, calibration does not.

    Two panels sharing one x-axis rather than one panel with two y-scales.  A
    dual axis would let the reader read a relationship out of an arbitrary
    scale alignment, and the relationship between these two quantities is
    exactly what is in dispute.

    Identity is carried by position (which site) and by facet, not by giving
    each of six labels its own colour - six scatter hues are not separable
    under common colour-vision deficiency.
    """
    df = transfer[transfer.reliable_estimate].copy()
    sites = _order_sites(df.site.unique())
    src = df[df.is_source]
    src_auroc = float(src.auroc.mean()) if len(src) else np.nan
    src_ece = float(src.ece.mean()) if len(src) else np.nan

    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.6), sharex=True,
                             gridspec_kw={"hspace": 0.28})
    x = np.arange(len(sites))

    for ax, col, ref, name, fmt in (
        (axes[0], "auroc", src_auroc, "Discrimination (AUROC)", "{:.3f}"),
        (axes[1], "ece", src_ece, "Calibration error (ECE)", "{:.3f}"),
    ):
        for i, s in enumerate(sites):
            d = df[df.site == s]
            if d.empty:
                continue
            jitter = np.linspace(-0.16, 0.16, len(d))
            ax.scatter(i + jitter, d[col], s=22, color=site_color(s), zorder=3,
                       edgecolors="white", linewidths=0.7)
            ax.plot([i - 0.26, i + 0.26], [d[col].mean()] * 2, color=site_color(s),
                    lw=2.2, zorder=4, solid_capstyle="round")
        if np.isfinite(ref):
            ax.axhline(ref, color=INK_MUTED, lw=0.9, zorder=1)
            ax.text(len(sites) - 0.45, ref, "  source", color=INK_MUTED, fontsize=7.5,
                    va="center", ha="left")
        ax.set_ylabel(name)
        despine(ax)
        ax.grid(axis="x", visible=False)

    axes[0].set_ylim(max(0.5, df.auroc.min() - 0.05), 1.005)
    axes[1].set_ylim(0, max(df.ece.max() * 1.15, 0.02))
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([_lab(s) for s in sites], rotation=18, ha="right")
    title(axes[0], "Discrimination survives cross-national transfer",
          "one point per label; bar marks the site mean")
    title(axes[1], "Calibration does not")

    p = out / "fig1_dissociation.pdf"
    fig.savefig(p)
    fig.savefig(p.with_suffix(".png"))
    plt.close(fig)
    return p


# ---------------------------------------------------------------- Figure 2
def fig_reliability(scored_npz: Path | None, transfer: pd.DataFrame, out: Path,
                    label: str | None = None) -> Path | None:
    """Reliability diagrams per site, raw versus the free prior correction.

    Wilson intervals are drawn per bin.  Without them a reliability diagram in a
    rare-label, small-site regime shows wiggles that are entirely sampling noise
    and invites the reader to over-read the shape.
    """
    if scored_npz is None or not Path(scored_npz).exists():
        return None
    from recalib_kit.label_shift import prior_correction
    from recalib_kit.metrics import reliability_curve

    z = np.load(scored_npz, allow_pickle=True)
    names = [str(s) for s in z["label_names"]]
    sites = _order_sites([str(s) for s in z["sites"]])
    if label is None:                       # the label with the most positives at the source
        src = transfer[transfer.is_source]
        label = src.sort_values("n_positive", ascending=False).label.iloc[0] if len(src) else names[0]
    j = names.index(label)

    ncol = min(3, len(sites))
    nrow = int(np.ceil(len(sites) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 3.0 * nrow), squeeze=False)
    pi_s = float(z[f"{sites[0]}_labels"][:, j].mean())
    for k, s in enumerate(sites):
        ax = axes[k // ncol][k % ncol]
        sc, yy = z[f"{s}_scores"][:, j], z[f"{s}_labels"][:, j].astype(float)
        ax.plot([0, 1], [0, 1], color=INK_MUTED, lw=0.9, zorder=1)
        rc = reliability_curve(sc, yy, n_bins=10, min_count=15)
        ax.errorbar(rc["mean_score"], rc["mean_label"],
                    yerr=[rc["mean_label"] - rc["ci_low"], rc["ci_high"] - rc["mean_label"]],
                    fmt="o-", color=CATEGORICAL[0], ms=4.5, lw=1.6, elinewidth=0.9,
                    capsize=0, zorder=3, label="as shipped")
        corrected = prior_correction(sc, pi_s, float(yy.mean()))
        rc2 = reliability_curve(corrected, yy, n_bins=10, min_count=15)
        ax.plot(rc2["mean_score"], rc2["mean_label"], "s-", color=CATEGORICAL[1],
                ms=4.0, lw=1.6, zorder=2, label="+ prior correction (no labels)")
        lim = max(0.02, float(np.nanmax([rc["mean_score"].max() if rc["mean_score"].size else 0,
                                         rc["mean_label"].max() if rc["mean_label"].size else 0])) * 1.15)
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_title(_lab(s), color=INK, fontsize=9)
        despine(ax)
        if k % ncol == 0:
            ax.set_ylabel("observed frequency")
        if k // ncol == nrow - 1:
            ax.set_xlabel("predicted probability")
    for k in range(len(sites), nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    axes[0][0].legend(loc="upper left", fontsize=7.2)
    fig.suptitle(f"Reliability at each site — {label}", x=0.005, ha="left",
                 color=INK, fontsize=10.5, fontweight="semibold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    p = out / "fig2_reliability.pdf"
    fig.savefig(p)
    fig.savefig(p.with_suffix(".png"))
    plt.close(fig)
    return p


# ---------------------------------------------------------------- Figure 3
def fig_decomposition(decomp: pd.DataFrame, out: Path) -> Path | None:
    """Theorem 1 per site: how much of the cliff is free.

    Grouped bars rather than a stack, because the interaction term is signed and
    a stacked bar cannot show a negative segment honestly.  A negative
    interaction is not a rounding artefact - it means the two shifts partially
    cancel, so the site looks better calibrated than either component alone
    implies and a naive prior correction can make things worse.
    """
    d = decomp[decomp.reliable_estimate].copy()
    if d.empty:
        return None
    g = d.groupby("site").agg(d_label=("d_label", "mean"), d_concept=("d_concept", "mean"),
                              interaction=("interaction", "mean"), total=("total", "mean"),
                              free=("free_fraction", "mean")).reset_index()
    sites = _order_sites(g.site)
    g = g.set_index("site").loc[sites].reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6), gridspec_kw={"width_ratios": [1.5, 1]})
    y = np.arange(len(g))
    h = 0.24
    comps = [("d_label", "label shift (free)", CATEGORICAL[0]),
             ("d_concept", "concept shift (needs labels)", CATEGORICAL[1]),
             ("interaction", "interaction", CATEGORICAL[2])]
    for k, (col, lab, c) in enumerate(comps):
        axes[0].barh(y + (k - 1) * h, g[col], height=h * 0.88, color=c, label=lab, zorder=3)
    axes[0].axvline(0, color=INK_SECONDARY, lw=0.9, zorder=4)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([_lab(s) for s in g.site])
    axes[0].invert_yaxis()
    axes[0].set_xlabel("component of squared calibration error")
    axes[0].legend(loc="lower right", fontsize=7.5)
    despine(axes[0])
    axes[0].grid(axis="y", visible=False)
    title(axes[0], "Decomposition of the calibration cliff",
          "positive interaction adds; negative means the two shifts cancel")

    axes[1].barh(y, np.clip(g.free, 0, 1), height=0.5, color=CATEGORICAL[0], zorder=3)
    for i, v in enumerate(np.clip(g.free, 0, 1)):
        axes[1].text(min(v + 0.02, 0.98), i, f"{v:.0%}", va="center", ha="left",
                     fontsize=8, color=INK_SECONDARY)
    axes[1].set_yticks(y)
    axes[1].set_yticklabels([])
    axes[1].invert_yaxis()
    axes[1].set_xlim(0, 1.12)
    axes[1].set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    axes[1].set_xticklabels(["0", "25%", "50%", "75%", "100%"])
    axes[1].set_xlabel("share removable with no target labels")
    despine(axes[1])
    axes[1].grid(axis="y", visible=False)
    title(axes[1], "Free fraction")

    fig.tight_layout()
    p = out / "fig3_decomposition.pdf"
    fig.savefig(p)
    fig.savefig(p.with_suffix(".png"))
    plt.close(fig)
    return p


# ---------------------------------------------------------------- Figure 4
def fig_sample_size(recal: pd.DataFrame, nstar: pd.DataFrame, out: Path) -> Path | None:
    """Theorem 2: calibration error against the target label budget.

    The x-axis is the number of target patients chart-reviewed, which is the
    quantity a hospital actually decides about.  Zero is plotted explicitly,
    because for a site near pure label shift the honest answer is zero.
    """
    if recal.empty:
        return None
    r = recal[recal.reliable_estimate]
    if r.empty:
        return None
    sites = _order_sites(r.site.unique())
    methods = [("identity", "ship unchanged", CATEGORICAL[7]),
               ("prior_correction", "prior correction (0 labels)", CATEGORICAL[2]),
               ("temperature", "temperature scaling", CATEGORICAL[1]),
               ("hybrid", "hybrid (Thm 2)", CATEGORICAL[0])]
    ncol = min(3, len(sites))
    nrow = int(np.ceil(len(sites) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 2.9 * nrow), squeeze=False, sharey=True)

    for k, s in enumerate(sites):
        ax = axes[k // ncol][k % ncol]
        sub = r[r.site == s]
        for m, lab, c in methods:
            mm = sub[sub.method == m].groupby("n_cal").ece_median.mean().sort_index()
            if mm.empty:
                continue
            xs = np.maximum(mm.index.values, 0.7)      # log axis: place n=0 at the left edge
            ax.plot(xs, mm.values, "o-", color=c, ms=3.6, lw=1.6, label=lab, zorder=3)
        ax.set_xscale("log")
        ax.set_title(_lab(s), color=INK, fontsize=9)
        despine(ax)
        if k % ncol == 0:
            ax.set_ylabel("ECE after recalibration")
        if k // ncol == nrow - 1:
            ax.set_xlabel("target patients labelled")
    for k in range(len(sites), nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    axes[0][0].legend(loc="upper right", fontsize=7)
    fig.suptitle("What a new site has to pay", x=0.005, ha="left", color=INK,
                 fontsize=10.5, fontweight="semibold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    p = out / "fig4_sample_size.pdf"
    fig.savefig(p)
    fig.savefig(p.with_suffix(".png"))
    plt.close(fig)
    return p


# ---------------------------------------------------------------- Figure 5
def fig_collapse_map(transfer: pd.DataFrame, decomp: pd.DataFrame, out: Path) -> Path | None:
    """The collapse map: calibration degradation and free fraction by site and label.

    A single-hue ramp, light to dark, because both quantities are magnitudes.
    A rainbow would imply ordered categories that do not exist and would not
    survive greyscale printing.
    """
    t = transfer[~transfer.is_source]
    if t.empty:
        return None
    src = transfer[transfer.is_source].groupby("label").ece.mean()
    piv = t.pivot_table(index="site", columns="label", values="ece", aggfunc="mean")
    ratio = piv.divide(src.reindex(piv.columns).clip(lower=1e-9), axis=1)
    sites = _order_sites(ratio.index)
    ratio = ratio.loc[sites]

    panels = [(ratio, "ECE relative to source site", "{:.1f}×")]
    if not decomp.empty:
        ff = decomp.pivot_table(index="site", columns="label", values="free_fraction", aggfunc="mean")
        ff = ff.reindex(index=[s for s in sites if s in ff.index])
        panels.append((ff, "Share fixable with no labels", "{:.0%}"))

    fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 0.52 * len(sites) + 2.2),
                             squeeze=False)
    cmap = sequential_cmap()
    for ax, (mat, name, fmt) in zip(axes[0], panels):
        vals = mat.to_numpy(dtype=float)
        im = ax.imshow(vals, cmap=cmap, aspect="auto",
                       vmin=np.nanmin(vals), vmax=np.nanmax(vals))
        ax.set_xticks(range(mat.shape[1]))
        ax.set_xticklabels(mat.columns, rotation=0, fontsize=8)
        ax.set_yticks(range(mat.shape[0]))
        ax.set_yticklabels([_lab(s) for s in mat.index], fontsize=8)
        ax.grid(False)
        despine(ax, keep=())
        # Direct labels in every cell: with a handful of cells the grid *is* the
        # table, and a colour-only heatmap is unreadable in greyscale.
        rng = np.nanmax(vals) - np.nanmin(vals)
        for i in range(mat.shape[0]):
            for j2 in range(mat.shape[1]):
                v = vals[i, j2]
                if not np.isfinite(v):
                    continue
                dark = (v - np.nanmin(vals)) > 0.6 * rng if rng > 0 else False
                ax.text(j2, i, fmt.format(v), ha="center", va="center", fontsize=7.4,
                        color="white" if dark else INK_SECONDARY)
        cb = fig.colorbar(im, ax=ax, fraction=0.032, pad=0.02)
        cb.outline.set_visible(False)
        cb.ax.tick_params(labelsize=7, length=0, colors=INK_SECONDARY)
        title(ax, name)
    fig.suptitle("Collapse map", x=0.005, ha="left", color=INK, fontsize=10.5, fontweight="semibold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    p = out / "fig5_collapse_map.pdf"
    fig.savefig(p)
    fig.savefig(p.with_suffix(".png"))
    plt.close(fig)
    return p


# ---------------------------------------------------------------- Figure 6
def fig_conformal(conf: pd.DataFrame, out: Path) -> Path | None:
    """Theorem 3: coverage of each conformal procedure against its nominal level."""
    if conf.empty:
        return None
    c = conf[conf.reliable_estimate]
    if c.empty:
        return None
    order = ["split_target", "weighted_source", "hybrid_g0", "hybrid_g0.02", "hybrid_g0.05"]
    pretty = {"split_target": "split conformal\n(target labels)",
              "weighted_source": "reweighted source\n(0 labels)",
              "hybrid_g0": "hybrid γ=0", "hybrid_g0.02": "hybrid γ=0.02",
              "hybrid_g0.05": "hybrid γ=0.05"}
    methods = [m for m in order if m in set(c.method)]
    sites = _order_sites(c.site.unique())
    alpha = float(c.alpha.iloc[0])

    fig, ax = plt.subplots(figsize=(7.6, 3.8))
    w = 0.8 / max(len(sites), 1)
    x = np.arange(len(methods))
    for i, s in enumerate(sites):
        sub = c[c.site == s].groupby("method").coverage_mean.mean()
        vals = [sub.get(m, np.nan) for m in methods]
        ax.bar(x + (i - (len(sites) - 1) / 2) * w, vals, width=w * 0.86,
               color=site_color(s), label=_lab(s), zorder=3)
    ax.axhline(1 - alpha, color=INK_SECONDARY, lw=1.0, zorder=4)
    ax.text(len(methods) - 0.4, 1 - alpha, f"  nominal {1 - alpha:.0%}", fontsize=7.5,
            color=INK_SECONDARY, va="center", ha="left")
    ax.set_xticks(x)
    ax.set_xticklabels([pretty[m] for m in methods], fontsize=7.8)
    lo = min(0.8, float(np.nanmin(c.coverage_mean)) - 0.02)
    ax.set_ylim(lo, 1.0)
    ax.set_ylabel("empirical coverage")
    ax.legend(fontsize=7.2, ncol=min(3, len(sites)), loc="lower left")
    despine(ax)
    ax.grid(axis="x", visible=False)
    title(ax, "Conformal coverage under cross-national transfer",
          "mean over repeated calibration draws; bars below the line are invalid")
    fig.tight_layout()
    p = out / "fig6_conformal.pdf"
    fig.savefig(p)
    fig.savefig(p.with_suffix(".png"))
    plt.close(fig)
    return p


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--label", default=None, help="label to show in the reliability figure")
    args = ap.parse_args(argv)

    res = args.results
    out = args.out or (ROOT / "paper" / "figures")
    out.mkdir(parents=True, exist_ok=True)
    apply_style()

    def load(name):
        p = res / f"{name}.csv"
        return pd.read_csv(p) if p.exists() else pd.DataFrame()

    transfer, decomp = load("transfer"), load("decomposition")
    recal, nstar, conf = load("recalibration"), load("nstar"), load("conformal")
    if transfer.empty:
        print(f"no transfer.csv in {res}", file=sys.stderr)
        return 1

    made = [
        fig_dissociation(transfer, out),
        fig_reliability(res / "scored.npz", transfer, out, args.label),
        fig_decomposition(decomp, out),
        fig_sample_size(recal, nstar, out),
        fig_collapse_map(transfer, decomp, out),
        fig_conformal(conf, out),
    ]
    for p in made:
        print(f"  {'wrote' if p else 'skipped'} {p if p else '(no data)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
