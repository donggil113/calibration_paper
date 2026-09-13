#!/usr/bin/env python3
"""Emit the manuscript's numbers as LaTeX macros, straight from a results directory.

    python experiments/make_numbers.py results/main_foundation

Every quantity quoted in the paper is defined here and nowhere else.  The point
is that a number in the manuscript cannot drift from the run that produced it:
if the run changes, the macros change, and if a macro is missing the PDF says so
in red rather than silently keeping a stale value.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def _fmt(v, spec: str = "{:.3f}") -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "n/a"
    return spec.format(v)


def build(res: Path) -> dict[str, str]:
    def load(n):
        p = res / f"{n}.csv"
        return pd.read_csv(p) if p.exists() else pd.DataFrame()

    transfer, decomp = load("transfer"), load("decomposition")
    recal, nstar, conf = load("recalibration"), load("nstar"), load("conformal")
    cmap = load("collapse_map")
    manifest = json.loads((res / "manifest.json").read_text()) if (res / "manifest.json").exists() else {}

    m: dict[str, str] = {}
    t = transfer[transfer.reliable_estimate] if "reliable_estimate" in transfer else transfer
    src, tgt = t[t.is_source], t[~t.is_source]

    # --- scope -------------------------------------------------------------
    m["nsites"] = str(transfer.site.nunique())
    m["ncountries"] = str(transfer.country.nunique())
    m["nlabels"] = str(transfer.label.nunique())
    m["nrecords"] = f"{sum(v['n'] for v in manifest.get('sites', {}).values()):,}"
    m["datasource"] = manifest.get("data_source", "unknown").replace("_", " ")

    # --- the dissociation ---------------------------------------------------
    if len(src) and len(tgt):
        d = tgt.auroc.mean() - src.auroc.mean()
        m["aurocsource"] = _fmt(src.auroc.mean())
        m["auroctargetmean"] = _fmt(tgt.auroc.mean())
        m["aurocdelta"] = _fmt(d, "{:+.3f}")
        per = tgt.groupby("site").auroc.mean() - src.auroc.mean()
        m["aurocdeltarange"] = f"{per.min():+.3f} to {per.max():+.3f}"
        m["ecesource"] = _fmt(src.ece.mean(), "{:.4f}")
        m["ecetargetmean"] = _fmt(tgt.ece.mean(), "{:.4f}")

    if len(cmap):
        c = cmap.dropna(subset=["ece_ratio"])
        m["eceratiomax"] = _fmt(c.ece_ratio.max(), "{:.1f}") + r"$\times$"
        m["eceratiomaxsite"] = str(c.loc[c.ece_ratio.idxmax(), "site"]).replace("_", " ")
        m["eceratiomin"] = _fmt(c.ece_ratio.min(), "{:.2f}") + r"$\times$"
        ctl = c[c.site == "georgia_like"]
        if len(ctl):
            m["controlratio"] = _fmt(float(ctl.ece_ratio.iloc[0]), "{:.2f}") + r"$\times$"
            m["controlaurocdelta"] = _fmt(float(ctl.auroc_delta.iloc[0]), "{:+.3f}")
        for _, r in c.iterrows():
            key = str(r.site).replace("_", "")
            m[f"{key}eceratio"] = _fmt(r.ece_ratio, "{:.2f}") + r"$\times$"
            m[f"{key}aurocdelta"] = _fmt(r.auroc_delta, "{:+.3f}")
            if "free_fraction" in r and np.isfinite(r.free_fraction):
                m[f"{key}free"] = _fmt(100 * np.clip(r.free_fraction, 0, 1), "{:.0f}") + r"\%"

    # --- Theorem 1 ----------------------------------------------------------
    if len(decomp):
        dd = decomp[decomp.reliable_estimate] if "reliable_estimate" in decomp else decomp
        tot = dd.total.sum()
        m["freefractionpooled"] = _fmt(100 * dd.d_label.sum() / tot, "{:.0f}") + r"\%" if tot > 0 else "n/a"
        m["conceptfractionpooled"] = _fmt(100 * dd.d_concept.sum() / tot, "{:.0f}") + r"\%" if tot > 0 else "n/a"
        m["interactionsharepooled"] = _fmt(100 * dd.interaction.sum() / tot, "{:+.0f}") + r"\%" if tot > 0 else "n/a"
        m["decompresidualmax"] = f"{np.abs(dd.residual).max():.1e}"
        m["nnegativeinteraction"] = str(int((dd.interaction < 0).sum()))
        m["ndecomprows"] = str(len(dd))
        if "concept_shift_detected" in dd:
            m["gofrejectfrac"] = _fmt(100 * dd.concept_shift_detected.mean(), "{:.0f}") + r"\%"
        if "bbse_trustworthy" in dd:
            ok, bad = dd[dd.bbse_trustworthy], dd[~dd.bbse_trustworthy]
            # Median absolute log ratio, not mean relative error: the latter is
            # dominated by a single pair whose concept shift lay along the
            # label-shift cone, where the test is structurally blind, and quoting
            # it would either overstate the test or bury the real limitation.
            def logerr(g):
                return float(np.median(np.abs(np.log(
                    g.pi_t_bbse.clip(lower=1e-6) / g.pi_t_true.clip(lower=1e-9)))))

            if len(ok):
                m["bbselogerrtrusted"] = _fmt(logerr(ok), "{:.2f}")
                m["nbbsetrusted"] = str(len(ok))
            if len(bad):
                m["bbselogerruntrusted"] = _fmt(logerr(bad), "{:.2f}")
                m["nbbseuntrusted"] = str(len(bad))
            if "prior_ratio_implausible" in dd:
                m["nimplausibleprior"] = str(int(dd.prior_ratio_implausible.sum()))

    # --- Theorem 2 ----------------------------------------------------------
    if len(nstar):
        ns = nstar[nstar.reliable_estimate] if "reliable_estimate" in nstar else nstar
        hyb = ns[ns.method == "hybrid"]
        if len(hyb):
            zero = hyb[hyb.n_star_empirical.fillna(-1) == 0]
            m["nzerolabelsites"] = str(len(zero))
            m["nstarrows"] = str(len(hyb))
            fin = hyb.n_star_empirical.dropna()
            if len(fin):
                m["nstarmedian"] = _fmt(float(fin.median()), "{:.0f}")
                m["nstarmax"] = _fmt(float(fin.max()), "{:.0f}")
            m["nstarunreachable"] = str(int(hyb.n_star_empirical.isna().sum()))
        tmp = ns[ns.method == "temperature"]
        if len(tmp):
            m["ntempunreachable"] = str(int(tmp.n_star_empirical.isna().sum()))
            m["ntemprows"] = str(len(tmp))
        m["epsbudget"] = _fmt(float(ns.eps.iloc[0]), "{:.3f}")
        m["alphalevel"] = _fmt(float(ns.alpha.iloc[0]), "{:.2f}")
        # Below the threshold the sufficient bound applies and the empirical
        # requirement should sit inside it; above it, only the necessary bound
        # is meaningful.  The two never apply together, so no ratio between them
        # is reported - an earlier version quoted one and it was meaningless.
        below = ns[ns.upper_feasible & ns.n_star_empirical.notna()]
        if len(below):
            inside = (below.n_star_empirical <= below.n_star_upper).mean()
            m["nstarwithinbound"] = _fmt(100 * float(inside), "{:.0f}") + r"\%"
            m["nstarbelowthreshold"] = str(len(below))

    # --- recalibration sweep ------------------------------------------------
    if len(recal):
        r = recal[recal.reliable_estimate] if "reliable_estimate" in recal else recal
        base = r[(r.method == "identity")].ece_mean.mean()
        for meth, key in [("prior_correction", "prior"), ("temperature", "temp"),
                          ("hybrid", "hyb"), ("isotonic", "iso")]:
            sub = r[(r.method == meth) & (r.n_cal.isin([0, 100]))]
            sub = sub[sub.n_cal == (0 if meth == "prior_correction" else 100)]
            if len(sub):
                m[f"ece{key}"] = _fmt(sub.ece_mean.mean(), "{:.4f}")
        if np.isfinite(base):
            m["eceidentity"] = _fmt(base, "{:.4f}")

    # --- Theorem 3 ----------------------------------------------------------
    if len(conf):
        c = conf[conf.reliable_estimate] if "reliable_estimate" in conf else conf
        for meth, key in [("split_target", "split"), ("weighted_source", "wsrc"),
                          ("hybrid_g0.05", "hybg")]:
            sub = c[c.method == meth]
            if len(sub):
                m[f"cov{key}"] = _fmt(sub.coverage_mean.mean(), "{:.3f}")
                m[f"cov{key}p05"] = _fmt(sub.coverage_p05.mean(), "{:.3f}")
        m["conformalalpha"] = _fmt(float(c.alpha.iloc[0]), "{:.2f}")
        m["conformalncal"] = str(int(c.n_cal.iloc[0]))

    m["runseconds"] = _fmt(manifest.get("elapsed_seconds"), "{:.0f}")
    return m


def recal_table_tex(res: Path) -> str | None:
    """The recalibration comparison, as a LaTeX table generated from the run.

    Generated rather than typed for the same reason the numbers are: a table
    transcribed by hand is a table that silently goes stale.
    """
    path = res / "recalibration.csv"
    if not path.exists():
        return None
    r = pd.read_csv(path)
    if "reliable_estimate" in r:
        r = r[r.reliable_estimate]
    if r.empty:
        return None

    rows = [
        ("identity", 0, "ship unchanged", "0"),
        ("prior_correction", 0, "prior correction", "0"),
        ("prior_correction_gated", 0, "\\quad gated by the unlabeled test", "0"),
        ("prior_correction_minimax", 0, "\\quad minimax over the identified set", "0"),
        ("temperature", 100, "temperature scaling", "100"),
        ("platt", 100, "Platt scaling", "100"),
        ("isotonic", 100, "isotonic regression", "100"),
        ("hybrid", 100, "hybrid (Thm 2)", "100"),
        ("hybrid_minimax", 100, "\\quad with the minimax prior", "100"),
    ]
    out = [
        r"\begin{table}[t]", r"\centering\small",
        r"\begin{tabular}{lrrr}", r"\toprule",
        r"Method & Target labels & Mean ECE & 90th pct ECE \\", r"\midrule",
    ]
    for method, n_cal, label, budget in rows:
        sub = r[(r.method == method) & (r.n_cal == n_cal)]
        if sub.empty:
            continue
        out.append(
            f"{label} & {budget} & {sub.ece_mean.mean():.4f} & {sub.ece_q90.mean():.4f} " + r"\\"
        )
    out += [
        r"\bottomrule", r"\end{tabular}",
        r"\caption{Recalibration across target sites, averaged over labels with at least",
        r"25 positive cases. The label budget is the number of target cases adjudicated;",
        r"the zero-label rows use only unlabeled target scores plus a labelled source",
        r"sample. Generated from the run by \texttt{experiments/make\_numbers.py}.}",
        r"\label{tab:recal}", r"\end{table}", "",
    ]
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results", type=Path)
    ap.add_argument("--out", type=Path, default=ROOT / "paper" / "numbers.tex")
    args = ap.parse_args(argv)

    macros = build(args.results)
    lines = [
        "% Generated by experiments/make_numbers.py -- do not edit by hand.",
        f"% source: {args.results}",
        "",
        "% \\NUM{key} expands to the value if this run produced it, and to a red",
        "% marker if it did not.  A stale number can therefore never survive a",
        "% rerun unnoticed: it either updates or it turns red.",
        "\\makeatletter",
        "\\providecommand{\\NUM}[1]{%",
        "  \\@ifundefined{NUM#1}{\\textcolor{red}{[#1: not produced by this run]}}%",
        "                      {\\csname NUM#1\\endcsname}}",
        "\\makeatother",
        "",
    ]
    for k, v in sorted(macros.items()):
        key = "".join(ch for ch in k if ch.isalpha())
        lines.append(rf"\expandafter\newcommand\csname NUM{key}\endcsname{{{v}}}")
    args.out.write_text("\n".join(lines) + "\n")
    print(f"wrote {args.out} with {len(macros)} macros")

    tbl = recal_table_tex(args.results)
    if tbl:
        (args.out.parent / "table_recal.tex").write_text(tbl)
        print(f"wrote {args.out.parent / 'table_recal.tex'}")

    # --- verify every \NUM{...} used in the manuscript actually resolves ------
    used: dict[str, list[str]] = {}
    for tex in sorted(args.out.parent.glob("*.tex")):
        if tex.name == args.out.name:
            continue
        for key in re.findall(r"\\NUM\{([A-Za-z]+)\}", tex.read_text()):
            used.setdefault(key, []).append(tex.name)
    have = {"".join(c for c in k if c.isalpha()) for k in macros}
    missing = {k: v for k, v in used.items() if k not in have}
    unused = sorted(have - set(used))

    print(f"\nmanuscript uses {len(used)} distinct macros; {len(missing)} unresolved")
    for k, files in sorted(missing.items()):
        near = sorted(h for h in have if h.startswith(k[:6]) or k.startswith(h[:6]))
        hint = f"  (did you mean: {', '.join(near[:3])})" if near else ""
        print(f"  MISSING \\NUM{{{k}}} used in {', '.join(sorted(set(files)))}{hint}")
    if unused:
        print(f"  ({len(unused)} macros generated but unused, which is fine)")
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
