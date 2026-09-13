#!/usr/bin/env python3
"""Structural checks on the manuscript, for environments with no LaTeX toolchain.

    python scripts/check_manuscript.py paper/

Catches the errors that would otherwise surface only at compile time, plus two
that LaTeX would not catch at all: a figure the text references but no run has
produced, and a citation key absent from the bibliography.
"""

from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path


def check(paper: Path) -> list[str]:
    problems: list[str] = []
    tex_files = sorted(paper.glob("*.tex"))
    if not tex_files:
        return [f"no .tex files in {paper}"]

    def _strip_comments(txt: str) -> str:
        # A commented-out example is not a use; scanning raw text made the
        # checker report its own generated header as a missing macro.
        return re.sub(r"(?<!\\)%.*", "", txt)

    text_by_file = {f: _strip_comments(f.read_text()) for f in tex_files}
    joined = "\n".join(text_by_file.values())

    # --- \input targets exist -------------------------------------------------
    for f, txt in text_by_file.items():
        for target in re.findall(r"\\input\{([^}]+)\}", txt):
            cand = paper / (target if target.endswith(".tex") else target + ".tex")
            if not cand.exists():
                problems.append(f"{f.name}: \\input{{{target}}} -> {cand.name} does not exist")

    # --- environments balance -------------------------------------------------
    for f, txt in text_by_file.items():
        begins = Counter(re.findall(r"\\begin\{([^}]+)\}", txt))
        ends = Counter(re.findall(r"\\end\{([^}]+)\}", txt))
        for env in set(begins) | set(ends):
            if begins[env] != ends[env]:
                problems.append(
                    f"{f.name}: environment {env!r} has {begins[env]} \\begin and {ends[env]} \\end"
                )

    # --- braces balance -------------------------------------------------------
    for f, txt in text_by_file.items():
        stripped = txt.replace(r"\{", "").replace(r"\}", "")
        depth = stripped.count("{") - stripped.count("}")
        if depth != 0:
            problems.append(f"{f.name}: {abs(depth)} unbalanced brace(s) ({'{' if depth > 0 else '}'} excess)")

    # --- labels and refs ------------------------------------------------------
    labels = set(re.findall(r"\\label\{([^}]+)\}", joined))
    refs = set(re.findall(r"\\(?:ref|autoref|eqref)\{([^}]+)\}", joined))
    for r in sorted(refs - labels):
        problems.append(f"\\ref{{{r}}} has no matching \\label")
    dup = [k for k, v in Counter(re.findall(r"\\label\{([^}]+)\}", joined)).items() if v > 1]
    for d in dup:
        problems.append(f"label {d!r} is defined more than once")

    # --- citations ------------------------------------------------------------
    bib = paper / "refs.bib"
    if bib.exists():
        keys = set(re.findall(r"@\w+\{([^,]+),", bib.read_text()))
        cited = set()
        for c in re.findall(r"\\cite[a-z]*\{([^}]+)\}", joined):
            cited.update(k.strip() for k in c.split(","))
        for c in sorted(cited - keys):
            problems.append(f"\\cite{{{c}}} is not in refs.bib")

    # --- figures the text expects --------------------------------------------
    for target in sorted(set(re.findall(r"\\includegraphics\[[^\]]*\]\{([^}]+)\}", joined))):
        cand = paper / target
        if not cand.exists() and not cand.with_suffix(".pdf").exists():
            problems.append(f"figure {target} not built (run experiments/make_figures.py)")

    # --- generated numbers ----------------------------------------------------
    numbers = paper / "numbers.tex"
    used = set(re.findall(r"\\NUM\{([A-Za-z]+)\}", joined))
    if not numbers.exists():
        if used:
            problems.append(f"{len(used)} \\NUM macros used but numbers.tex is missing "
                            "(run experiments/make_numbers.py)")
    else:
        defined = set(re.findall(r"\\csname\s*NUM([A-Za-z]+)\\endcsname",
                                 _strip_comments(numbers.read_text())))
        for u in sorted(used - defined):
            problems.append(f"\\NUM{{{u}}} is used but not defined by the current run")
    return problems


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paper", type=Path, nargs="?", default=Path("paper"))
    args = ap.parse_args(argv)

    problems = check(args.paper)
    n_tex = len(list(args.paper.glob("*.tex")))
    print(f"checked {n_tex} .tex file(s) in {args.paper}: {len(problems)} problem(s)")
    for p in problems:
        print(f"  ! {p}")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
