# Working in this repository

Research code for *The Calibration Cliff*. Read this before changing anything
that produces a number in the manuscript.

## What the repository is claiming

Cross-national transfer of an ECG foundation model preserves discrimination and
destroys calibration. That dissociation is the result, so anything that inflates
discrimination or flatters calibration destroys the finding rather than
improving it.

Two consequences that have already caused real bugs here:

- **Patient-level splitting is not optional.** Record-level splitting inflates
  AUROC, and "AUROC is preserved" is the headline. `assert_no_leakage` is called,
  not assumed.
- **A label a cohort does not annotate is `unobserved`, not negative.** Treating
  an annotation gap as a confirmed absence manufactures a prevalence difference,
  and that artefact lands in the label-shift component the study measures. Every
  label matrix carries a mask.

## Layout

- `src/recalib_kit/` — the toolbox. Model-agnostic; must not import anything
  that knows what a lead is.
- `src/ecgcal/` — everything ECG-specific: cohorts, simulator, encoder.
- `experiments/` — pipeline, analysis tables, figures, manuscript numbers.
- `paper/` — manuscript. `numbers.tex` and `table_recal.tex` are **generated**;
  never hand-edit them.

## Rules that exist because breaking them caused a wrong number

1. **No bare `except: continue` around an estimator call.** One hid a NameError
   that silently deleted the entire zero-label column from the results — the
   column the paper's central claim lives in. Record the failure and print it.
2. **Unlabeled methods get the whole site.** `fit_recalibrator(...,
   target_unlabeled=s)`. Passing the small labeled calibration subset made a
   zero-label method appear to improve as labels were added.
3. **Ratios are pooled, never averaged.** `sum(target)/sum(source)`, not
   `mean(target/source)`. A label whose source ECE is near zero otherwise
   dominates a site summary.
4. **Gains are undefined below a floor.** A site moving from ECE 0.0001 to
   0.0006 has not "lost 500%".
5. **Evaluate noise-free functions per sample, then average within a bin.**
   Evaluating a nonlinear map at the bin mean injects a Jensen bias that fakes
   a cross-component interaction.
6. **Measure test size with the source sample redrawn.** Holding one source
   operator fixed measures a conditional size and made a correct test look
   broken.
7. **The debiased ECE estimator is not clipped at zero.** Clipping reintroduces
   positive bias exactly where one would claim a model is well calibrated.
8. **Never write `1 - (1 - p)`.** It equals `p` exactly and returns `0` for any
   `p` below the double-precision resolution of 1 — which is what a model that
   separates a label cleanly produces. It collapsed every conformal calibration
   score to zero, emptied every prediction set, and drove measured coverage to
   0.02 on a procedure whose guarantee is distribution-free. Compute each branch
   directly, and keep the calibration side and the set-construction side written
   as the *same* expression.
9. **Recalibrating is not free.** Broken out by site and label, a fitted map was
   worse than shipping unchanged in a majority of pairs at small budgets, and it
   made the one site with no cliff measurably worse. Any new recalibration
   method is compared per pair against `identity`, not only on the mean.

## Before committing a change that touches an estimator

```bash
pytest -q                                     # 79 tests
python experiments/run_study.py --preset smoke --out /tmp/check   # wiring only
python scripts/check_manuscript.py paper/
```

`smoke` is a wiring check, not a scientific configuration: its source
calibration split is too small to establish source calibration, so its
unlabeled-correction numbers are not interpretable. Use `medium` or `main` for
anything you intend to read.

To re-run the analysis after fixing an estimator, without retraining:

```bash
python experiments/run_study.py --from-scored results/main_foundation \
                               --out results/main_foundation
```

## Figures

Follow `experiments/figstyle.py`: a validated categorical order assigned in
fixed order and never cycled, a single-hue ramp for magnitude, solid hairline
grids, and **no dual axes** — discrimination and calibration get adjacent panels
sharing an x-axis, because their relationship is what is in dispute. With more
than three categories, identity goes on facet or position, not on more hues.

Render and look at the output before calling a figure done; three defects here
were found that way and none of them by a test.
