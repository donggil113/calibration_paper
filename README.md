# The Calibration Cliff

**What breaks when a medical foundation model crosses a border is not
discrimination but calibration.**

An ECG foundation model evaluated in a second country typically retains its
AUROC. That is the evidence usually offered for readiness to deploy, and it is
close to uninformative: AUROC is invariant to any monotone transformation of the
score, so a model whose predicted probability of atrial fibrillation is
uniformly five times too high scores identically to one that is correct. Every
clinical use that is not pure ranking — a referral threshold, an expected-value
calculation, a risk quoted to a patient, a number submitted to a regulator —
depends on the half that AUROC discards.

This repository contains the study of that failure and the toolbox for repairing
it.

---

## The claim, in three parts

1. **The phenomenon.** Across a source-to-target transfer matrix spanning five
   countries, discrimination is preserved while calibration degrades severely —
   a dissociation, not a general loss of performance. A same-country site acts
   as the negative control that separates a *national* effect from a generic
   site effect.

2. **The decomposition.** Target calibration error splits exactly into a
   **label-shift** component, removable with *no target labels*, and a
   **concept-shift** component that requires them, plus a signed interaction
   term that flags sites where the two cancel and a naive correction backfires.
   Identification conditions are given, relaxing black-box shift estimation to
   a rank condition checkable on source data; where nothing can be defended,
   sharp partial-identification bounds replace a false point estimate.

3. **The price.** Matching upper and lower bounds on the number of labeled
   target recordings a new hospital needs, with an explicit threshold below
   which the answer is **zero**, and a sample-size formula whose only unknown is
   the model's mean predictive variance on the site's own *unlabeled* data.

---

## Quickstart

```bash
pip install -e ".[all]"
pytest -q                                        # 56 tests, ~30 s

# Run the whole study on the simulator (no data access needed)
python experiments/run_study.py --preset smoke   # ~1 min, checks the wiring
python experiments/run_study.py --preset main    # the full study

# Figures and the manuscript's numbers
python experiments/make_figures.py results/main_foundation
python experiments/make_numbers.py results/main_foundation
```

On real cohorts, the same code path:

```bash
python scripts/build_cohorts.py --plan            # what to fetch, and what needs credentials
python scripts/build_cohorts.py --cohorts ptbxl georgia chapman ningbo cpsc code15
python experiments/run_study.py --preset main --data-root data/bundles
```

## `recalib-kit`: the deployment workflow

Three commands, in the order they cost money.

```bash
# 1. FREE. Unlabeled target scores only. Is a prior correction licensed here?
recalib audit  --target site_scores.csv --source source_scores.csv

# 2. If not, price the labeled work before committing to it.
recalib budget --target site_scores.csv --source source_scores.csv --eps 0.02

# 3. Apply a correction and report what it did.
recalib fix    --target site_scores.csv --source source_scores.csv \
               --method prior_correction_minimax --out calibrated.csv
```

A site should not reach `fix` without having run `audit`. The ungated prior
correction is *not* risk-free: where the label-shift assumption fails, the
prevalence it depends on can be wrong by an order of magnitude, and applying it
then makes calibration substantially worse than shipping the model unchanged.
`prior_correction_minimax` is the recommended default — it damps itself by how
little the unlabeled data pins the prevalence, and was never worse than
shipping unchanged in any regime we tested.

### For a cohort that cannot export data

Every estimator here depends on the target site only through per-bin counts,
mean scores and event rates, plus conformal order statistics. A cohort of 5,000
patients exports as ~1.5 kB:

```bash
recalib export --scores local.csv --edges source_edges.json \
               --label AF --model-id ecgfm-v1 --out payload.json
```

The export is audited before it is written and refuses any payload containing a
cell describing fewer than ten people. See [docs/KOREA_COHORT.md](docs/KOREA_COHORT.md).

---

## Layout

```
src/recalib_kit/     the toolbox — model-agnostic, no ECG assumptions
  metrics.py         calibration + discrimination, with binning bias handled explicitly
  label_shift.py     BBSE, MLLS, anchor restriction, partial identification, unlabeled test
  decomposition.py   Theorem 1
  sample_size.py     Theorem 2
  conformal.py       Theorem 3
  recalibrate.py     identity → prior correction → gated → minimax → hybrid
  cli.py             audit / budget / fix / export

src/ecgcal/          everything that knows what a lead or an SCP code is
  data/              registry, loaders, harmonisation, preprocessing, splits, federated export
  sim/               multi-site ECG generator with controllable label and concept shift
  models/            encoder, masked-patch pretraining, linear probe

experiments/         pipeline, analysis tables, figures, manuscript numbers
paper/               manuscript; numbers.tex is generated, never hand-edited
tests/               56 tests, including regressions for every defect found
docs/                data access, the federated protocol, environment limits
```

The toolbox is deliberately separable from the ECG code: it is meant to be used
by people with their own model and their own site.

## Reproducibility

- Every figure and every number in the manuscript is generated from the CSVs a
  run writes. `make_numbers.py` emits LaTeX macros and **verifies that every
  `\NUM{...}` the manuscript uses actually resolves**; an unresolved one renders
  in red rather than silently keeping a stale value.
- Splits are patient-level with a leakage assertion that is executed, not
  assumed. Record-level splitting inflates discrimination, and this paper's
  headline is that discrimination is *preserved* — leakage would manufacture the
  result.
- Scored arrays are persisted (`scored.npz`), so any table or figure can be
  regenerated without retraining.

## Data

Seven public cohorts across five countries plus one restricted institutional
cohort. See [docs/DATA_ACCESS.md](docs/DATA_ACCESS.md) for what is open, what
needs PhysioNet credentialing, and exact retrieval steps; and
[docs/BLOCKED_RESOURCES.md](docs/BLOCKED_RESOURCES.md) for what could not be
retrieved in the environment this repository was developed in, and what that
does and does not affect.

## License

MIT for the code. The cohorts carry their own licences and data use agreements;
none of them are redistributed here.
