# The Korean screening cohort: why it matters, and how to include it without exporting data

## Why this cohort is not just a sixth country

Every publicly available ECG cohort is a hospital population. That makes
hospital-to-hospital transfer a poor instrument for this study, because the two
shifts the decomposition separates move together: sites that see more disease
also see *different* disease. Prevalence and case mix are confounded, and no
amount of data from more hospitals unconfounds them.

An asymptomatic health-screening population breaks the confounding. Prevalence
falls by roughly an order of magnitude — atrial fibrillation at well under 1%
against 10%+ in an emergency department — while the signal-generating process is
essentially unchanged: an ECG is recorded the same way and a bundle branch block
looks the same. That is the regime in which `D_label` and `D_concept` are most
nearly orthogonal, and it is what makes Theorem 1 empirically *identifiable*
rather than merely true.

It is also the deployment case that matters most. Screening is where a
hospital-trained model is most likely to be pointed next, and where a
miscalibrated probability does the most damage per patient, because almost
everyone is negative and the positive predictive value at any fixed threshold
collapses.

## The obstacle, and why it is not a technical one

The cohort requires IRB approval and a data transfer agreement, and its signals
do not leave the institution. The federated protocol below removes the
*technical* obstacle to participation. It does not remove the governance one:
approval is still required, and this protocol is what makes the approval easy to
grant rather than something to work around.

## The protocol

Every estimator in `recalib_kit` depends on the target site only through a small
set of aggregates. Not "can be adapted to" — depends on, structurally:

| Estimator | What it needs from the target |
|---|---|
| Prior estimation (BBSE, partial ID, the unlabeled test) | the score histogram over source-defined bins |
| Theorem 1 decomposition | per-bin count, mean score, event rate |
| Theorem 2 label budget | per-bin count and mean score (unlabeled); event rates for Γ |
| Theorem 3 conformal | nonconformity order statistics at the levels of interest |

So the exchange is:

**Inbound** (coordinating site → institution): the frozen model, the
preprocessing config, and the **source bin edges**.

**Outbound** (institution → coordinating site): one JSON payload per label.
For a 5,000-patient cohort it is about 1.5 kB.

```bash
# inside the institutional environment, after scoring locally
recalib export --scores local_scores.csv --edges source_edges.json \
               --label AF --model-id ecgfm-v1 --cohort kbsmc \
               --out kbsmc_AF.json
```

No signal, no record-level row, no free text, and no identifier crosses the
boundary.

### The bin edges must come from the source

This is the one step that is easy to get wrong. Deriving bins locally would leak
the target score distribution into the bin definition *and* make the exported
histogram incomparable with every other site's. The edges travel inbound and are
used unchanged.

### Disclosure audit

`recalib export` audits the payload before writing it and **refuses** to write
one that contains:

- a non-empty bin describing fewer than `--min-cell` people (default 10), or
- a small bin whose event rate is exactly 0 or 1, where the rate itself
  identifies its members.

The remedy is to merge the offending bin with a neighbour and re-export — never
to lower the threshold. A refusal exits non-zero so it cannot pass unnoticed in
a pipeline.

## Building the local bundle

`ecgcal.data.korea.build` deliberately raises `NotImplementedError`: the file
layout is institution-specific and there is no public path to write against.
Inside the environment, adapt `ecgcal/data/physionet2021.py` — the only
site-specific parts are locating the records and reading the over-read column.
Everything downstream (preprocessing, masking, splitting) is shared, and using
the shared path is what guarantees the exported statistics are comparable with
the other cohorts.

Requirements for the local export:

- 12-lead, 500 Hz, 10 s (the shared chain resamples to 250 Hz);
- a cardiologist over-read mapped to `core6` via `configs/labels.yaml`;
- a patient identifier, so splits are patient-level — the screening population
  has repeat visits and record-level splitting would leak;
- labels the cohort does not adjudicate left **masked**, not set to zero.

## What the coordinating site can then compute

From the payload alone: the full Theorem 1 decomposition, the unlabeled
goodness-of-fit test, partial-identification intervals, the label budget, and
conformal coverage. That is every number this cohort contributes to the paper.

## Reciprocity

The payload is small enough to publish. We recommend the institution do so
alongside the paper: it lets others reproduce this cohort's contribution exactly,
and it costs nothing beyond what has already been disclosed.
