# What could not be retrieved here, and what that affects

This repository was developed in an environment whose outbound network policy
permits PyPI and GitHub and denies the dataset hosts. Recording this plainly
matters more than it might seem: a reader needs to know exactly which claims
rest on cohort data and which do not.

## Denied hosts

| Host | Needed for | Status |
|---|---|---|
| `physionet.org` | PTB-XL, PhysioNet/CinC 2021 (CPSC, Chapman, Ningbo, Georgia), MIMIC-IV-ECG, EchoNext | HTTP 403 at the egress proxy |
| `zenodo.org` | CODE-15% | HTTP 403 at the egress proxy |

These are organisation egress-policy denials, not transient failures. They were
not retried and not routed around; `ecgcal.data.download` distinguishes the two
cases and refuses to retry a policy denial by design.

Two cohorts would have been out of reach regardless:

- **MIMIC-IV-ECG** and **EchoNext** require PhysioNet credentialed access (CITI
  training plus a signed data use agreement). The loaders never attempt an
  anonymous download.
- The **Kangbuk Samsung** screening cohort requires IRB approval and a data
  transfer agreement, and its signals do not leave the institution under any
  circumstances. That is why the federated protocol exists.

## What this does and does not affect

**Unaffected — validated here, on ground truth.** The three theorems are
statements about *estimators*: that a decomposition is identified, that a label
count suffices, that a coverage guarantee survives a bounded perturbation.
Claims of that kind cannot be checked on real cohorts, because on real data the
split between label shift and concept shift is exactly what is unknown. They are
validated against a generator with known `(π_t, η)`, which is the only
instrument that can falsify them, and the validation is in `tests/test_theory.py`:

- the decomposition is an exact identity to machine precision (residual ~1e-17);
- pure label shift gives free fraction > 0.97 and the free correction removes
  > 90% of ECE; pure concept shift gives exactly zero free fraction;
- BBSE is unbiased under label shift and the unlabeled goodness-of-fit test has
  near-nominal unconditional size (0.035–0.065 at nominal 0.05) with power 0.93
  against a concept shift small enough to leave AUROC unchanged;
- partial-identification intervals are sharp — they cover the truth exactly when
  the assumed budget is at least the true shift, and are empty below the data's
  own floor;
- the sample-size bounds scale as ε⁻² and agree to within a few per cent at the
  threshold;
- conformal coverage behaves as Theorem 3 states, including undercoverage by
  approximately the concept-shift budget and its repair by level inflation.

**Affected — awaiting cohort access.** The empirical magnitudes: the size of the
cliff at each real site, the real free fraction per country, the real label bill
per hospital, the collapse map. These are reported in this repository from the
simulator, which is configured with prevalences near published figures but is
not a substitute for the cohorts. Every results table is stamped with
`data_source: simulator` in its `manifest.json`, and `build_cohorts` refuses to
mix real and simulated sites in one table.

## Running it with data access

The code path is identical; only the flag changes.

```bash
python scripts/build_cohorts.py --plan          # ~38 GB across the open cohorts
python scripts/build_cohorts.py --cohorts ptbxl georgia chapman ningbo cpsc code15
python scripts/build_cohorts.py --cohorts mimic --raw /path/to/your/credentialed/copy

python experiments/run_study.py --preset main --data-root data/bundles
python experiments/make_figures.py results/main_foundation
python experiments/make_numbers.py results/main_foundation
```

`scripts/verify_cohorts.py` recomputes record counts from the files on disk and
fails loudly on a mismatch with the registry — several of these datasets have
had records withdrawn between versions (PTB-XL went from 21,837/18,885 records
and patients in v1.0.1 to 21,799/18,869 in v1.0.3), so a version pin is part of
every result table.
