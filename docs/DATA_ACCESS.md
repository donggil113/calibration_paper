# Obtaining the cohorts

Nine cohorts in three access tiers. Nothing here is redistributed; each carries
its own licence and data use agreement.

Run `python scripts/build_cohorts.py --plan` for a live report of what is
present, what is missing, and roughly how much it weighs (~38 GB for the open
set, ~87 GB more for MIMIC-IV-ECG).

## Open — download directly

| Cohort | Country | Records | Where |
|---|---|---:|---|
| PTB-XL v1.0.3 | DE | 21,799 | `physionet.org/content/ptb-xl/1.0.3/` |
| CPSC-2018 | CN | 6,877 | `physionet.org/content/challenge-2021/1.0.3/` |
| Chapman-Shaoxing | CN | 10,247 | same archive |
| Ningbo | CN | 34,905 | same archive |
| Georgia 12-lead | US | 10,344 | same archive |
| CODE-15% | BR | 345,779 | `zenodo.org/records/4916206` |

**Pin the version.** PTB-XL v1.0.1 reported 21,837 records from 18,885 patients;
v1.0.3 reports 21,799 from 18,869 after withdrawals. Both figures appear in the
literature. `scripts/verify_cohorts.py` recomputes counts from the files on disk
and fails if they disagree with the registry by more than 1%.

**CODE-15 arrives padded.** Every record is zero-padded to 4,096 samples at
400 Hz. The padding length is a near-perfect site fingerprint and a network will
use it in preference to the ECG, so it is stripped before anything else —
`ecgcal.data.preprocess.strip_zero_padding`, called first in the chain.

## Credentialed — PhysioNet account plus a signed agreement

| Cohort | Country | Records | Labels from |
|---|---|---:|---|
| MIMIC-IV-ECG v1.0 | US | 800,035 | cart interpretation statements |
| EchoNext | US | ~100,000 | paired echocardiography |

1. Create a PhysioNet account.
2. Complete CITI "Data or Specimens Only Research" training.
3. Submit the training report for credentialing (review takes days).
4. Sign the dataset-specific DUA on its landing page.
5. Download with your credentials, then build from your local copy:
   `python scripts/build_cohorts.py --cohorts mimic --raw /path/to/copy`

The loaders never attempt an anonymous download for these.

**MIMIC labels are the study's weakest link, deliberately handled as such.**
They are extracted from free-text cart statements by a versioned rule set in
`ecgcal.data.harmonize`, which drops any statement carrying a negation or hedge
— "no evidence of atrial fibrillation" must not set the AF label on precisely
the records a clinician thought worth commenting on. Every headline result
should be repeated under a stricter rule variant.

**EchoNext is the control for label provenance.** Its labels come from a
separate imaging modality rather than from reading the ECG, so a calibration
cliff that persists there cannot be an artefact of how sites annotate tracings.

## Restricted — institutional, never exported

**Kangbuk Samsung Health Study** (KR, health screening). Requires IRB approval
and a data transfer agreement. Signals do not leave the institution. Build the
bundle inside that environment and export only sufficient statistics; see
[KOREA_COHORT.md](KOREA_COHORT.md).

## The harmonised label space

CODE-15 scores exactly six rhythm and conduction abnormalities, which bounds any
label set shared by every cohort:

- **core6** — AF (incl. flutter), IAVB, RBBB, LBBB, SB, STACH. Every cohort.
- **ext12** — adds PAC, PVC, LVH, STTC, MI, NORM. PhysioNet family and MIMIC.

Mappings are in `configs/labels.yaml` with SNOMED-CT codes (PhysioNet/CinC 2021
convention including its three equivalence classes), PTB-XL SCP statements, and
CODE-15 columns, plus an `excluded:` block giving the reason for every label
deliberately left out.

**A label a cohort does not annotate is recorded as unobserved, never as
negative.** Treating an annotation gap as a confirmed absence manufactures a
prevalence difference out of nothing, and that artefact lands squarely in the
label-shift component this study exists to measure. Every label matrix carries
an observation mask; masked entries contribute to neither training nor
evaluation.
