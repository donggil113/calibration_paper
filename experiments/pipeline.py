"""The study pipeline: build cohorts, train one frozen model, measure every site.

The design follows the claim.  A single model is trained on one source country
and then **frozen** - no target labels, no target statistics, no per-site
re-normalisation touch it again.  Every number reported for a target site is
therefore a property of transfer, not of adaptation.

Order of operations, and why each step is where it is:

1. **Cohorts.**  Real bundles if they are on disk, otherwise the simulator.
   The same analysis code runs on both, so the switch is one flag rather than a
   parallel codebase that can drift.
2. **Pretrain** on the pooled *unlabeled* union of all sites.  This is the
   foundation-model protocol, and it is also the control: an encoder that has
   already seen every site's acquisition characteristics cannot have its
   transfer failure blamed on unfamiliarity with them.
3. **Probe** on source labels only, over the frozen encoder.
4. **Calibrate on a held-out source split.**  This is not a detail.  Theorem 1
   decomposes the miscalibration of a model that *starts* calibrated on its own
   domain; without this step the target numbers would be contaminated by
   residual source miscalibration and the decomposition would attribute it to
   shift.
5. **Score every site** and hand the scores to recalib_kit.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ecgcal.models.probe import LinearProbe, embed_signals, fit_linear_probe
from ecgcal.models.ssl import SSLConfig, pretrain_masked, transfer_backbone
from ecgcal.models.train import TrainConfig, predict_scores, set_seed, train_model
from ecgcal.sim.generator import LABELS, SITE_LIBRARY, simulate_site
from recalib_kit.recalibrate import TemperatureScaling

__all__ = ["StudyConfig", "SiteData", "build_cohorts", "train_source_model", "score_sites",
           "run_pipeline", "MIN_SOURCE_POSITIVES"]

#: Below this many positive source cases per label, a fitted temperature does not
#: establish source calibration and the source operator is too ill-conditioned
#: for the unlabeled estimators built on it.
MIN_SOURCE_POSITIVES = 100


@dataclass
class StudyConfig:
    source: str = "mimic_like"
    targets: tuple[str, ...] = ("georgia_like", "ptbxl_like", "chapman_like", "code15_like", "korea_like")
    n_source: int = 6000
    n_target: int = 4000
    arm: str = "foundation"          # 'foundation' (SSL + linear probe) or 'supervised'
    model_size: str = "small"
    ssl_epochs: int = 8
    sup_epochs: int = 10
    fs: int = 250
    seconds: float = 10.0
    seed: int = 0
    data_root: str | None = None     # if set and bundles exist, use real cohorts
    label_tier: str = "core6"
    out_dir: str = "results"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SiteData:
    """One site's signals, labels and metadata, whatever the source."""

    name: str
    country: str
    signals: np.ndarray
    labels: np.ndarray
    label_names: list[str]
    patient_ids: np.ndarray
    mask: np.ndarray | None = None
    is_source: bool = False
    extra: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.signals)

    @property
    def observed(self) -> np.ndarray:
        return np.ones_like(self.labels, bool) if self.mask is None else self.mask


# --------------------------------------------------------------------------
# 1. cohorts
# --------------------------------------------------------------------------
def build_cohorts(cfg: StudyConfig) -> dict[str, SiteData]:
    """Load real cohort bundles when available, else simulate.

    The real path is preferred whenever ``data_root`` contains built bundles;
    it is never silently mixed with simulated sites, because a results table
    combining the two would be uninterpretable.
    """
    if cfg.data_root:
        root = Path(cfg.data_root)
        from ecgcal.data.base import bundle_exists, load_bundle

        wanted = [cfg.source, *cfg.targets]
        have = [k for k in wanted if bundle_exists(root, k)]
        if len(have) == len(wanted):
            print(f"[cohorts] using real bundles from {root}")
            out = {}
            for k in wanted:
                b = load_bundle(root, k)
                out[k] = SiteData(
                    name=k, country=b.meta["cohort"].iloc[0] if "cohort" in b.meta else "??",
                    signals=b.signals, labels=b.labels.y, label_names=b.labels.names,
                    patient_ids=b.patient_ids, mask=b.labels.mask, is_source=(k == cfg.source),
                )
            return out
        missing = sorted(set(wanted) - set(have))
        raise FileNotFoundError(
            f"data_root={root} is set but bundles are missing for {missing}. "
            "Build them first (scripts/build_cohorts.py), or clear data_root to run on the "
            "simulator. Mixing real and simulated sites in one table is not supported."
        )

    print("[cohorts] simulating sites (no real bundles requested)")
    out = {}
    for i, name in enumerate([cfg.source, *cfg.targets]):
        site = SITE_LIBRARY[name]
        n = cfg.n_source if name == cfg.source else cfg.n_target
        d = simulate_site(site, n, fs=cfg.fs, seconds=cfg.seconds,
                          seed=cfg.seed + 1000 * i, n_patients=max(1, int(n * 0.85)))
        out[name] = SiteData(
            name=name, country=d["country"], signals=d["signals"], labels=d["labels"],
            label_names=d["label_names"], patient_ids=d["meta"]["patient_id"],
            is_source=(name == cfg.source),
            extra={"age": d["meta"]["age"], "sex": d["meta"]["sex"], "hr": d["meta"]["heart_rate"]},
        )
    return out


# --------------------------------------------------------------------------
# 2-4. model
# --------------------------------------------------------------------------
@dataclass
class SourceModel:
    """A frozen scorer plus the temperature that calibrated it on source data."""

    encoder: object
    probe: LinearProbe | None
    temperatures: dict[str, float]
    label_names: list[str]
    arm: str
    train_info: dict

    def raw_scores(self, signals: np.ndarray) -> np.ndarray:
        if self.probe is not None:
            return self.probe.predict_from_signals(self.encoder, signals)
        return predict_scores(self.encoder, signals)

    def scores(self, signals: np.ndarray) -> np.ndarray:
        """Source-calibrated scores: the model as it would actually be shipped."""
        raw = self.raw_scores(signals)
        out = np.empty_like(raw)
        for j, name in enumerate(self.label_names):
            ts = TemperatureScaling()
            ts.temperature_ = self.temperatures.get(name, 1.0)
            out[:, j] = ts.transform(raw[:, j])
        return out


def train_source_model(
    cohorts: dict[str, SiteData],
    cfg: StudyConfig,
    verbose: bool = True,
) -> tuple[SourceModel, dict[str, np.ndarray]]:
    """Pretrain, probe, and calibrate on held-out source data.

    Returns the frozen model and the index split used on the source site, so
    downstream analysis can be sure the source evaluation is out of sample.
    """
    from ecgcal.data.splits import assert_no_leakage, patient_split

    set_seed(cfg.seed)
    src = cohorts[cfg.source]
    names = src.label_names

    # Patient-level source split: fit / calibrate / evaluate.
    tr, cal, ev = patient_split(src.patient_ids, (0.6, 0.2, 0.2), seed=cfg.seed)
    assert_no_leakage(src.patient_ids, [tr, cal, ev])
    if verbose:
        print(f"[source] {cfg.source}: {len(tr)} train / {len(cal)} calibrate / {len(ev)} eval")

    t0 = time.time()
    info: dict = {"arm": cfg.arm}

    if cfg.arm == "foundation":
        # Pretrain on the pooled unlabeled union of every site.
        pool = np.concatenate([c.signals for c in cohorts.values()])
        if verbose:
            print(f"[ssl] masked-patch pretraining on {len(pool)} pooled unlabeled records")
        backbone, ssl_info = pretrain_masked(
            pool, SSLConfig(epochs=cfg.ssl_epochs, size=cfg.model_size, seed=cfg.seed), verbose=verbose
        )
        encoder = transfer_backbone(backbone, len(names), freeze=True)
        probe = fit_linear_probe(encoder, src.signals[tr], src.labels[tr], names,
                                 None if src.mask is None else src.mask[tr])
        if verbose:
            print(f"[probe] {probe.report()}")
        info["ssl"] = {"final_recon_mse": ssl_info["history"][-1]["recon_mse"],
                       "epochs": cfg.ssl_epochs, "n_pool": int(len(pool))}
        info["probe"] = probe.report()
    elif cfg.arm == "supervised":
        encoder, hist = train_model(
            src.signals[tr], src.labels[tr], None if src.mask is None else src.mask[tr],
            src.signals[cal], src.labels[cal], None if src.mask is None else src.mask[cal],
            names, TrainConfig(epochs=cfg.sup_epochs, size=cfg.model_size, seed=cfg.seed), verbose=verbose,
        )
        probe = None
        info["supervised"] = {"best_epoch": hist["best_epoch"], "best_val_macro_auroc": hist["best_val_macro_auroc"]}
    else:
        raise ValueError(f"unknown arm {cfg.arm!r}; use 'foundation' or 'supervised'")

    model = SourceModel(encoder=encoder, probe=probe, temperatures={}, label_names=names,
                        arm=cfg.arm, train_info=info)

    # --- calibrate on the held-out source split -----------------------------
    # Theorem 1 decomposes the miscalibration of a model that is calibrated on
    # its own domain.  Skipping this would push residual source miscalibration
    # into the target numbers and let the decomposition attribute it to shift.
    raw_cal = model.raw_scores(src.signals[cal])
    temps, thin = {}, []
    for j, name in enumerate(names):
        sel = src.observed[cal][:, j]
        yj = src.labels[cal][sel, j]
        n_pos = int(yj.sum())
        if sel.sum() < 50 or n_pos in (0, int(sel.sum())):
            temps[name] = 1.0
            continue
        if n_pos < MIN_SOURCE_POSITIVES:
            thin.append((name, n_pos))
        ts = TemperatureScaling().fit(raw_cal[sel, j], yj)
        temps[name] = float(ts.temperature_)
    model.temperatures = temps
    info["temperatures"] = temps
    info["source_calibration_n"] = int(len(cal))
    info["thin_source_labels"] = thin

    if thin and verbose:
        # The whole decomposition assumes the model starts calibrated on its own
        # domain.  A temperature fitted on a handful of positives does not
        # deliver that, the source operator built from the same split is badly
        # conditioned, and every downstream unlabeled estimate inherits both -
        # which looks like the free correction failing rather than like the
        # source split being too small.
        print(f"[warn] source calibration split has {len(cal)} records but only "
              f"{', '.join(f'{n}={k}' for n, k in thin)} positives.")
        print(f"[warn] Assumption 1 (source calibration) is not reliably established "
              f"below ~{MIN_SOURCE_POSITIVES} positives per label. Increase n_source "
              f"or the calibration fraction; the 'smoke' preset is a wiring check, "
              f"not a scientific configuration.")
    info["seconds"] = time.time() - t0
    if verbose:
        print(f"[calibrate] source temperatures: " +
              ", ".join(f"{k}={v:.3f}" for k, v in temps.items()))
        print(f"[source] model ready in {info['seconds']:.1f}s")
    return model, {"train": tr, "calibrate": cal, "eval": ev}


# --------------------------------------------------------------------------
# 5. scoring
# --------------------------------------------------------------------------
def score_sites(
    model: SourceModel,
    cohorts: dict[str, SiteData],
    source_splits: dict[str, np.ndarray],
    cfg: StudyConfig,
    verbose: bool = True,
) -> dict[str, dict]:
    """Score every site with the frozen, source-calibrated model.

    For the source site only the held-out evaluation split is scored, so the
    source row of every table is honestly out of sample.  The source
    *calibration* split is retained separately: the label-shift estimators need
    a labeled source sample to build their operator from, and it must be one the
    probe did not train on.
    """
    out: dict[str, dict] = {}
    for name, site in cohorts.items():
        t0 = time.time()
        if site.is_source:
            idx = source_splits["eval"]
            cal_idx = source_splits["calibrate"]
            out[name] = {
                "scores": model.scores(site.signals[idx]),
                "labels": site.labels[idx],
                "mask": site.observed[idx],
                "patient_ids": site.patient_ids[idx],
                "source_ref_scores": model.scores(site.signals[cal_idx]),
                "source_ref_labels": site.labels[cal_idx],
                "source_ref_mask": site.observed[cal_idx],
                "country": site.country,
                "is_source": True,
                "n": int(len(idx)),
            }
        else:
            out[name] = {
                "scores": model.scores(site.signals),
                "labels": site.labels,
                "mask": site.observed,
                "patient_ids": site.patient_ids,
                "country": site.country,
                "is_source": False,
                "n": int(len(site)),
            }
        if verbose:
            print(f"[score] {name}: {out[name]['n']} records in {time.time() - t0:.1f}s")
    return out


def run_pipeline(cfg: StudyConfig, verbose: bool = True) -> dict:
    """Cohorts -> frozen model -> scores for every site."""
    t0 = time.time()
    cohorts = build_cohorts(cfg)
    if verbose:
        for k, v in cohorts.items():
            prev = dict(zip(v.label_names, np.round(v.labels.mean(0), 4)))
            print(f"  {k:14s} n={len(v):6d}  country={v.country}  prevalence={prev}")
    model, splits = train_source_model(cohorts, cfg, verbose)
    scored = score_sites(model, cohorts, splits, cfg, verbose)
    if verbose:
        print(f"[pipeline] total {time.time() - t0:.1f}s")
    return {"config": cfg.to_dict(), "cohorts": cohorts, "model": model,
            "scored": scored, "splits": splits, "label_names": model.label_names}
