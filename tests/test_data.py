"""Data-layer tests: harmonisation, preprocessing, splitting, federated export."""

import numpy as np
import pandas as pd
import pytest

from ecgcal.data.harmonize import HarmonisedLabels, map_code15, map_ptbxl_scp, map_snomed, map_text_statements
from ecgcal.data.preprocess import (
    PreprocessConfig, fixed_window, preprocess_record, robust_scale, strip_zero_padding,
)
from ecgcal.data.registry import get_cohort, load_cohorts, load_label_space
from ecgcal.data.splits import assert_no_leakage, calibration_split, patient_split, stratified_patient_split


# ------------------------------------------------------------------ registry
def test_label_tiers_are_nested_and_core6_maps_to_code15():
    core, ext = load_label_space("core6"), load_label_space("ext12")
    assert set(core.names) <= set(ext.names)
    assert all(ext.labels[n]["code15"] for n in core.names), "core6 must be CODE-15 representable"


def test_every_cohort_declares_an_access_tier():
    for c in load_cohorts().values():
        assert c.access in {"open", "credentialed", "restricted"}, c.key


def test_unknown_cohort_names_fail_loudly():
    with pytest.raises(KeyError, match="unknown cohort"):
        get_cohort("not_a_cohort")


# -------------------------------------------------------------- harmonisation
def test_unscored_labels_are_masked_not_negative():
    """The single most consequential rule in the data layer: an annotation gap
    must not become a fake prevalence difference."""
    space = load_label_space("ext12")
    h = map_snomed([["164889003"], [], ["164889003"]], space, scored_codes={"164889003"})
    af = space.index("AF")
    assert h.mask[:, af].all() and h.y[:, af].tolist() == [1, 0, 1]
    lvh = space.index("LVH")
    assert not h.mask[:, lvh].any()
    assert np.isnan(h.prevalence()["LVH"])          # not 0.0


def test_code15_masks_the_six_labels_it_does_not_score():
    space = load_label_space("ext12")
    df = pd.DataFrame({"AF": [1, 0], "1dAVb": [0, 1], "RBBB": [0, 0],
                       "LBBB": [0, 0], "SB": [1, 0], "ST": [0, 0]})
    h = map_code15(df, space)
    assert h.mask[:, space.index("AF")].all()
    assert not h.mask[:, space.index("LVH")].any()
    assert h.n_observed()["MI"] == 0


def test_text_rules_reject_negations_and_hedges():
    space = load_label_space("ext12")
    h = map_text_statements([
        ["Atrial fibrillation with rapid ventricular response"],
        ["No evidence of atrial fibrillation"],
        ["Cannot exclude myocardial infarction"],
        ["Rule out atrial fibrillation"],
        ["Sinus bradycardia"],
    ], space)
    assert h.y[:, space.index("AF")].tolist() == [1, 0, 0, 0, 0]
    assert h.y[:, space.index("MI")].tolist() == [0, 0, 0, 0, 0]
    assert h.y[:, space.index("SB")].tolist() == [0, 0, 0, 0, 1]


def test_ptbxl_likelihood_threshold_changes_prevalence():
    space = load_label_space("core6")
    scp = [{"AFIB": 15.0}, {"AFIB": 100.0}, {"SBRAD": 80.0}]
    lo = map_ptbxl_scp(scp, space, likelihood_threshold=0.0)
    hi = map_ptbxl_scp(scp, space, likelihood_threshold=50.0)
    assert lo.prevalence()["AF"] > hi.prevalence()["AF"]


def test_harmonised_labels_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        HarmonisedLabels(np.zeros((3, 2)), np.ones((3, 3), bool), ["a", "b"])


# -------------------------------------------------------------- preprocessing
def test_every_cohort_geometry_normalises_to_one_shape():
    cfg = PreprocessConfig()
    rng = np.random.default_rng(0)
    cases = {
        "code15 400Hz zero-padded": (np.pad(rng.normal(size=(12, 4000)), ((0, 0), (20, 76))).astype(np.float32), 400),
        "ptbxl 500Hz 10s": (rng.normal(size=(12, 5000)).astype(np.float32), 500),
        "cpsc 500Hz 60s": (rng.normal(size=(12, 30000)).astype(np.float32), 500),
        "cpsc 500Hz 6s": (rng.normal(size=(12, 3000)).astype(np.float32), 500),
    }
    for name, (sig, fs) in cases.items():
        out = preprocess_record(sig, fs, cfg)
        assert out.shape == (12, cfg.n_samples), name
        assert np.isfinite(out).all(), name


def test_zero_padding_is_stripped():
    x = np.zeros((12, 1000), np.float32)
    x[:, 200:800] = 1.0
    assert strip_zero_padding(x).shape[-1] == 600


def test_robust_scale_is_outlier_resistant():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(12, 1000)).astype(np.float32)
    spiked = x.copy()
    spiked[:, 0] = 500.0                       # one ectopic-like spike
    a, b = robust_scale(x), robust_scale(spiked)
    assert np.abs(a[:, 1:] - b[:, 1:]).max() < 0.05


def test_fixed_window_is_deterministic_when_centered():
    x = np.arange(1000, dtype=np.float32)[None, :].repeat(12, 0)
    assert np.array_equal(fixed_window(x, 500, "center"), fixed_window(x, 500, "center"))


def test_notch_is_off_by_default():
    """Mains frequency differs by country; removing it per cohort would erase a
    real site signature and flatter the transfer results."""
    assert PreprocessConfig().notch_hz is None


# ------------------------------------------------------------------ splitting
def test_patient_splits_never_leak():
    rng = np.random.default_rng(2)
    pid = rng.integers(0, 500, 3000)
    for splits in (patient_split(pid), stratified_patient_split(pid, (rng.random(3000) < 0.01).astype(int))):
        assert_no_leakage(pid, splits)
        assert sum(len(s) for s in splits) == len(pid)


def test_leakage_detector_actually_fires():
    pid = np.arange(100) // 2
    with pytest.raises(AssertionError, match="leakage"):
        assert_no_leakage(pid, [np.arange(60), np.arange(50, 100)])


def test_stratified_split_preserves_rare_positives():
    rng = np.random.default_rng(3)
    pid = rng.integers(0, 2000, 8000)
    y = (rng.random(8000) < 0.004).astype(int)
    splits = stratified_patient_split(pid, y, (0.7, 0.1, 0.2), seed=0)
    assert all(y[s].sum() > 0 for s in splits)


def test_calibration_split_is_patient_disjoint():
    rng = np.random.default_rng(4)
    pid = rng.integers(0, 400, 2000)
    cal, ev = calibration_split(2000, 200, patient_ids=pid)
    assert not (set(pid[cal]) & set(pid[ev]))


# ------------------------------------------------------------------ federated
def test_federated_export_is_small_and_audited():
    from ecgcal.data.korea import audit_export, export_sufficient_statistics
    from recalib_kit.metrics import bin_edges

    rng = np.random.default_rng(5)
    s = rng.beta(1, 20, 4000)
    y = (rng.random(4000) < s).astype(float)
    edges = bin_edges(rng.beta(1, 8, 40000), 15, "equal_mass")     # source edges, inbound
    stats = export_sufficient_statistics(s, y, edges, "AF", "model-v1")
    assert len(stats.to_json()) < 5000            # a whole cohort in a few KB
    assert sum(stats.count) == 4000

    tiny = export_sufficient_statistics(s[:30], y[:30], edges, "AF", "model-v1")
    assert not audit_export(tiny)["safe"]         # disclosure threshold must bite
