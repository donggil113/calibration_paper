"""A multi-site 12-lead ECG simulator with controllable label and concept shift.

Why a simulator is part of the method, not a stand-in for data
--------------------------------------------------------------
The three theorems are statements about *estimators*: that a decomposition is
identified, that a label count suffices, that a coverage guarantee survives a
bounded perturbation.  Claims of that kind cannot be checked on real cohorts,
because on real data the ground-truth split between label shift and concept
shift is exactly what is unknown.  A generator with a known
:math:`(\\pi_t, \\eta)` is the only instrument that can falsify them.  It is
used here for that, and for powering the study design; the empirical claims
about ECG foundation models are reserved for the real cohorts.

How the signal is built
-----------------------
Beats are generated as a three-dimensional cardiac dipole - a vectorcardiogram -
and projected onto the 12 standard leads through an inverse-Dower matrix.  This
is deliberate rather than decorative: it means a conduction abnormality is
introduced once, as a change in the depolarisation loop, and then appears with
physiologically consistent polarity and amplitude across all twelve leads.
Generating each lead independently would let a network solve the task from a
per-lead artefact that no real ECG contains, and every transfer result would
then be measuring the artefact.

Site effects are separated into three independently controllable layers, matching
the three things that actually differ between the cohorts in this study:

1. **Prevalence** (``prior``) - the label shift.  Defaults are set near the
   reported prevalences of the real cohorts, so a screening site really is an
   order of magnitude rarer than an emergency department.
2. **Morphology** (``concept``) - the concept shift.  Changes how a *given*
   diagnosis presents: amplitude, duration, axis.  This is the component no
   unlabeled method can correct.
3. **Acquisition** (``acquisition``) - cart and environment: noise level,
   baseline wander, mains frequency (50 Hz in Germany and China, 60 Hz in the
   Americas and Korea), lead-placement jitter.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

__all__ = ["LEADS", "SiteConfig", "SITE_LIBRARY", "simulate_site", "simulate_cohort_set",
           "vcg_to_12lead", "vcg_to_independent", "derive_augmented", "LABELS"]

LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
LABELS = ["AF", "IAVB", "RBBB", "LBBB", "SB", "STACH"]

# Inverse-Dower transform: 8 independent leads from the (X, Y, Z) dipole.
# I, II and the six precordials are independent; III and the augmented leads
# follow from Einthoven's and Goldberger's relations, applied below.
_DOWER = np.array(
    [
        [0.632, -0.235, 0.059],    # I
        [0.235,  1.066, -0.132],   # II
        [-0.397, -0.298, -0.639],  # V1
        [-0.130, -0.414, -0.917],  # V2
        [0.005, -0.415, -0.719],   # V3
        [0.311, -0.319, -0.373],   # V4
        [0.596, -0.198, -0.132],   # V5
        [0.630, -0.044,  0.049],   # V6
    ]
)


def vcg_to_independent(vcg: np.ndarray) -> np.ndarray:
    """Project a (3, T) dipole onto the 8 *independently measured* leads.

    Order: ``I, II, V1..V6``.  A 12-lead cart measures only these; the
    remaining four are arithmetic.
    """
    return (_DOWER @ np.asarray(vcg, float)).astype(np.float32)


def derive_augmented(independent: np.ndarray) -> np.ndarray:
    """Expand 8 independent leads to the canonical 12.

    III, aVR, aVL and aVF are *derived* by the recording cart from I and II,
    never measured separately, so Einthoven's law
    (:math:`\mathrm{II} = \mathrm{I} + \mathrm{III}`) and Goldberger's
    relations hold exactly in real data - noise included, because the noise on
    lead III is by construction the noise on II minus that on I.

    Deriving them here rather than projecting them from the dipole is what keeps
    that true after the acquisition layer is applied.  Adding independent noise
    to a derived lead would break a constraint every real ECG satisfies and hand
    a network a way to separate signal from noise that it would not have at
    deployment - quietly inflating every transfer number in the study.
    """
    ind = np.asarray(independent, float)
    if ind.shape[0] != 8:
        raise ValueError(f"expected 8 independent leads (I, II, V1..V6), got {ind.shape[0]}")
    i, ii = ind[0], ind[1]
    iii = ii - i
    avr = -(i + ii) / 2.0
    avl = i - ii / 2.0
    avf = ii - i / 2.0
    return np.vstack([i, ii, iii, avr, avl, avf, ind[2:]]).astype(np.float32)


def vcg_to_12lead(vcg: np.ndarray) -> np.ndarray:
    """Project a (3, T) dipole onto the 12 standard leads in canonical order."""
    return derive_augmented(vcg_to_independent(vcg))


def _gauss(t: np.ndarray, centre: float, width: float, amp: float) -> np.ndarray:
    return amp * np.exp(-0.5 * ((t - centre) / max(width, 1e-4)) ** 2)


@dataclass
class SiteConfig:
    """Everything that distinguishes one simulated site from another."""

    name: str
    country: str
    prior: dict[str, float]                        # label prevalences
    concept: dict[str, float] = field(default_factory=dict)
    acquisition: dict[str, float] = field(default_factory=dict)
    hr_mean: float = 72.0
    hr_sd: float = 12.0
    age_mean: float = 62.0
    age_sd: float = 16.0
    frac_female: float = 0.5

    def with_concept(self, **kw) -> "SiteConfig":
        return replace(self, concept={**self.concept, **kw})

    def with_prior(self, **kw) -> "SiteConfig":
        return replace(self, prior={**self.prior, **kw})


# Prevalences chosen near published figures for the corresponding real cohorts:
# an academic emergency department, a primary-care telehealth network, a mixed
# European inpatient set, Chinese general hospitals, and - the decisive
# contrast - an asymptomatic screening population an order of magnitude rarer.
SITE_LIBRARY: dict[str, SiteConfig] = {
    "mimic_like": SiteConfig(
        "mimic_like", "US",
        prior={"AF": 0.120, "IAVB": 0.075, "RBBB": 0.070, "LBBB": 0.030, "SB": 0.095, "STACH": 0.130},
        acquisition={"noise": 0.030, "wander": 0.040, "mains_hz": 60.0, "mains_amp": 0.010},
        hr_mean=80.0, age_mean=64.0,
    ),
    "code15_like": SiteConfig(
        "code15_like", "BR",
        prior={"AF": 0.018, "IAVB": 0.019, "RBBB": 0.026, "LBBB": 0.011, "SB": 0.042, "STACH": 0.028},
        concept={"amp_scale": 0.92, "qrs_width": 1.04, "axis_deg": -6.0},
        acquisition={"noise": 0.045, "wander": 0.060, "mains_hz": 60.0, "mains_amp": 0.014},
        hr_mean=73.0, age_mean=52.0, frac_female=0.60,
    ),
    "ptbxl_like": SiteConfig(
        "ptbxl_like", "DE",
        prior={"AF": 0.085, "IAVB": 0.040, "RBBB": 0.055, "LBBB": 0.025, "SB": 0.030, "STACH": 0.040},
        concept={"amp_scale": 1.06, "qrs_width": 0.98, "axis_deg": 4.0},
        acquisition={"noise": 0.025, "wander": 0.035, "mains_hz": 50.0, "mains_amp": 0.012},
        hr_mean=70.0, age_mean=60.0,
    ),
    "chapman_like": SiteConfig(
        "chapman_like", "CN",
        prior={"AF": 0.070, "IAVB": 0.060, "RBBB": 0.080, "LBBB": 0.015, "SB": 0.180, "STACH": 0.075},
        concept={"amp_scale": 1.12, "qrs_width": 0.95, "axis_deg": 8.0, "t_amp": 1.10},
        acquisition={"noise": 0.035, "wander": 0.045, "mains_hz": 50.0, "mains_amp": 0.016},
        hr_mean=74.0, age_mean=58.0,
    ),
    "georgia_like": SiteConfig(
        "georgia_like", "US",
        prior={"AF": 0.095, "IAVB": 0.065, "RBBB": 0.062, "LBBB": 0.028, "SB": 0.085, "STACH": 0.100},
        acquisition={"noise": 0.032, "wander": 0.042, "mains_hz": 60.0, "mains_amp": 0.011},
        hr_mean=78.0, age_mean=62.0,
    ),
    # The screening cohort: same signal-generating process, an order of
    # magnitude lower prevalence, younger and healthier.  This is the site that
    # separates the two shift components.
    "korea_like": SiteConfig(
        "korea_like", "KR",
        prior={"AF": 0.006, "IAVB": 0.012, "RBBB": 0.021, "LBBB": 0.002, "SB": 0.110, "STACH": 0.012},
        concept={"amp_scale": 1.03, "qrs_width": 0.97, "axis_deg": 3.0},
        acquisition={"noise": 0.022, "wander": 0.030, "mains_hz": 60.0, "mains_amp": 0.009},
        hr_mean=67.0, hr_sd=9.0, age_mean=42.0, age_sd=9.0, frac_female=0.42,
    ),
}


def _beat(t: np.ndarray, labels: dict[str, int], cfg: SiteConfig, rng: np.random.Generator) -> np.ndarray:
    """One (3, T) cardiac dipole beat, shaped by the record's diagnoses."""
    c = cfg.concept
    amp = c.get("amp_scale", 1.0)
    qw = c.get("qrs_width", 1.0)
    t_amp = c.get("t_amp", 1.0)

    pr = 0.16 + (0.09 if labels.get("IAVB") else 0.0) + rng.normal(0, 0.012)
    qrs_w = 0.032 * qw * (1.0 + 0.85 * (labels.get("RBBB", 0) or labels.get("LBBB", 0))) 
    qrs_w *= (1.0 + rng.normal(0, 0.06))

    xyz = np.zeros((3, t.size))
    # --- P wave: absent in atrial fibrillation, replaced by fibrillatory noise
    if labels.get("AF"):
        f = rng.uniform(5.0, 9.0)
        xyz[0] += 0.035 * amp * np.sin(2 * np.pi * f * t + rng.uniform(0, 6.3)) * (t < pr)
        xyz[1] += 0.025 * amp * np.sin(2 * np.pi * f * 1.3 * t + rng.uniform(0, 6.3)) * (t < pr)
    else:
        p_c = pr - 0.06
        xyz[0] += _gauss(t, p_c, 0.020, 0.09 * amp)
        xyz[1] += _gauss(t, p_c, 0.020, 0.13 * amp)
        xyz[2] += _gauss(t, p_c, 0.022, -0.03 * amp)

    # --- QRS: a Q-R-S triplet in each dipole axis.  Bundle branch blocks widen
    # it and rotate the terminal forces, which is what produces the
    # characteristic V1/V6 patterns after projection.
    r_c = pr
    xyz[0] += _gauss(t, r_c - 0.018 * qw, 0.009 * qw, -0.12 * amp)
    xyz[0] += _gauss(t, r_c, qrs_w, 1.05 * amp)
    xyz[0] += _gauss(t, r_c + 0.026 * qw, 0.011 * qw, -0.22 * amp)
    xyz[1] += _gauss(t, r_c, qrs_w * 1.05, 0.80 * amp)
    xyz[1] += _gauss(t, r_c + 0.030 * qw, 0.012 * qw, -0.18 * amp)
    xyz[2] += _gauss(t, r_c, qrs_w * 0.95, -0.55 * amp)

    if labels.get("RBBB"):      # terminal rightward/anterior forces -> rSR' in V1
        xyz[2] += _gauss(t, r_c + 0.052 * qw, 0.016 * qw, 0.60 * amp)
        xyz[0] += _gauss(t, r_c + 0.055 * qw, 0.018 * qw, -0.26 * amp)
    if labels.get("LBBB"):      # delayed, leftward, monophasic
        xyz[0] += _gauss(t, r_c + 0.048 * qw, 0.022 * qw, 0.55 * amp)
        xyz[2] += _gauss(t, r_c + 0.050 * qw, 0.020 * qw, -0.40 * amp)

    # --- T wave, discordant with the QRS when conduction is abnormal
    disc = -0.55 if (labels.get("LBBB") or labels.get("RBBB")) else 1.0
    t_c = pr + 0.24
    xyz[0] += _gauss(t, t_c, 0.055, 0.26 * amp * t_amp * disc)
    xyz[1] += _gauss(t, t_c, 0.055, 0.30 * amp * t_amp * disc)
    xyz[2] += _gauss(t, t_c, 0.058, -0.10 * amp * t_amp * disc)

    # --- frontal-plane axis rotation, a real inter-population difference
    ang = np.deg2rad(c.get("axis_deg", 0.0) + rng.normal(0, 6.0))
    rot = np.array([[np.cos(ang), -np.sin(ang), 0], [np.sin(ang), np.cos(ang), 0], [0, 0, 1]])
    return rot @ xyz


def _draw_labels(cfg: SiteConfig, rng: np.random.Generator) -> dict[str, int]:
    """Sample a diagnosis set, respecting the exclusions real ECGs obey."""
    lab = {k: int(rng.random() < cfg.prior.get(k, 0.0)) for k in LABELS}
    if lab["SB"] and lab["STACH"]:                      # cannot be both
        drop = "SB" if rng.random() < 0.5 else "STACH"
        lab[drop] = 0
    if lab["AF"]:                                        # AF has no sinus rate label
        lab["SB"] = lab["STACH"] = 0
    if lab["LBBB"] and lab["RBBB"]:                      # complete bilateral block is not a resting ECG
        lab["RBBB"] = 0
    return lab


def simulate_site(
    cfg: SiteConfig,
    n: int,
    fs: int = 250,
    seconds: float = 10.0,
    seed: int = 0,
    n_patients: int | None = None,
) -> dict:
    """Generate ``n`` records for one site.

    Returns signals ``(n, 12, fs*seconds)``, a label matrix, and metadata.
    ``n_patients`` below ``n`` produces repeat recordings per patient, so the
    patient-level splitting machinery is exercised on data that actually needs it.
    """
    rng = np.random.default_rng(seed)
    T = int(round(fs * seconds))
    t_axis = np.arange(T) / fs
    acq = cfg.acquisition
    sigs = np.zeros((n, 12, T), np.float32)
    Y = np.zeros((n, len(LABELS)), np.int8)
    hrs, ages, sexes = np.zeros(n), np.zeros(n), []

    n_pat = n_patients or n
    pid = rng.integers(0, n_pat, n)

    for i in range(n):
        lab = _draw_labels(cfg, rng)
        Y[i] = [lab[k] for k in LABELS]

        hr = rng.normal(cfg.hr_mean, cfg.hr_sd)
        if lab["SB"]:
            hr = rng.uniform(38, 58)
        if lab["STACH"]:
            hr = rng.uniform(101, 150)
        if lab["AF"]:
            hr = rng.normal(max(cfg.hr_mean + 12, 85), 22)
        hr = float(np.clip(hr, 32, 190))
        hrs[i] = hr

        rr = 60.0 / hr
        vcg = np.zeros((3, T))
        beat_t = np.arange(0, min(rr, 1.6), 1.0 / fs)
        onset = rng.uniform(0, rr)
        while onset < seconds:
            # atrial fibrillation is defined by an irregularly irregular rhythm
            this_rr = rr * (1 + rng.uniform(-0.35, 0.35)) if lab["AF"] else rr * (1 + rng.normal(0, 0.02))
            b = _beat(beat_t, lab, cfg, rng)
            s0 = int(round(onset * fs))
            e0 = min(s0 + b.shape[1], T)
            if e0 > s0:
                vcg[:, s0:e0] += b[:, : e0 - s0]
            onset += max(this_rr, 0.25)

        # The acquisition layer acts on the 8 measured leads only; the derived
        # leads inherit their noise arithmetically, exactly as a real cart does.
        ind = vcg_to_independent(vcg).astype(float)
        ind = ind + acq.get("noise", 0.03) * rng.normal(size=ind.shape)
        for f_w, a_w in ((0.15, 1.0), (0.33, 0.6)):
            ind = ind + acq.get("wander", 0.04) * a_w * np.sin(
                2 * np.pi * f_w * t_axis + rng.uniform(0, 6.3)
            )
        if acq.get("mains_amp", 0) > 0:
            ind = ind + acq["mains_amp"] * np.sin(
                2 * np.pi * acq.get("mains_hz", 50.0) * t_axis + rng.uniform(0, 6.3)
            )
        # electrode-siting jitter: a per-electrode gain on the measured leads
        ind = ind * (1.0 + 0.05 * rng.normal(size=(8, 1)))
        sigs[i] = derive_augmented(ind)

        ages[i] = np.clip(rng.normal(cfg.age_mean, cfg.age_sd), 18, 95)
        sexes.append("F" if rng.random() < cfg.frac_female else "M")

    return {
        "signals": sigs,
        "labels": Y,
        "label_names": list(LABELS),
        "site": cfg.name,
        "country": cfg.country,
        "meta": {
            "patient_id": np.array([f"{cfg.name}_p{p}" for p in pid]),
            "record": np.array([f"{cfg.name}_r{i}" for i in range(n)]),
            "heart_rate": hrs,
            "age": ages,
            "sex": np.array(sexes),
        },
        "config": cfg,
    }


def simulate_cohort_set(
    sites: dict[str, int],
    fs: int = 250,
    seconds: float = 10.0,
    seed: int = 0,
    n_patients_frac: float = 0.85,
) -> dict[str, dict]:
    """Generate several sites at once, each with its own seed stream."""
    out = {}
    for k, (name, n) in enumerate(sites.items()):
        cfg = SITE_LIBRARY[name] if isinstance(name, str) else name
        out[cfg.name] = simulate_site(
            cfg, n, fs, seconds, seed=seed + 1000 * k,
            n_patients=max(1, int(n * n_patients_frac)),
        )
    return out
