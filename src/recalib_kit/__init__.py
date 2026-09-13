"""recalib-kit: measure, decompose and repair calibration under distribution shift.

The toolbox behind *The Calibration Cliff*.  Four layers, each usable alone:

``metrics``
    Calibration and discrimination estimators, with the bias of the binned
    estimator handled explicitly rather than assumed away.
``label_shift``
    Prior estimation from unlabeled target data (BBSE, MLLS), its relaxation
    under partial concept shift (anchor restriction, partial identification),
    and an unlabeled test for whether label shift suffices at a site.
``decomposition``
    Theorem 1: split target calibration error into the part a free unlabeled
    correction removes and the part that requires labels.
``sample_size`` / ``conformal``
    Theorems 2 and 3: how many labeled target cases a site needs, and
    prediction sets that stay valid when the concept has shifted too.

Typical use at a deploying site, in order of what it costs::

    from recalib_kit import decompose_unlabeled_budget, empirical_n_star

    # 1. Free: unlabeled ECGs only. Is a prior correction going to be enough?
    budget = decompose_unlabeled_budget(target_scores, pi_s, src_scores, src_labels)

    # 2. If not, price the label work before committing to it.
    n = empirical_n_star(target_scores, target_labels, eps=0.02, alpha=0.1)
"""

from .conformal import (
    coverage_report,
    hybrid_transfer_conformal,
    min_n_for_level,
    mondrian_conformal,
    split_conformal,
    weighted_conformal,
)
from .decomposition import Decomposition, decompose, decompose_l1, decompose_unlabeled_budget
from .label_shift import (
    LabelShiftEstimate,
    anchor_bbse,
    bbse,
    min_gamma,
    mlls,
    partial_identification,
    prior_correction,
    sensitivity_curve,
    test_label_shift_sufficiency,
)
from .metrics import (
    brier_decomposition,
    calibration_report,
    expected_calibration_error,
    multilabel_report,
    reliability_curve,
    squared_calibration_error,
)
from .recalibrate import METHODS, HybridPriorFewShot, PriorCorrection, fit_recalibrator
from .sample_size import empirical_n_star, n_star_curve, n_star_lower_bound, n_star_upper_plugin

__version__ = "0.1.0"

__all__ = [
    "expected_calibration_error", "squared_calibration_error", "calibration_report",
    "multilabel_report", "reliability_curve", "brier_decomposition",
    "bbse", "mlls", "anchor_bbse", "partial_identification", "sensitivity_curve",
    "min_gamma", "test_label_shift_sufficiency", "prior_correction", "LabelShiftEstimate",
    "decompose", "decompose_l1", "decompose_unlabeled_budget", "Decomposition",
    "fit_recalibrator", "METHODS", "PriorCorrection", "HybridPriorFewShot",
    "empirical_n_star", "n_star_curve", "n_star_upper_plugin", "n_star_lower_bound",
    "split_conformal", "weighted_conformal", "hybrid_transfer_conformal",
    "mondrian_conformal", "coverage_report", "min_n_for_level",
    "__version__",
]
