"""Linear probing on a frozen encoder - the study's default adaptation protocol.

Why not gradient descent on a linear head
-----------------------------------------
A linear probe over a frozen backbone is a convex problem, and fitting it with
a handful of SGD steps solves it badly.  During development the SGD probe
reached macro-AUROC 0.43 on a representation whose frozen embeddings a properly
converged logistic regression separated at 0.68-0.98.  Nothing was wrong with
the encoder; the optimiser had simply not converged.

That failure mode is dangerous for *this* study specifically.  Its endpoint is
calibration, and an undertrained head is badly calibrated for reasons that have
nothing to do with distribution shift.  Reporting it would mean measuring
optimisation failure and calling it a calibration cliff.  So the probe is fitted
to convergence, and :meth:`LinearProbe.fit` records the convergence status of
every per-label solve.

Standardisation uses **source statistics only** and is then applied unchanged at
every target site.  Re-standardising per site would be unsupervised domain
adaptation smuggled in through the back door: it would silently remove part of
the very shift being measured.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .resnet1d import ECGEncoder

__all__ = ["LinearProbe", "embed_signals", "fit_linear_probe"]


@torch.no_grad()
def embed_signals(model: ECGEncoder, signals: np.ndarray, batch_size: int = 256) -> np.ndarray:
    """Frozen-encoder embeddings for every record, in input order."""
    model.eval()
    out = []
    for i in range(0, len(signals), batch_size):
        xb = torch.from_numpy(np.ascontiguousarray(np.asarray(signals[i : i + batch_size], np.float32)))
        out.append(model.backbone(xb).numpy())
    return np.concatenate(out, 0) if out else np.zeros((0, model.backbone.out_dim), np.float32)


@dataclass
class LinearProbe:
    """A per-label L2 logistic probe over frozen embeddings.

    Holds the source-domain standardisation so that applying the probe at a new
    site is a pure forward pass - which is what "ship the model unchanged"
    means operationally.
    """

    label_names: list[str]
    mean_: np.ndarray | None = None
    scale_: np.ndarray | None = None
    models_: list | None = None
    converged_: list[bool] | None = None
    C: float = 1.0

    def fit(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        mask: np.ndarray | None = None,
        max_iter: int = 5000,
    ) -> "LinearProbe":
        from sklearn.exceptions import ConvergenceWarning
        from sklearn.linear_model import LogisticRegression
        import warnings

        Z = np.asarray(embeddings, float)
        Y = np.asarray(labels)
        M = np.ones_like(Y, bool) if mask is None else np.asarray(mask, bool)

        self.mean_ = Z.mean(0)
        self.scale_ = np.maximum(Z.std(0), 1e-6)
        Zs = (Z - self.mean_) / self.scale_

        self.models_, self.converged_ = [], []
        for j, name in enumerate(self.label_names):
            sel = M[:, j]
            yj = Y[sel, j]
            if sel.sum() < 20 or yj.sum() == 0 or yj.sum() == sel.sum():
                # Not estimable. Store the base rate rather than a fitted model;
                # a constant predictor is at least honestly calibrated to it.
                self.models_.append(float(yj.mean()) if yj.size else 0.0)
                self.converged_.append(False)
                continue
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always", ConvergenceWarning)
                lr = LogisticRegression(C=self.C, max_iter=max_iter, solver="lbfgs")
                lr.fit(Zs[sel], yj)
                self.models_.append(lr)
                self.converged_.append(not any(issubclass(x.category, ConvergenceWarning) for x in w))
        return self

    def _standardise(self, embeddings: np.ndarray) -> np.ndarray:
        if self.mean_ is None:
            raise RuntimeError("probe is not fitted")
        return (np.asarray(embeddings, float) - self.mean_) / self.scale_

    def predict_proba(self, embeddings: np.ndarray) -> np.ndarray:
        Zs = self._standardise(embeddings)
        out = np.zeros((len(Zs), len(self.label_names)), float)
        for j, m in enumerate(self.models_ or []):
            out[:, j] = float(m) if isinstance(m, float) else m.predict_proba(Zs)[:, 1]
        return out

    def predict_from_signals(self, model: ECGEncoder, signals: np.ndarray) -> np.ndarray:
        return self.predict_proba(embed_signals(model, signals))

    def report(self) -> dict:
        return {
            "n_labels": len(self.label_names),
            "converged": dict(zip(self.label_names, self.converged_ or [])),
            "all_converged": bool(all(self.converged_)) if self.converged_ else False,
            "embedding_dim": int(len(self.mean_)) if self.mean_ is not None else None,
        }

    def save(self, path: Path) -> None:
        import pickle

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(self, fh)

    @staticmethod
    def load(path: Path) -> "LinearProbe":
        import pickle

        with open(path, "rb") as fh:
            return pickle.load(fh)


def fit_linear_probe(
    model: ECGEncoder,
    signals: np.ndarray,
    labels: np.ndarray,
    label_names: list[str],
    mask: np.ndarray | None = None,
    C: float = 1.0,
) -> LinearProbe:
    """Embed with a frozen encoder, then fit the probe to convergence."""
    return LinearProbe(label_names=list(label_names), C=C).fit(
        embed_signals(model, signals), labels, mask
    )
