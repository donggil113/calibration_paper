"""Training and inference for the ECG encoder.

Three properties the experiments depend on:

* **Masked loss.**  A label a cohort does not annotate contributes nothing to
  the gradient.  Treating it as a negative would teach the model that the label
  is rare in that cohort, which is a prevalence artefact and would land in the
  label-shift component the study is trying to measure.
* **No target-domain leakage.**  Normalisation statistics, early stopping and
  model selection all use source data only.  Touching the target - even through
  a BatchNorm running mean - is unsupervised domain adaptation, and this study's
  premise is a model shipped unchanged.
* **Deterministic.**  Seeds are set for Python, NumPy and Torch, and the
  resulting scores are what every calibration number downstream is computed
  from, so a rerun has to reproduce them exactly.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .resnet1d import ECGEncoder, build_model

__all__ = ["TrainConfig", "ECGArrayDataset", "masked_bce", "train_model", "predict_scores",
           "evaluate_multilabel", "set_seed", "save_checkpoint", "load_checkpoint"]


def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)     # 1-D convs have no deterministic CPU kernel
    torch.backends.cudnn.deterministic = True


@dataclass
class TrainConfig:
    epochs: int = 12
    batch_size: int = 64
    lr: float = 3e-3
    weight_decay: float = 1e-4
    warmup_frac: float = 0.1
    size: str = "small"
    dropout: float = 0.1
    seed: int = 0
    num_workers: int = 0
    augment: bool = True
    patience: int = 4
    min_epochs: int = 3
    amp: bool = False
    max_grad_norm: float = 5.0

    def to_dict(self) -> dict:
        return asdict(self)


class ECGArrayDataset(Dataset):
    """Wrap arrays already in memory (or memory-mapped) for the loader.

    Augmentation is deliberately limited to transformations a real recording
    could undergo - a time shift, a small amplitude gain, additive noise.  Lead
    dropout and time reversal are *not* used: reversal destroys the P-QRS-T
    ordering that defines every label here, and would teach the encoder to
    ignore the one thing it must not.
    """

    def __init__(
        self,
        signals: np.ndarray,
        labels: np.ndarray,
        mask: np.ndarray | None = None,
        augment: bool = False,
        seed: int = 0,
    ):
        self.x = signals
        self.y = np.asarray(labels, np.float32)
        self.m = np.ones_like(self.y, bool) if mask is None else np.asarray(mask, bool)
        self.augment = augment
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, i: int):
        x = np.asarray(self.x[i], np.float32)
        if self.augment:
            shift = int(self.rng.integers(-x.shape[-1] // 20, x.shape[-1] // 20 + 1))
            if shift:
                x = np.roll(x, shift, axis=-1)
            x = x * float(self.rng.normal(1.0, 0.05))
            x = x + self.rng.normal(0, 0.01, x.shape).astype(np.float32)
        return torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(self.y[i]), torch.from_numpy(self.m[i])


def masked_bce(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    pos_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Binary cross-entropy averaged over observed entries only.

    ``pos_weight`` up-weights positives per label.  It is applied with care:
    reweighting the loss changes the model's implied prior and therefore its
    calibration, so any run that uses it records the weights, and the headline
    calibration results are produced with ``pos_weight=None`` so that the source
    model is calibrated to the source prior by construction - the starting point
    Theorem 1 assumes.
    """
    loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none", pos_weight=pos_weight)
    m = mask.float()
    denom = m.sum().clamp_min(1.0)
    return (loss * m).sum() / denom


@torch.no_grad()
def predict_scores(model: ECGEncoder, signals: np.ndarray, batch_size: int = 256) -> np.ndarray:
    """Sigmoid scores for every record, in input order."""
    model.eval()
    out = []
    for i in range(0, len(signals), batch_size):
        xb = torch.from_numpy(np.ascontiguousarray(np.asarray(signals[i : i + batch_size], np.float32)))
        out.append(torch.sigmoid(model(xb)).numpy())
    return np.concatenate(out, 0) if out else np.zeros((0, model.n_labels), np.float32)


def evaluate_multilabel(scores: np.ndarray, labels: np.ndarray, mask: np.ndarray, names: list[str]) -> dict:
    """Per-label AUROC/ECE on observed entries."""
    from recalib_kit.metrics import calibration_report

    rows = {}
    for j, n in enumerate(names):
        sel = mask[:, j]
        if sel.sum() < 20 or labels[sel, j].sum() in (0, sel.sum()):
            rows[n] = {"auroc": float("nan"), "ece": float("nan"), "n": int(sel.sum())}
            continue
        r = calibration_report(scores[sel, j], labels[sel, j])
        rows[n] = {"auroc": r.auroc, "ece": r.ece, "ece2": r.ece2, "n": r.n, "prevalence": r.prevalence}
    aurocs = [v["auroc"] for v in rows.values() if np.isfinite(v["auroc"])]
    return {"per_label": rows, "macro_auroc": float(np.mean(aurocs)) if aurocs else float("nan")}


def train_model(
    train_signals: np.ndarray,
    train_labels: np.ndarray,
    train_mask: np.ndarray | None = None,
    val_signals: np.ndarray | None = None,
    val_labels: np.ndarray | None = None,
    val_mask: np.ndarray | None = None,
    label_names: list[str] | None = None,
    cfg: TrainConfig | None = None,
    model: ECGEncoder | None = None,
    verbose: bool = True,
) -> tuple[ECGEncoder, dict]:
    """Train with cosine decay, warmup and early stopping on validation macro-AUROC.

    Selection is on AUROC rather than on validation loss on purpose: the study's
    premise is that a model can be *well discriminating* and badly calibrated,
    so selecting on a loss that mixes the two would pre-empt the finding.
    """
    cfg = cfg or TrainConfig()
    set_seed(cfg.seed)
    n_labels = train_labels.shape[1]
    names = label_names or [f"label_{i}" for i in range(n_labels)]
    model = model or build_model(n_labels, cfg.size, dropout=cfg.dropout)

    ds = ECGArrayDataset(train_signals, train_labels, train_mask, augment=cfg.augment, seed=cfg.seed)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, drop_last=False)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps = max(1, cfg.epochs * len(dl))
    warm = max(1, int(cfg.warmup_frac * steps))

    def lr_at(step: int) -> float:
        if step < warm:
            return step / warm
        p = (step - warm) / max(1, steps - warm)
        return 0.5 * (1 + np.cos(np.pi * min(p, 1.0)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    history: list[dict] = []
    best = {"macro_auroc": -np.inf, "epoch": -1, "state": None}
    step = 0

    for ep in range(cfg.epochs):
        model.train()
        t0, tot, nb = time.time(), 0.0, 0
        for xb, yb, mb in dl:
            opt.zero_grad(set_to_none=True)
            loss = masked_bce(model(xb), yb, mb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm)
            opt.step()
            sched.step()
            tot += float(loss.item())
            nb += 1
            step += 1

        rec = {"epoch": ep, "train_loss": tot / max(nb, 1), "seconds": time.time() - t0,
               "lr": float(sched.get_last_lr()[0])}
        if val_signals is not None:
            vs = predict_scores(model, val_signals)
            vm = np.ones_like(val_labels, bool) if val_mask is None else val_mask
            ev = evaluate_multilabel(vs, val_labels, vm, names)
            rec["val_macro_auroc"] = ev["macro_auroc"]
            if ev["macro_auroc"] > best["macro_auroc"]:
                best = {"macro_auroc": ev["macro_auroc"], "epoch": ep,
                        "state": {k: v.clone() for k, v in model.state_dict().items()}}
        history.append(rec)
        if verbose:
            msg = f"  epoch {ep:2d}  loss {rec['train_loss']:.4f}  {rec['seconds']:.1f}s"
            if "val_macro_auroc" in rec:
                msg += f"  val_macro_auroc {rec['val_macro_auroc']:.4f}"
            print(msg)

        if (val_signals is not None and ep >= cfg.min_epochs
                and ep - best["epoch"] >= cfg.patience):
            if verbose:
                print(f"  early stop at epoch {ep} (best {best['epoch']}, {best['macro_auroc']:.4f})")
            break

    if best["state"] is not None:
        model.load_state_dict(best["state"])
    return model, {"history": history, "best_epoch": best["epoch"],
                   "best_val_macro_auroc": best["macro_auroc"] if np.isfinite(best["macro_auroc"]) else None,
                   "config": cfg.to_dict(), "label_names": names}


def save_checkpoint(model: ECGEncoder, path: Path, meta: dict | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "n_labels": model.n_labels, "meta": meta or {}}, path)
    (path.with_suffix(".json")).write_text(json.dumps(meta or {}, indent=2, default=str))


def load_checkpoint(path: Path, size: str = "small") -> tuple[ECGEncoder, dict]:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model(ck["n_labels"], size)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, ck.get("meta", {})
