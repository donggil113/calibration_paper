"""Self-supervised pretraining: the 'foundation model' step.

The study's subject is a *foundation* model - one pretrained on pooled,
unlabeled, multi-country ECG and then adapted with a light head.  That detail
matters to the result rather than being background: a model pretrained across
countries has already seen the acquisition characteristics of every site, so
whatever calibration collapse remains cannot be dismissed as the encoder simply
never having seen a 50 Hz mains hum or a Brazilian cart.  Pretraining is the
control that closes off that explanation.

The task is masked-patch reconstruction: contiguous spans of the recording are
zeroed and the model predicts them from context.  Chosen over contrastive
pretraining for one reason specific to this study - contrastive objectives need
augmentations declared invariant, and the standard ECG set (amplitude scaling,
lead dropout) makes the model invariant to exactly the acquisition differences
that distinguish these sites.  That would erase the site signal by construction
and quietly assume the paper's conclusion.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .resnet1d import ECGEncoder, MultiLabelHead, ResNet1d, build_model

__all__ = ["SSLConfig", "MaskedPatchDataset", "pretrain_masked", "transfer_backbone"]


@dataclass
class SSLConfig:
    epochs: int = 8
    batch_size: int = 64
    lr: float = 2e-3
    weight_decay: float = 1e-4
    mask_ratio: float = 0.3
    span_samples: int = 50          # 0.2 s at 250 Hz: long enough to hide a whole beat segment
    size: str = "small"
    seed: int = 0
    num_workers: int = 0


class MaskedPatchDataset(Dataset):
    """Yield (masked input, original, mask) triples."""

    def __init__(self, signals: np.ndarray, cfg: SSLConfig):
        self.x = signals
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, i: int):
        x = np.asarray(self.x[i], np.float32)
        t = x.shape[-1]
        span = max(4, self.cfg.span_samples)
        n_span = max(1, int(self.cfg.mask_ratio * t / span))
        m = np.zeros(t, np.float32)
        for _ in range(n_span):
            s = int(self.rng.integers(0, max(1, t - span)))
            m[s : s + span] = 1.0
        xm = x * (1.0 - m)                       # masked spans set to zero
        return torch.from_numpy(xm), torch.from_numpy(x), torch.from_numpy(m)


class _SSLModel(torch.nn.Module):
    """Backbone plus a decoder that upsamples back to input resolution."""

    def __init__(self, backbone: ResNet1d, out_leads: int = 12):
        super().__init__()
        self.backbone = backbone
        self.proj = torch.nn.Conv1d(backbone.out_dim, out_leads, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.backbone(x, return_sequence=True)          # (B, C, T')
        y = self.proj(h)
        return F.interpolate(y, size=x.shape[-1], mode="linear", align_corners=False)


def pretrain_masked(
    signals: np.ndarray,
    cfg: SSLConfig | None = None,
    verbose: bool = True,
) -> tuple[ResNet1d, dict]:
    """Pretrain a backbone by reconstructing masked spans.

    ``signals`` should be the **pooled, unlabeled** union of every available
    site.  Loss is computed on masked positions only; including the visible
    ones lets the model score well by copying its input.
    """
    cfg = cfg or SSLConfig()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    probe = build_model(1, cfg.size)
    model = _SSLModel(probe.backbone, out_leads=int(signals.shape[1]))
    dl = DataLoader(
        MaskedPatchDataset(signals, cfg), batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, cfg.epochs * len(dl)))

    history = []
    for ep in range(cfg.epochs):
        model.train()
        tot, nb = 0.0, 0
        for xm, x0, m in dl:
            opt.zero_grad(set_to_none=True)
            pred = model(xm)
            mm = m.unsqueeze(1)                              # (B, 1, T) broadcast over leads
            loss = ((pred - x0) ** 2 * mm).sum() / mm.sum().clamp_min(1.0) / x0.shape[1]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            tot += float(loss.item())
            nb += 1
        history.append({"epoch": ep, "recon_mse": tot / max(nb, 1)})
        if verbose:
            print(f"  ssl epoch {ep:2d}  masked recon MSE {history[-1]['recon_mse']:.5f}")
    return model.backbone, {"history": history, "config": cfg.__dict__}


def transfer_backbone(backbone: ResNet1d, n_labels: int, freeze: bool = True) -> ECGEncoder:
    """Attach a fresh linear head to a pretrained backbone.

    ``freeze=True`` gives the linear-probe protocol the paper reports by
    default: the representation is fixed, so any cross-site difference in the
    results is a property of the representation and not of how much the head
    was allowed to re-fit.
    """
    enc = ECGEncoder.__new__(ECGEncoder)
    torch.nn.Module.__init__(enc)
    enc.backbone = backbone
    # The head is sized from the backbone rather than from a preset name, so a
    # backbone pretrained at one size can never be silently paired with a head
    # built for another.
    enc.head = MultiLabelHead(backbone.out_dim, n_labels)
    enc.n_labels = n_labels
    if freeze:
        enc.freeze_backbone()
    else:
        # Unfreeze explicitly. A backbone reused across calls carries whatever
        # requires_grad state the previous call left on it, so freeze=False has
        # to be an instruction, not merely the absence of one.
        for prm in enc.backbone.parameters():
            prm.requires_grad = True
        enc.backbone.train()
    return enc
