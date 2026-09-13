"""1-D residual encoder for 12-lead ECG, plus the heads used in this study.

Sized for the claim being made, not for a leaderboard.  The paper's finding is
a *dissociation* - discrimination survives cross-national transfer while
calibration does not - and that claim is only interesting if discrimination is
genuinely good to begin with.  It does not require a state-of-the-art model,
and using one would make the experiments unrunnable on the hardware most
readers (and this repository's CI) have.  The architecture below reaches the
AUROC range reported for published ECG models on the shared label set while
training on CPU in minutes.

Two details are load-bearing rather than stylistic:

* **No per-lead normalisation inside the network.**  Amplitude calibration
  differs between carts, and normalising it away inside the model would hide a
  real component of site shift that a deployed model would face.
* **Global *average* pooling, not max.**  Max pooling over time makes the
  logit scale depend on record length, and CPSC-2018 records vary from 6 to 60
  seconds.  That would create a length-driven calibration artefact confounded
  with the cohort effect this study measures.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["ResNet1d", "ECGEncoder", "MultiLabelHead", "MaskedReconstructionHead", "build_model"]


class _Block(nn.Module):
    """Pre-activation residual block with dropout."""

    def __init__(self, cin: int, cout: int, stride: int = 1, kernel: int = 7, dropout: float = 0.1):
        super().__init__()
        pad = kernel // 2
        self.bn1 = nn.BatchNorm1d(cin)
        self.conv1 = nn.Conv1d(cin, cout, kernel, stride=stride, padding=pad, bias=False)
        self.bn2 = nn.BatchNorm1d(cout)
        self.drop = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(cout, cout, kernel, padding=pad, bias=False)
        self.short = (
            nn.Conv1d(cin, cout, 1, stride=stride, bias=False)
            if (stride != 1 or cin != cout)
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.bn1(x))
        s = self.short(h)
        h = self.conv1(h)
        h = self.drop(F.relu(self.bn2(h)))
        return self.conv2(h) + s


class ResNet1d(nn.Module):
    """Stack of residual blocks over time, returning a pooled embedding."""

    def __init__(
        self,
        in_leads: int = 12,
        widths: tuple[int, ...] = (32, 64, 96, 128),
        blocks_per_stage: int = 2,
        kernel: int = 7,
        stride: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.stem = nn.Conv1d(in_leads, widths[0], kernel_size=15, stride=2, padding=7, bias=False)
        stages: list[nn.Module] = []
        cin = widths[0]
        for w in widths:
            for b in range(blocks_per_stage):
                stages.append(_Block(cin, w, stride=stride if b == 0 else 1, kernel=kernel, dropout=dropout))
                cin = w
        self.stages = nn.Sequential(*stages)
        self.norm = nn.BatchNorm1d(cin)
        self.out_dim = cin

    def forward(self, x: torch.Tensor, return_sequence: bool = False) -> torch.Tensor:
        h = self.stages(self.stem(x))
        h = F.relu(self.norm(h))
        if return_sequence:
            return h
        return h.mean(dim=-1)          # length-invariant


class MultiLabelHead(nn.Module):
    """Linear head producing one logit per label.

    Kept strictly linear so a *linear probe* on a frozen encoder is the default
    evaluation.  If the head were an MLP, a transfer failure could always be
    blamed on head capacity rather than on the representation, and the paper's
    claim is about the representation.
    """

    def __init__(self, in_dim: int, n_labels: int, dropout: float = 0.0):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(in_dim, n_labels)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.fc(self.drop(z))


class MaskedReconstructionHead(nn.Module):
    """Decoder for masked-patch self-supervision."""

    def __init__(self, in_dim: int, out_leads: int = 12, patch: int = 25):
        super().__init__()
        self.patch = patch
        self.fc = nn.Sequential(
            nn.Linear(in_dim, in_dim * 2), nn.GELU(), nn.Linear(in_dim * 2, out_leads * patch)
        )
        self.out_leads = out_leads

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: (B, C, T') -> per-position patch reconstruction
        b, _, t = h.shape
        y = self.fc(h.transpose(1, 2))                       # (B, T', leads*patch)
        return y.reshape(b, t, self.out_leads, self.patch).permute(0, 2, 1, 3)


class ECGEncoder(nn.Module):
    """Encoder plus an optional multi-label head; the object the study freezes."""

    def __init__(
        self,
        n_labels: int,
        in_leads: int = 12,
        widths: tuple[int, ...] = (32, 64, 96, 128),
        blocks_per_stage: int = 2,
        dropout: float = 0.1,
        head_dropout: float = 0.0,
    ):
        super().__init__()
        self.backbone = ResNet1d(in_leads, widths, blocks_per_stage, dropout=dropout)
        self.head = MultiLabelHead(self.backbone.out_dim, n_labels, head_dropout)
        self.n_labels = n_labels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def freeze_backbone(self) -> "ECGEncoder":
        """Freeze the representation so only the linear probe trains.

        Also puts the backbone in eval mode permanently: leaving BatchNorm in
        training mode would let its running statistics adapt to the target
        site, which is a silent form of unsupervised domain adaptation and
        would confound the very transfer measurement being made.
        """
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()
        return self

    def train(self, mode: bool = True):            # keep a frozen backbone in eval
        super().train(mode)
        if not any(p.requires_grad for p in self.backbone.parameters()):
            self.backbone.eval()
        return self

    def n_parameters(self) -> dict:
        tot = sum(p.numel() for p in self.parameters())
        tr = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": tot, "trainable": tr, "embedding_dim": self.backbone.out_dim}


def build_model(n_labels: int, size: str = "small", **kw) -> ECGEncoder:
    """Named size presets, so an experiment config names a model rather than shapes."""
    presets = {
        "tiny":   dict(widths=(16, 32, 48), blocks_per_stage=1),
        "small":  dict(widths=(32, 64, 96, 128), blocks_per_stage=2),
        "medium": dict(widths=(48, 96, 144, 192, 256), blocks_per_stage=2),
    }
    if size not in presets:
        raise KeyError(f"unknown size {size!r}; choose from {sorted(presets)}")
    return ECGEncoder(n_labels=n_labels, **{**presets[size], **kw})
