"""The only trainable part of a genome evaluation: a linear probe on top of
the frozen, cached ProstNFound+ patch embeddings.

Keeping this to a single linear layer (no fine-tuning of the encoder) is what
makes it feasible to train-and-evaluate one genome per NSGA-II individual,
every generation, on a single GPU: with embeddings cached (prostea/region_cache.py),
one genome's full inner-fold training is a few epochs of logistic regression
over ~D=256-dim vectors, not a forward+backward pass through a ViT.
"""

from __future__ import annotations

import torch
from torch import nn


class LinearProbe(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.proj = nn.Conv2d(in_dim, 1, kernel_size=1)

    def forward(self, feats: torch.Tensor, age=None, psa=None) -> torch.Tensor:
        """feats: (Gh, Gw, D) -> logits (Gh, Gw), or a batch
        (B, Gh, Gw, D) -> (B, Gh, Gw) in one conv call (every core shares the
        same 64x64 grid, so batching here is what keeps a genome's training
        loop from paying one Python/kernel-dispatch overhead per core).

        age/psa are accepted (and ignored) only so call sites can use the
        same `model(feats, age=..., psa=...)` signature regardless of
        whether the genome's detector is this probe or pnf_detector.py's
        PNFDetector, which does use them.
        """
        single = feats.dim() == 3
        x = feats.unsqueeze(0) if single else feats
        x = x.permute(0, 3, 1, 2)  # (B, D, Gh, Gw)
        logits = self.proj(x)[:, 0]  # (B, Gh, Gw)
        return logits[0] if single else logits
