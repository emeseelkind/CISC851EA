"""Pooling operators: reduce a set of per-patch cancer probabilities within a
region to a single core-level (or batch-level) probability.

This is one of the four search axes in the region-wise weak supervision design
space (genome.py): which pooling operator turns "many weakly-labeled patches"
into "one core-level training signal" for the BCE loss. mean-pooling is the
ProstNFound+ fixed point's implicit choice (broadcast label to every pixel,
which is equivalent -- for BCE gradient direction, not value -- to a uniform
mean over the region); max/top-k/generalized-mean/LSE are standard MIL
alternatives.

All ops are differentiable and operate on probabilities in [0, 1], not logits,
so region weights and pooled scores stay directly comparable to
protoprostate.metrics's max_density / mean_density convention.
"""

from __future__ import annotations

import torch

POOLING_CHOICES = ["mean", "max", "topk_mean", "generalized_mean", "lse"]


def pool_probs(probs: torch.Tensor, mode: str, param: float) -> torch.Tensor:
    """probs: 1-D tensor of per-patch probabilities in [0, 1] for one region of
    one core (or concatenated across cores for a batch-level PU pool). Returns
    a 0-D scalar tensor. `param` meaning depends on `mode`:
        mean, max               ignored
        topk_mean               fraction of patches kept, in (0, 1]
        generalized_mean        power p > 0 (p=1 -> mean, p->inf -> max)
        lse                     temperature tau > 0 (tau->0 -> max, tau->inf -> mean)
    """
    if probs.numel() == 0:
        return probs.new_tensor(0.0)
    eps = 1e-6
    if mode == "mean":
        return probs.mean()
    if mode == "max":
        return probs.max()
    if mode == "topk_mean":
        # `param` is searched on a range shared with generalized_mean/lse
        # (genome.BOUNDS["pooling_param"] is a single field), so clamp the
        # fraction to its only valid range here rather than constraining the
        # search space itself.
        fraction = min(max(param, eps), 1.0)
        k = min(probs.numel(), max(1, int(round(fraction * probs.numel()))))
        return torch.topk(probs, k).values.mean()
    if mode == "generalized_mean":
        p = max(param, eps)
        return (probs.clamp(min=eps).pow(p).mean()).pow(1.0 / p)
    if mode == "lse":
        tau = max(param, eps)
        # softmax-weighted average -- a smooth max as tau -> 0, a mean as tau -> inf
        weights = torch.softmax(probs / tau, dim=0)
        return (weights * probs).sum()
    raise ValueError(f"unknown pooling mode: {mode!r}")
