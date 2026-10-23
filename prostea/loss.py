"""Builds a trainable loss from a Genome (prostea/genome.py) over the three
regions of a core (prostea/region_cache.py): weak (needle ∩ prostate),
unlabeled (prostate \\ needle), negative (outside prostate).

    weak term       pool patch probs in the weak region with the genome's
                     pooling operator -> one core-level probability -> BCE
                     against the genome's GG target encoding, weighted
                     `pos_weight` on the positive class.
    negative term    same pooling, over the negative region, BCE against 0
                     (known-negative -- this is what lets a genome be
                     penalized for activating tissue outside the prostate,
                     the proposal's "out-of-prostate activation" concern).
    unlabeled term   non-negative PU (nnPU, Kiryo et al. 2017) risk, patch-
                     level, with class prior `pu_prior`: P = weak-region
                     patches of cores the genome's target calls positive,
                     U = unlabeled-region patches of every core in the batch.
                     A BCE-with-logits surrogate stands in for nnPU's usual
                     symmetric zero-one surrogate (standard in practice; see
                     README for the caveat this implies for nnPU's
                     non-negativity guarantee).

total = w_weak * weak + w_negative * negative + w_unlabeled * pu
"""

from __future__ import annotations

import torch
from torch.nn import functional as F

from prostea.genome import Genome
from prostea.pooling import pool_probs

_EPS = 1e-6


def gg_target(genome: Genome, grade_group: int, involvement: float) -> float:
    mode = genome.gg_target_mode
    if mode == "binary_any_cancer":
        return 1.0 if grade_group > 0 else 0.0
    if mode == "binary_cspca":
        return 1.0 if grade_group >= 3 else 0.0
    if mode == "involvement_scaled":
        return float(involvement)
    if mode == "ordinal_linear":
        return grade_group / 5.0
    raise ValueError(f"unknown gg_target_mode: {mode!r}")


def _weighted_bce(prob: torch.Tensor, target: float, pos_weight: float) -> torch.Tensor:
    p = prob.clamp(_EPS, 1 - _EPS)
    w = pos_weight if target > 0.5 else 1.0
    return -w * (target * torch.log(p) + (1 - target) * torch.log(1 - p))


class GenomeLoss:
    """Not an nn.Module: it consumes a *list* of per-core dicts (variable
    patch counts per region rule out a batched tensor), matching how
    prostNfound/prostNfound/losses.py already loops per-core internally.
    """

    def __init__(self, genome: Genome):
        self.genome = genome

    def __call__(self, cores: list[dict], model, device: str) -> tuple[torch.Tensor, dict]:
        g = self.genome
        weak_losses, negative_losses = [], []
        pu_pos_logits, pu_neg_logits = [], []

        # One conv call for the whole batch (every core shares the 64x64
        # grid) instead of one Python-level model(...) call per core.
        feats_batch = torch.stack([c["feats"] for c in cores])  # (B, Gh, Gw, D)
        age_batch = torch.tensor([c["age"] for c in cores], dtype=torch.float32, device=feats_batch.device)
        psa_batch = torch.tensor([c["psa"] for c in cores], dtype=torch.float32, device=feats_batch.device)
        logits_batch = model(feats_batch, age=age_batch, psa=psa_batch)  # (B, Gh, Gw)
        probs_batch = torch.sigmoid(logits_batch)

        for core, logits, probs in zip(cores, logits_batch, probs_batch):
            needle_mask = core["needle_mask"]
            prostate_mask = core["prostate_mask"]

            weak_mask = needle_mask & prostate_mask
            unlabeled_mask = prostate_mask & ~needle_mask
            negative_mask = ~prostate_mask

            target = gg_target(g, core["grade_group"], core["involvement"])

            if weak_mask.any():
                score = pool_probs(probs[weak_mask], g.pooling, g.pooling_param)
                weak_losses.append(_weighted_bce(score, target, g.pos_weight))
                if target > 0.5:
                    pu_pos_logits.append(logits[weak_mask])

            if negative_mask.any() and g.w_negative > 0:
                score = pool_probs(probs[negative_mask], g.pooling, g.pooling_param)
                negative_losses.append(_weighted_bce(score, 0.0, 1.0))

            if unlabeled_mask.any() and g.w_unlabeled > 0:
                pu_neg_logits.append(logits[unlabeled_mask])

        weak = torch.stack(weak_losses).mean() if weak_losses else logits.new_tensor(0.0)
        negative = (
            torch.stack(negative_losses).mean() if negative_losses else logits.new_tensor(0.0)
        )
        pu = _pu_risk(pu_pos_logits, pu_neg_logits, g.pu_prior, logits)

        total = g.w_weak * weak + g.w_negative * negative + g.w_unlabeled * pu
        aux = dict(weak=float(weak.detach()), negative=float(negative.detach()), pu=float(pu.detach()))
        return total, aux


def _pu_risk(pos_logits_list, neg_logits_list, prior: float, ref: torch.Tensor) -> torch.Tensor:
    """nnPU risk (Kiryo et al., 2017), BCE-with-logits surrogate.

    R = prior * E_P[l(f(x), +1)] + max(0, E_U[l(f(x), -1)] - prior * E_P[l(f(x), -1)])
    """
    if not pos_logits_list or not neg_logits_list:
        return ref.new_tensor(0.0)
    pos = torch.cat(pos_logits_list)
    unl = torch.cat(neg_logits_list)
    r_p_plus = F.binary_cross_entropy_with_logits(pos, torch.ones_like(pos))
    r_p_minus = F.binary_cross_entropy_with_logits(pos, torch.zeros_like(pos))
    r_u_minus = F.binary_cross_entropy_with_logits(unl, torch.zeros_like(unl))
    negative_risk = r_u_minus - prior * r_p_minus
    return prior * r_p_plus + torch.clamp(negative_risk, min=0.0)
