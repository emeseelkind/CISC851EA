"""The region-wise weak-supervision design space.

Every micro-US core is split by needle + prostate masks into three regions
(see prostea/region_cache.py for how the masks are cached at patch-grid
resolution):

    weak        needle trace (intersected with the prostate mask) -- has a
                core-level pathology label (Gleason grade group + involvement)
    unlabeled   prostate tissue outside the needle trace -- no label, but not
                known-benign either (positive-unlabeled)
    negative    outside the prostate entirely -- known-negative

A Genome is one point in the search space that determines, from these three
regions, a trainable loss (prostea/loss.py). NSGA-II (prostea/search.py)
searches this space directly; ProstNFound+'s published loss and standard
top-k MIL are both single fixed points in it (constants below).
"""

from __future__ import annotations

from dataclasses import dataclass

from prostea.pooling import POOLING_CHOICES

GG_TARGET_MODES = [
    "binary_any_cancer",  # target = 1[grade_group > 0]        -- any cancer (isPCa or csPCa)
    "binary_cspca",       # target = 1[grade_group >= 3]        -- true clinically-significant threshold
    "involvement_scaled", # target = involvement (in [0, 1])    -- ProstNFound+'s own heatmap target
    "ordinal_linear",     # target = grade_group / 5
]

# Search bounds, used both to build the pymoo mixed-variable Problem
# (prostea/search.py) and to clip/validate a decoded Genome.
BOUNDS = dict(
    pooling_param=(0.05, 8.0),
    w_weak=(0.0, 3.0),
    w_negative=(0.0, 3.0),
    w_unlabeled=(0.0, 3.0),
    pu_prior=(0.01, 0.5),
    pos_weight=(1.0, 4.0),
)


@dataclass(frozen=True)
class Genome:
    pooling: str = "mean"
    pooling_param: float = 1.0
    gg_target_mode: str = "binary_any_cancer"
    w_weak: float = 1.0
    w_negative: float = 0.0
    w_unlabeled: float = 0.0
    pu_prior: float = 0.1
    pos_weight: float = 1.0

    def __post_init__(self):
        if self.pooling not in POOLING_CHOICES:
            raise ValueError(f"unknown pooling: {self.pooling!r}")
        if self.gg_target_mode not in GG_TARGET_MODES:
            raise ValueError(f"unknown gg_target_mode: {self.gg_target_mode!r}")

    def as_dict(self) -> dict:
        return dict(
            pooling=self.pooling,
            pooling_param=self.pooling_param,
            gg_target_mode=self.gg_target_mode,
            w_weak=self.w_weak,
            w_negative=self.w_negative,
            w_unlabeled=self.w_unlabeled,
            pu_prior=self.pu_prior,
            pos_weight=self.pos_weight,
        )


def sample_random_genome(rng) -> Genome:
    """Uniform sample over the same space NSGA-II searches, for the random-
    search baseline (H2: does evolution beat random search at equal budget?).
    `rng`: a numpy.random.Generator.
    """
    return Genome(
        pooling=rng.choice(POOLING_CHOICES),
        pooling_param=float(rng.uniform(*BOUNDS["pooling_param"])),
        gg_target_mode=rng.choice(GG_TARGET_MODES),
        w_weak=float(rng.uniform(*BOUNDS["w_weak"])),
        w_negative=float(rng.uniform(*BOUNDS["w_negative"])),
        w_unlabeled=float(rng.uniform(*BOUNDS["w_unlabeled"])),
        pu_prior=float(rng.uniform(*BOUNDS["pu_prior"])),
        pos_weight=float(rng.uniform(*BOUNDS["pos_weight"])),
    )


def decode(x: dict) -> Genome:
    """x: the dict pymoo hands `_evaluate` for a mixed-variable Problem (one
    individual), with the same keys as prostea.search.problem_vars().
    """
    return Genome(
        pooling=x["pooling"],
        pooling_param=float(x["pooling_param"]),
        gg_target_mode=x["gg_target_mode"],
        w_weak=float(x["w_weak"]),
        w_negative=float(x["w_negative"]),
        w_unlabeled=float(x["w_unlabeled"]),
        pu_prior=float(x["pu_prior"]),
        pos_weight=float(x["pos_weight"]),
    )


# --- Named fixed points, for the controlled comparison in the proposal ---

# The single point in this space that *approximates* ProstNFound+'s actual,
# shipped loss.
#
# CORRECTED (was wrong -- built from ../prostNfound/prostNfound/losses.py +
# conf/main_training.json, which is the public, pre-"+" ProstNFound repo's
# demo config, not what actually produced
# /datasets/exactvu_pca/checkpoint_store/pnf_plus_final.pth). The real
# training code is in the (unpublished, lab-internal) `medAI` package --
# see /home/emese/.../medprojects/proside-dino-pca/medAI/losses/prostnfound.py
# -- and the checkpoint's own saved `args`:
#   loss="mil_prop_bce_entropy_reg" -> CancerDetectionMILLoss(
#       entropy_penalty_lambda=0.01, pos_weight=1.0 [args.pos_weight key is
#       absent from the saved args, i.e. LossArgs' default, NOT the 2.0 in
#       main_training.json], treat_gg1_as_benign=False)
#   + add_image_clf=True, image_clf_mode="cspca" -> a SEPARATE
#     ImageLevelClassificationLoss on PNF+'s class-decoder head, summed in
#     with weight 1.0 (medAI.losses.prostnfound.build_loss / SumLoss).
# CancerDetectionMILLoss's core term is: pool = mean(sigmoid(logits) over the
# needle & prostate region); BCE(pool, involvement) [a *continuous* target,
# not a binary cancer/no-cancer label] with pos_weight applied whenever
# involvement > 0. That pooling-then-BCE-against-involvement is exactly
# `gg_target_mode="involvement_scaled"` + `pooling="mean"` here.
#
# Two things this genome space still CANNOT represent, so this fixed point
# stays an approximation even now (flag this in any write-up):
#   1. CancerDetectionMILLoss's entropy_penalty_lambda=0.01 term -- no field
#      in Genome for a per-pixel entropy regularizer.
#   2. The separate ImageLevelClassificationLoss on PNF+'s class-decoder head
#      (a second, scalar-output decoder entirely) -- prostea's model.py has
#      only one (heatmap) head, so there's no way to add a second loss term
#      against a second head's output. Note medAI's ImageLevelClassificationLoss
#      has its own bug: mode="cspca" doesn't match its `if self.mode == "pca"`
#      branch, so it silently falls through to `grade_group > 1` (i.e. "any
#      cancer", same off-by-one as the old "binary_cspca" mode here) instead
#      of true csPCa (grade_group >= 3) -- so even a from-scratch
#      reimplementation of this term should NOT assume mode="cspca" means
#      what its name says.
PROSTNFOUND_PLUS = Genome(
    pooling="mean",
    pooling_param=1.0,
    gg_target_mode="involvement_scaled",
    w_weak=1.0,
    w_negative=0.0,
    w_unlabeled=0.0,
    pu_prior=0.1,
    pos_weight=1.0,
)

# Standard top-k MIL (prostNfound/prostNfound/losses.py::simple_mil_loss's
# default top_percentile=0.2), still with no out-of-prostate / PU terms.
TOPK_MIL = Genome(
    pooling="topk_mean",
    pooling_param=0.2,
    gg_target_mode="binary_any_cancer",
    w_weak=1.0,
    w_negative=0.0,
    w_unlabeled=0.0,
    pu_prior=0.1,
    pos_weight=1.0,
)
