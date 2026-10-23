"""One genome's fitness = train a fresh LinearProbe on a train split, score a
val split. This is the function NSGA-II calls once per individual per
generation (prostea/search.py); everything here operates on already-cached
embeddings (prostea/region_cache.py), so it never touches the frozen encoder.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from prostea.genome import Genome
from prostea.loss import GenomeLoss
from prostea.metrics import core_metrics, localization_consistency, out_of_prostate_activation
from prostea.model import LinearProbe
from prostea.region_cache import RegionCache

ACTIVATION_THRESHOLD = 0.5


def score_cores(model: LinearProbe, cores: list[dict], threshold: float = ACTIVATION_THRESHOLD) -> pd.DataFrame:
    rows = []
    with torch.no_grad():
        feats_batch = torch.stack([c["feats"] for c in cores])
        probs_batch = torch.sigmoid(model(feats_batch))
        for core, probs in zip(cores, probs_batch):
            weak_mask = core["needle_mask"] & core["prostate_mask"]
            negative_mask = ~core["prostate_mask"]

            weak_probs = probs[weak_mask]
            neg_probs = probs[negative_mask]

            rows.append(dict(
                core_id=core["core_id"],
                patient_id=core["patient_id"],
                center=core["center"],
                grade_group=core["grade_group"],
                involvement=core["involvement"],
                max_density=float(weak_probs.max()) if weak_probs.numel() else float("nan"),
                mean_density=float(weak_probs.mean()) if weak_probs.numel() else float("nan"),
                needle_activated_fraction=(
                    float((weak_probs > threshold).float().mean()) if weak_probs.numel() else float("nan")
                ),
                negative_mean_prob=float(neg_probs.mean()) if neg_probs.numel() else float("nan"),
            ))
    return pd.DataFrame(rows)


def train_probe(
    genome: Genome,
    cache: RegionCache,
    train_idx: np.ndarray,
    device: str = "cpu",
    epochs: int = 8,
    lr: float = 1e-2,
    batch_size: int = 16,
    weight_decay: float = 1e-4,
    seed: int = 0,
) -> LinearProbe:
    rng = np.random.default_rng(seed)
    # `seed` previously only drove the numpy batch-order RNG above; the
    # LinearProbe's own init below drew from torch's uncontrolled global RNG,
    # so two "identically seeded" runs could still diverge (observed as the
    # h1_fixed vs h1_v2 discrepancy -- same genome/split, different results).
    torch.manual_seed(seed)
    d = cache.load(int(train_idx[0]), device=device)["feats"].shape[-1]
    model = LinearProbe(d).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = GenomeLoss(genome)

    order = np.array(train_idx)
    for _ in range(epochs):
        rng.shuffle(order)
        for start in range(0, len(order), batch_size):
            batch_rows = order[start : start + batch_size]
            cores = cache.load_many(batch_rows, device=device)
            opt.zero_grad()
            loss, _ = loss_fn(cores, model, device)
            loss.backward()
            opt.step()
    return model


def evaluate_genome(
    genome: Genome,
    cache: RegionCache,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    device: str = "cpu",
    epochs: int = 8,
    seed: int = 0,
) -> dict:
    """Trains on train_idx, scores val_idx. Returns a flat metrics dict:
    auc_cancer_max/mean, auc_cspca_max (H1's target metric), involvement_mae/
    corr, localization_consistency, out_of_prostate_activation, n.
    """
    model = train_probe(genome, cache, train_idx, device=device, epochs=epochs, seed=seed)
    val_cores = cache.load_many(val_idx, device=device)
    df = score_cores(model, val_cores)

    # A core's needle trace can fall entirely outside its own (imperfectly
    # aligned) prostate mask, leaving the weak region (needle & prostate)
    # empty -> max_density/mean_density are NaN for that core. core_metrics'
    # roc_auc_score can't handle NaN scores, so drop those cores from the
    # classification metrics rather than letting them crash the whole
    # evaluation; out_of_prostate_activation doesn't depend on the weak
    # region so it's computed on the full val set.
    n_empty_weak_region = int(df.max_density.isna().sum())
    scored_df = df[df.max_density.notna()]

    metrics = core_metrics(scored_df)
    metrics.update(localization_consistency(scored_df))
    metrics["out_of_prostate_activation"] = out_of_prostate_activation(df)
    metrics["n_empty_weak_region"] = n_empty_weak_region
    return metrics


def evaluate_genome_nested(
    genome: Genome,
    cache: RegionCache,
    inner_folds: list[tuple[np.ndarray, np.ndarray]],
    device: str = "cpu",
    epochs: int = 8,
) -> dict:
    """Fitness for the NSGA-II search: mean metrics over the inner folds of
    one outer-train set (see prostea/splits.py::nested_patient_folds). This
    is what shields the search from overfitting a single train/val split.
    """
    per_fold = [
        evaluate_genome(genome, cache, tr, va, device=device, epochs=epochs, seed=i)
        for i, (tr, va) in enumerate(inner_folds)
    ]
    # core_metrics() only adds involvement_mae/involvement_corr when a fold's
    # val split has >1 core with known involvement, so per-fold key sets can
    # differ -- union them rather than indexing off per_fold[0].
    all_keys = set().union(*(m.keys() for m in per_fold))
    return {
        k: float(np.nanmean([m.get(k, float("nan")) for m in per_fold]))
        for k in all_keys
    }
