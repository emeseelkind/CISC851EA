"""Synthetic-data smoke test: no GPU, no medAI, no real OPTIMUM data needed.
Builds a fake region cache on disk (mirrors what scripts/cache_regions.py
would produce) and exercises pooling, the genome loss, the linear probe, the
nested-CV fitness function, and NSGA-II search end to end.

Run with: pytest tests/test_smoke.py -q
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest
import torch

from prostea.fitness import evaluate_genome, evaluate_genome_nested
from prostea.genome import PROSTNFOUND_PLUS, TOPK_MIL, sample_random_genome
from prostea.pooling import pool_probs
from prostea.region_cache import RegionCache
from prostea.splits import nested_patient_folds

GRID = 8
DIM = 16


def test_pooling_operators_agree_at_extremes():
    probs = torch.tensor([0.1, 0.9, 0.5, 0.3])
    assert pool_probs(probs, "max", 1.0).item() == pytest.approx(0.9)
    assert pool_probs(probs, "mean", 1.0).item() == pytest.approx(probs.mean().item())
    assert pool_probs(probs, "topk_mean", 1.0).item() == pytest.approx(probs.mean().item())
    assert pool_probs(probs, "topk_mean", 0.25).item() == pytest.approx(0.9)


def _make_synthetic_cache(tmp_path, n_cores=24, n_patients=8, seed=0):
    rng = np.random.default_rng(seed)
    out_dir = tmp_path / "cache"
    npz_dir = out_dir / "npz"
    npz_dir.mkdir(parents=True)

    rows = []
    for i in range(n_cores):
        core_id = f"SIM-{i:03d}"
        patient_id = f"P-{i % n_patients:02d}"
        grade_group = int(rng.integers(0, 6)) if rng.random() < 0.5 else 0
        involvement = float(rng.uniform(0.05, 1.0)) if grade_group > 0 else 0.0

        prostate_mask = np.zeros((GRID, GRID), dtype=np.uint8)
        prostate_mask[1:7, 1:7] = 1
        needle_mask = np.zeros((GRID, GRID), dtype=np.uint8)
        if i == 0:
            # Real-data edge case (hit on the actual OPTIMUM cache): the
            # needle trace can fall entirely outside its own core's
            # (imperfectly aligned) prostate mask, leaving needle&prostate
            # empty. See fitness.py::evaluate_genome's n_empty_weak_region.
            needle_mask[0, 0] = 1
        else:
            needle_mask[2:5, 2:5] = 1

        feats = rng.normal(size=(GRID, GRID, DIM)).astype(np.float32)
        if grade_group > 0:
            # make needle-region features separable from the rest, so a
            # linear probe actually has signal to learn on synthetic data.
            feats[2:5, 2:5, 0] += 3.0

        np.savez_compressed(
            npz_dir / f"{core_id}.npz",
            feats=feats.astype(np.float16),
            needle_mask=needle_mask,
            prostate_mask=prostate_mask,
        )
        rows.append(dict(
            core_id=core_id, patient_id=patient_id, center="SIM",
            grade_group=grade_group, involvement=involvement,
            pri_mus=float("nan"), age=60.0, psa=5.0,
            n_prostate_patches=int(prostate_mask.sum()), n_needle_patches=int(needle_mask.sum()),
            has_prostate_mask=True,
        ))
    pd.DataFrame(rows).to_csv(out_dir / "metadata.csv", index=False)
    return str(out_dir)


def test_end_to_end_fitness_on_synthetic_cache(tmp_path):
    cache_dir = _make_synthetic_cache(tmp_path)
    cache = RegionCache(cache_dir)
    assert len(cache) == 24

    nested = nested_patient_folds(cache.meta, n_outer=2, n_inner=2)
    fold = nested[0]

    for genome in (PROSTNFOUND_PLUS, TOPK_MIL, sample_random_genome(np.random.default_rng(1))):
        metrics = evaluate_genome_nested(genome, cache, fold["inner"], device="cpu", epochs=2)
        assert "auc_cspca_max" in metrics
        assert "localization_consistency" in metrics
        assert "out_of_prostate_activation" in metrics

    final = evaluate_genome(
        PROSTNFOUND_PLUS, cache, fold["outer_train"], fold["outer_test"], device="cpu", epochs=2,
    )
    assert np.isfinite(final["n"])
    # core 0 (in whichever split it lands in) has an empty weak region by
    # construction -- must be dropped from AUROC, not crash roc_auc_score.
    assert "n_empty_weak_region" in final


def test_nsga2_search_runs_on_synthetic_cache(tmp_path):
    pytest.importorskip("pymoo")
    from prostea.search import pareto_front_to_records, run_search

    cache_dir = _make_synthetic_cache(tmp_path, n_cores=16, n_patients=6)
    cache = RegionCache(cache_dir)
    nested = nested_patient_folds(cache.meta, n_outer=2, n_inner=2)
    fold = nested[0]

    res = run_search(
        cache, fold["inner"], pop_size=4, n_gen=2, device="cpu", epochs=1, verbose=False,
        # A single epoch from random init won't reliably suppress out-of-
        # prostate activation, so loosen the constraint to exercise the
        # search/decode plumbing rather than genuine constraint satisfaction.
        max_oop_activation=1.0,
    )
    records = pareto_front_to_records(res)
    assert len(records) >= 1
    assert all("genome" in r and "auc_cspca_max" in r for r in records)


def test_pareto_front_to_records_handles_no_feasible_individual():
    from types import SimpleNamespace

    from prostea.search import pareto_front_to_records

    assert pareto_front_to_records(SimpleNamespace(X=None, F=None)) == []
