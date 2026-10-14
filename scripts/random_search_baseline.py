#!/usr/bin/env python
"""H2: does NSGA-II actually beat random search over the same genome space at
an equal evaluation budget (pop_size * n_gen fitness evaluations)? Samples
that many genomes uniformly, evaluates each the same way run_search.py's
NSGA-II does (mean over inner folds), and reports the non-dominated set plus
the held-out-test comparison for the best-csPCa genome found.

Usage matches scripts/run_search.py -- run both with the same --outer_fold,
--pop_size and --n_gen for a like-for-like H2 comparison.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from prostea.fitness import evaluate_genome, evaluate_genome_nested
from prostea.genome import decode, sample_random_genome
from prostea.region_cache import RegionCache
from prostea.search import non_dominated_front
from prostea.splits import nested_patient_folds


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache_dir", required=True)
    p.add_argument("--exp_dir", required=True)
    p.add_argument("--outer_fold", type=int, default=0)
    p.add_argument("--n_outer", type=int, default=5)
    p.add_argument("--n_inner", type=int, default=4)
    p.add_argument("--pop_size", type=int, default=16)
    p.add_argument("--n_gen", type=int, default=10)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--max_oop_activation", type=float, default=0.15)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=1)
    args = p.parse_args()

    os.makedirs(args.exp_dir, exist_ok=True)
    cache = RegionCache(args.cache_dir)
    nested = nested_patient_folds(cache.meta, n_outer=args.n_outer, n_inner=args.n_inner)
    fold = nested[args.outer_fold]

    n_evals = args.pop_size * args.n_gen
    print(f"[random search] outer fold {args.outer_fold}: {n_evals} genome evaluations "
          f"(equal budget to run_search.py --pop_size {args.pop_size} --n_gen {args.n_gen})", flush=True)

    rng = np.random.default_rng(args.seed)
    records = []
    for i in range(n_evals):
        genome = sample_random_genome(rng)
        metrics = evaluate_genome_nested(genome, cache, fold["inner"], device=args.device, epochs=args.epochs)
        feasible = not np.isnan(metrics["out_of_prostate_activation"]) and \
            metrics["out_of_prostate_activation"] <= args.max_oop_activation
        records.append(dict(
            genome=genome.as_dict(),
            auc_cspca_max=0.5 if np.isnan(metrics["auc_cspca_max"]) else metrics["auc_cspca_max"],
            localization_consistency=0.0 if np.isnan(metrics["localization_consistency"]) else metrics["localization_consistency"],
            feasible=feasible,
        ))
        if (i + 1) % max(1, n_evals // 10) == 0:
            print(f"      {i+1}/{n_evals}", flush=True)

    feasible_records = [r for r in records if r["feasible"]] or records
    pareto = non_dominated_front(feasible_records)
    pareto.sort(key=lambda r: r["auc_cspca_max"], reverse=True)
    with open(os.path.join(args.exp_dir, "pareto_front.json"), "w") as f:
        json.dump(pareto, f, indent=2)
    print(f"[random search] {len(pareto)} non-dominated genomes -> {args.exp_dir}/pareto_front.json", flush=True)

    if not pareto:
        return
    best_genome = decode(pareto[0]["genome"])
    metrics = evaluate_genome(
        best_genome, cache, fold["outer_train"], fold["outer_test"],
        device=args.device, epochs=args.epochs, seed=0,
    )
    held_out = {"random_search_best": dict(genome=best_genome.as_dict(), metrics=metrics)}
    with open(os.path.join(args.exp_dir, "held_out_test.json"), "w") as f:
        json.dump(held_out, f, indent=2)
    print(f"[held-out test] random_search_best: auc_cspca_max={metrics.get('auc_cspca_max'):.3f}", flush=True)


if __name__ == "__main__":
    main()
