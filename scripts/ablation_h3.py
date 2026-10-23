#!/usr/bin/env python
"""H3: "the out-of-prostate and PU/unlabeled-region terms improve the
localization-consistency proxy without hurting AUROC" (README). Ablate by
zeroing w_negative / w_unlabeled on each outer fold's evolved_best genome
(from a run_search.py H1 run) and re-evaluate on that fold's held-out test
split -- the same outer_train/outer_test indices run_search.py used, so this
is directly comparable to that fold's held_out_test.json entry.

Usage:
    python scripts/ablation_h3.py --cache_dir cache/regions \
        --h1_dir experiments/h1_fixed --out experiments/h1_fixed/h3_ablation.csv
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from prostea.fitness import evaluate_genome
from prostea.genome import Genome
from prostea.region_cache import RegionCache
from prostea.splits import nested_patient_folds

VARIANTS = {
    "full": {},
    "zero_negative": {"w_negative": 0.0},
    "zero_unlabeled": {"w_unlabeled": 0.0},
    "zero_both": {"w_negative": 0.0, "w_unlabeled": 0.0},
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache_dir", default="cache/regions")
    p.add_argument("--h1_dir", required=True, help="dir with fold{i}/held_out_test.json")
    p.add_argument("--out", required=True)
    p.add_argument("--n_outer", type=int, default=5)
    p.add_argument("--n_inner", type=int, default=4)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--device", default="cpu")
    p.add_argument("--detector", choices=["pnf", "linear"], default="pnf",
                    help="see scripts/run_search.py --detector; should match whichever "
                         "detector built the h1_dir genomes being ablated here.")
    p.add_argument("--lr", type=float, default=None,
                    help="see scripts/run_search.py --lr; defaults to 1e-3 for pnf, 1e-2 for linear.")
    args = p.parse_args()

    model_builder = None
    if args.detector == "pnf":
        from prostea.pnf_detector import build_pnf_detector
        model_builder = build_pnf_detector
    lr = args.lr if args.lr is not None else (1e-3 if args.detector == "pnf" else 1e-2)

    cache = RegionCache(args.cache_dir)
    nested = nested_patient_folds(cache.meta, n_outer=args.n_outer, n_inner=args.n_inner)

    rows = []
    for fold_idx, fold in enumerate(nested):
        held_out_path = os.path.join(args.h1_dir, f"fold{fold_idx}", "held_out_test.json")
        if not os.path.exists(held_out_path):
            print(f"[ablation] fold {fold_idx}: no {held_out_path}, skipping")
            continue
        with open(held_out_path) as f:
            held_out = json.load(f)
        base_genome = Genome(**held_out["evolved_best"]["genome"])

        for variant_name, overrides in VARIANTS.items():
            genome = dataclasses.replace(base_genome, **overrides)
            metrics = evaluate_genome(
                genome, cache, fold["outer_train"], fold["outer_test"],
                device=args.device, epochs=args.epochs, seed=0, model_builder=model_builder, lr=lr,
            )
            metrics.update(fold=fold_idx, variant=variant_name,
                            w_negative=genome.w_negative, w_unlabeled=genome.w_unlabeled)
            rows.append(metrics)
            print(f"[ablation] fold {fold_idx} {variant_name}: "
                  f"auc_cspca_max={metrics.get('auc_cspca_max'):.3f} "
                  f"localization_consistency={metrics.get('localization_consistency')} "
                  f"out_of_prostate_activation={metrics.get('out_of_prostate_activation'):.3f}",
                  flush=True)

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"wrote {args.out}")

    print("\n=== summary (mean +/- std across folds) ===")
    summary = df.groupby("variant")[
        ["auc_cspca_max", "localization_consistency", "out_of_prostate_activation"]
    ].agg(["mean", "std"])
    print(summary)


if __name__ == "__main__":
    main()
