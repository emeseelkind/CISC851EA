#!/usr/bin/env python
"""Aggregate per-fold held_out_test.json files from scripts/run_search.py
(experiments/h1/fold{0..n_outer-1}/) and scripts/random_search_baseline.py
(experiments/h2/fold{0..n_outer-1}/) into:

  H1 -- per-fold + mean+/-std table of evolved_best vs prostnfound_plus vs
        topk_mil on the held-out outer-test cores of each fold (does the
        evolved objective beat ProstNFound+'s loss by more than fold-to-fold
        variance?).
  H2 -- per-fold + mean+/-std table comparing the NSGA-II-evolved genome
        (evolved_best, from experiments/h1) against the random-search genome
        found at equal budget (random_search_best, from experiments/h2), plus
        each fold's Pareto-front size as a cheap secondary signal.

Usage:
    python scripts/aggregate_cv.py --h1_dir experiments/h1 --h2_dir experiments/h2 \
        --n_outer 5 --out_dir experiments
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

METRIC_KEYS = [
    "auc_cspca_max", "auc_cancer_max", "auc_cancer_mean",
    "involvement_mae", "involvement_corr",
    "localization_consistency", "out_of_prostate_activation",
    "n", "n_localization", "n_empty_weak_region",
]


def load_fold_candidates(exp_dir: str, fold: int) -> dict | None:
    path = os.path.join(exp_dir, f"fold{fold}", "held_out_test.json")
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return json.load(f)


def rows_for(exp_dir: str, n_outer: int) -> pd.DataFrame:
    rows = []
    for fold in range(n_outer):
        data = load_fold_candidates(exp_dir, fold)
        if data is None:
            continue
        for candidate, payload in data.items():
            row = {"fold": fold, "candidate": candidate}
            row.update({k: payload["metrics"].get(k, float("nan")) for k in METRIC_KEYS})
            rows.append(row)
    return pd.DataFrame(rows)


def pareto_size_for(exp_dir: str, n_outer: int) -> pd.DataFrame:
    rows = []
    for fold in range(n_outer):
        path = os.path.join(exp_dir, f"fold{fold}", "pareto_front.json")
        if not os.path.isfile(path):
            continue
        with open(path) as f:
            pareto = json.load(f)
        rows.append({
            "fold": fold,
            "n_pareto": len(pareto),
            "best_auc_cspca_max": max((r["auc_cspca_max"] for r in pareto), default=float("nan")),
            "best_localization_consistency": max(
                (r["localization_consistency"] for r in pareto), default=float("nan")
            ),
        })
    return pd.DataFrame(rows)


def mean_std_table(df: pd.DataFrame) -> pd.DataFrame:
    agg = df.groupby("candidate")[[k for k in METRIC_KEYS if k in df.columns]].agg(["mean", "std", "count"])
    agg.columns = [f"{metric}_{stat}" for metric, stat in agg.columns]
    return agg.reset_index()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--h1_dir", default="experiments/h1")
    p.add_argument("--h2_dir", default="experiments/h2")
    p.add_argument("--n_outer", type=int, default=5)
    p.add_argument("--out_dir", default="experiments")
    args = p.parse_args()

    # --- H1: evolved_best vs prostnfound_plus vs topk_mil, 5-fold ---
    h1_df = rows_for(args.h1_dir, args.n_outer)
    if len(h1_df):
        h1_dir = os.path.join(args.out_dir, "h1")
        os.makedirs(h1_dir, exist_ok=True)
        h1_df.sort_values(["candidate", "fold"]).to_csv(
            os.path.join(h1_dir, "per_fold.csv"), index=False
        )
        h1_summary = mean_std_table(h1_df)
        h1_summary.to_csv(os.path.join(h1_dir, "summary_mean_std.csv"), index=False)
        print("=== H1: evolved_best vs ProstNFound+ (5-fold, held-out outer-test) ===")
        print(h1_summary[["candidate", "auc_cspca_max_mean", "auc_cspca_max_std",
                           "localization_consistency_mean", "localization_consistency_std",
                           "auc_cspca_max_count"]].to_string(index=False))

        pivot = h1_df.pivot(index="fold", columns="candidate", values="auc_cspca_max")
        if "evolved_best" in pivot and "prostnfound_plus" in pivot:
            delta = pivot["evolved_best"] - pivot["prostnfound_plus"]
            print("\nauc_cspca_max: evolved_best - prostnfound_plus, per fold:")
            print(delta.to_string())
            print(f"mean delta = {delta.mean():.4f} +/- {delta.std():.4f} (n_folds={delta.notna().sum()})")
    else:
        print(f"[aggregate] no held_out_test.json found under {args.h1_dir}/fold*/")

    # --- H2: NSGA-II (evolved_best, from H1) vs random search at equal budget ---
    h2_random_df = rows_for(args.h2_dir, args.n_outer)
    if len(h2_random_df) and len(h1_df):
        h2_dir = os.path.join(args.out_dir, "h2")
        os.makedirs(h2_dir, exist_ok=True)
        evolved = h1_df[h1_df.candidate == "evolved_best"].copy()
        evolved["candidate"] = "nsga2_evolved_best"
        combined = pd.concat([evolved, h2_random_df[h2_random_df.candidate == "random_search_best"]])
        combined.sort_values(["candidate", "fold"]).to_csv(
            os.path.join(h2_dir, "per_fold.csv"), index=False
        )
        h2_summary = mean_std_table(combined)
        h2_summary.to_csv(os.path.join(h2_dir, "summary_mean_std.csv"), index=False)

        nsga2_pareto = pareto_size_for(args.h1_dir, args.n_outer)
        random_pareto = pareto_size_for(args.h2_dir, args.n_outer)
        nsga2_pareto["method"] = "nsga2"
        random_pareto["method"] = "random_search"
        pareto_compare = pd.concat([nsga2_pareto, random_pareto])
        pareto_compare.to_csv(os.path.join(h2_dir, "pareto_front_compare.csv"), index=False)

        print("\n=== H2: NSGA-II vs random search, equal budget (5-fold, held-out outer-test) ===")
        print(h2_summary[["candidate", "auc_cspca_max_mean", "auc_cspca_max_std",
                           "localization_consistency_mean", "localization_consistency_std",
                           "auc_cspca_max_count"]].to_string(index=False))
        print("\nPareto-front size and best in-search objective values per fold:")
        print(pareto_compare.sort_values(["fold", "method"]).to_string(index=False))
    else:
        print(f"[aggregate] no held_out_test.json found under {args.h2_dir}/fold*/")


if __name__ == "__main__":
    main()
