#!/usr/bin/env python
"""Paired significance tests for H1 (evolved_best vs prostnfound_plus /
topk_mil) and H2 (NSGA-II vs random search), over the 5 outer folds.

Each outer fold is one paired observation (both candidates are evaluated on
the *same* held-out test cores), so this uses a paired test across folds --
not a per-core test, which would pretend 1118 cores are 1118 independent
replicates when they're nested in only 5 fold-level draws.

With n=5 folds, exact Wilcoxon's smallest achievable two-sided p-value is
2 * (1/2)^5 = 0.0625 -- it literally cannot reach the conventional p<0.05
threshold even if every fold agrees in direction. This script reports exact
Wilcoxon, a sign test, and a paired bootstrap CI on the mean difference, and
says so explicitly rather than letting a non-significant p-value alone imply
"no effect."

--h1_dir accepts multiple run directories (e.g. experiments/h1_fixed
experiments/h1_v2), pooled as additional fold-level paired observations
(n = 5 * number of dirs). This is only valid because each run's held-out
test cores are paired *within* that run (same fold, same split) -- folds
across different run directories are independent replicates of the same
procedure, not additional pairing structure. Before prostea/fitness.py's
torch.manual_seed fix, repeated runs of "the same" H1 config (e.g. h1_fixed
vs h1_v2) differed because of leaked global RNG state, not an intentional
seed sweep -- see that fix's comment. Treat them as unlabeled independent
replicates rather than a controlled seed comparison.

Usage:
    python scripts/significance_tests.py --h1_dir experiments/h1_fixed \
        --h2_csv experiments/h2/per_fold.csv --metric auc_cspca_max

    python scripts/significance_tests.py \
        --h1_dir experiments/h1_fixed experiments/h1_v2 \
        --h2_csv experiments/h2/per_fold.csv --metric auc_cspca_max
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from scipy import stats


def paired_report(name_a: str, a: np.ndarray, name_b: str, b: np.ndarray, n_boot: int = 20000, seed: int = 0):
    assert len(a) == len(b)
    n = len(a)
    diff = a - b
    print(f"\n--- {name_a} vs {name_b} (n={n} folds) ---")
    print(f"  {name_a}: {np.array2string(a, precision=3)}  mean={a.mean():.3f}")
    print(f"  {name_b}: {np.array2string(b, precision=3)}  mean={b.mean():.3f}")
    print(f"  per-fold diff ({name_a} - {name_b}): {np.array2string(diff, precision=3)}")
    print(f"  mean diff = {diff.mean():+.4f}, std diff = {diff.std(ddof=1):.4f}")

    wins_a = int((diff > 0).sum())
    ties = int((diff == 0).sum())
    print(f"  sign test: {name_a} wins {wins_a}/{n} folds ({ties} tie(s))")
    sign_p = stats.binomtest(wins_a, n - ties, 0.5).pvalue if n - ties > 0 else float("nan")
    print(f"  sign test two-sided p = {sign_p:.4f}")

    try:
        wres = stats.wilcoxon(a, b, method="exact")
        print(f"  exact Wilcoxon signed-rank: statistic={wres.statistic:.3f}, p={wres.pvalue:.4f}  "
              f"(floor at n={n} is p={2 * (0.5 ** n):.4f})")
    except ValueError as e:
        print(f"  Wilcoxon not computable: {e}")

    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    boot_means = np.array([diff[rng.choice(idx, size=n, replace=True)].mean() for _ in range(n_boot)])
    lo, hi = np.percentile(boot_means, [2.5, 97.5])
    print(f"  paired bootstrap 95% CI on mean diff ({n_boot} resamples): [{lo:+.4f}, {hi:+.4f}]")
    frac_positive = float((boot_means > 0).mean())
    print(f"  fraction of bootstrap resamples with diff > 0: {frac_positive:.3f}")
    return dict(
        name_a=name_a, name_b=name_b, n=n,
        mean_a=float(a.mean()), mean_b=float(b.mean()), mean_diff=float(diff.mean()),
        wins_a=wins_a, sign_test_p=float(sign_p),
        boot_ci_lo=float(lo), boot_ci_hi=float(hi), boot_frac_positive=frac_positive,
    )


def load_h1(h1_dirs: list[str], metric: str) -> dict[str, np.ndarray]:
    per_candidate = {"evolved_best": [], "prostnfound_plus": [], "topk_mil": []}
    for h1_dir in h1_dirs:
        n_folds = 0
        while os.path.exists(os.path.join(h1_dir, f"fold{n_folds}", "held_out_test.json")):
            with open(os.path.join(h1_dir, f"fold{n_folds}", "held_out_test.json")) as f:
                held_out = json.load(f)
            for cand in per_candidate:
                per_candidate[cand].append(held_out[cand]["metrics"][metric])
            n_folds += 1
        if n_folds == 0:
            raise FileNotFoundError(f"no fold*/held_out_test.json under {h1_dir}")
    return {k: np.array(v, dtype=float) for k, v in per_candidate.items()}


def load_h2(h2_csv: str, metric: str) -> dict[str, np.ndarray]:
    df = pd.read_csv(h2_csv)
    piv = df.pivot(index="fold", columns="candidate", values=metric).sort_index()
    return {col: piv[col].to_numpy(dtype=float) for col in piv.columns}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--h1_dir", nargs="+", default=["experiments/h1_fixed"])
    p.add_argument("--h2_csv", default="experiments/h2/per_fold.csv")
    p.add_argument("--metric", default="auc_cspca_max")
    p.add_argument("--out", default=None, help="optional JSON report path")
    args = p.parse_args()

    results = []

    print(f"=== H1: evolved loss vs. ProstNFound+'s loss / top-k MIL (metric={args.metric}) ===")
    print(f"source: {args.h1_dir} ({len(args.h1_dir)} run(s) pooled as independent fold replicates)")
    h1 = load_h1(args.h1_dir, args.metric)
    results.append(paired_report("evolved_best", h1["evolved_best"], "prostnfound_plus", h1["prostnfound_plus"]))
    results.append(paired_report("evolved_best", h1["evolved_best"], "topk_mil", h1["topk_mil"]))

    print(f"\n\n=== H2: NSGA-II vs. random search at equal budget (metric={args.metric}) ===")
    print(f"source: {args.h2_csv}")
    h2 = load_h2(args.h2_csv, args.metric)
    keys = list(h2.keys())
    nsga_key = next(k for k in keys if "nsga" in k.lower() or "evolved" in k.lower())
    rand_key = next(k for k in keys if "random" in k.lower())
    results.append(paired_report(nsga_key, h2[nsga_key], rand_key, h2[rand_key]))

    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
