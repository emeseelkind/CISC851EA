#!/usr/bin/env python
"""Full sensitivity-vs-specificity curve for H1's held-out fold, comparing
evolved_best / prostnfound_plus / topk_mil (csPCa = grade_group >= 3, matching
the corrected ../prototypeProstate/protoprostate/metrics.py::core_metrics
definition -- GG2 is isPCa, not clinically significant), using max_density as
the score (prostea/fitness.py::score_cores).

NOTE: scripts/sensitivity_at_specificity.py (this script's sibling for a
single operating point) still uses the stale `grade_group >= 2` threshold --
that was core_metrics' own definition before its off-by-one fix (see the
comment in protoprostate/metrics.py::core_metrics). Don't copy that script's
threshold without checking it against the current core_metrics first.

Also use experiments/h1_fixed/, not experiments/h1/ -- h1's own
PROSTNFOUND_PLUS fixed-point genome was itself wrong (built from the public
prostNfound repo's demo config, not the real medAI training code that
produced pnf_plus_final.pth; see prostea/genome.py's "CORRECTED" comment
above PROSTNFOUND_PLUS). h1_fixed reran H1 with the corrected genome.

Companion to scripts/sensitivity_at_specificity.py, which only reports the
single operating point at one target specificity (default 60%). Nothing in
the repo saves raw per-core scores or ROC points, so -- like that script --
this retrains each of the three genomes exactly as scripts/run_search.py's
held-out evaluation does (same train_idx/test_idx, epochs, seed), loading
each genome from the fold's held_out_test.json rather than re-running
NSGA-II.

Usage:
    python scripts/sens_spec_curve.py --cache_dir cache/regions \
        --held_out_json experiments/h1_fixed/fold0/held_out_test.json \
        --outer_fold 0 --out_dir experiments/h1_fixed/fold0
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from prostea import _paths  # noqa: F401,E402

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

from prostea.fitness import score_cores, train_probe
from prostea.genome import decode
from prostea.region_cache import RegionCache
from prostea.splits import nested_patient_folds

METHODS = ["evolved_best", "prostnfound_plus", "topk_mil"]
# dataviz categorical slots 1/2/3 (blue/orange/aqua) -- validated CVD-safe order
COLORS = {"evolved_best": "#1baf7a", "prostnfound_plus": "#eb6834", "topk_mil": "#2a78d6"}
TARGET_SPEC = 0.60


def curve(y_true, scores):
    fpr, tpr, thresh = roc_curve(y_true, scores)
    spec = 1 - fpr
    order = np.argsort(spec)
    return spec[order], tpr[order], thresh[order]


def sens_at_spec(spec, sens, target):
    idx = np.where(spec >= target)[0]
    if len(idx) == 0:
        return float("nan")
    return float(sens[idx].max())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache_dir", required=True)
    p.add_argument("--held_out_json", required=True)
    p.add_argument("--outer_fold", type=int, required=True)
    p.add_argument("--n_outer", type=int, default=5)
    p.add_argument("--n_inner", type=int, default=4)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    with open(args.held_out_json) as f:
        held_out = json.load(f)

    cache = RegionCache(args.cache_dir)
    nested = nested_patient_folds(cache.meta, n_outer=args.n_outer, n_inner=args.n_inner)
    fold = nested[args.outer_fold]
    print(f"[fold {args.outer_fold}] {len(fold['outer_train'])} train / "
          f"{len(fold['outer_test'])} test cores", flush=True)

    curves, summary = {}, {}
    for name in METHODS:
        genome = decode(held_out[name]["genome"])
        model = train_probe(genome, cache, fold["outer_train"], device=args.device,
                             epochs=args.epochs, seed=0)
        test_cores = cache.load_many(fold["outer_test"], device=args.device)
        df = score_cores(model, test_cores)
        df = df[df.max_density.notna()]

        y = (df.grade_group.to_numpy() >= 3).astype(int)
        scores = df.max_density.to_numpy()
        spec, sens, _ = curve(y, scores)
        curves[name] = (spec, sens)
        summary[name] = dict(
            auroc=float(roc_auc_score(y, scores)),
            sens_at_60spec=sens_at_spec(spec, sens, TARGET_SPEC),
            n_positive=int(y.sum()),
            n_negative=int((1 - y).sum()),
        )
        print(f"[fold {args.outer_fold}] {name}: AUROC={summary[name]['auroc']:.3f} "
              f"sens@60spec={summary[name]['sens_at_60spec']:.3f}", flush=True)

    with open(os.path.join(args.out_dir, "sens_spec_curve.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Save raw curve points too -- this venv (proside-dino-pca, needed for
    # torch/prostea) has no matplotlib, so plotting happens as a separate
    # step (plot_sens_spec_curve.py) under a venv that does.
    npz_path = os.path.join(args.out_dir, "sens_spec_curve.npz")
    np.savez(npz_path, **{f"{name}_spec": curves[name][0] for name in METHODS},
              **{f"{name}_sens": curves[name][1] for name in METHODS})
    print(f"Saved curve points to {npz_path}")
    print("Run scripts/plot_sens_spec_curve.py (needs matplotlib, e.g. the "
          "../prostNfound/.venv) to render the figure.")


if __name__ == "__main__":
    main()
