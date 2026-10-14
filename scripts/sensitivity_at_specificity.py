#!/usr/bin/env python
"""Sensitivity at a fixed specificity for the csPCa classification task
(grade_group >= 2), using max_density as the score -- the same score
core_metrics()'s auc_cspca_max is computed from (prostea/fitness.py::score_cores,
../prototypeProstate/protoprostate/metrics.py::core_metrics).

Nothing in the repo saves raw per-core scores or an ROC curve -- only the
aggregate AUROC -- so this retrains each candidate exactly as
scripts/run_search.py's held-out evaluation does (same train_idx/test_idx,
epochs, seed), loading the genome that was actually used from each fold's
held_out_test.json rather than re-running NSGA-II.

Usage (one outer fold; run once per fold, matching the 5 outer patient-
grouped folds the rest of H1/H2 use):
    python scripts/sensitivity_at_specificity.py --cache_dir cache/regions \
        --held_out_json experiments/h1/fold0/held_out_test.json \
        --outer_fold 0 --out_dir experiments/h1/fold0 --target_specificity 0.60
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from prostea import _paths  # noqa: F401,E402

import numpy as np
from sklearn.metrics import roc_curve

from prostea.fitness import score_cores, train_probe
from prostea.genome import decode
from prostea.region_cache import RegionCache
from prostea.splits import nested_patient_folds


def sensitivity_at_specificity(y_true, scores, target_spec: float) -> dict:
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    specificity = 1.0 - fpr
    # roc_curve returns fpr increasing (so specificity decreasing) -- flip so
    # np.interp sees an increasing x for interpolation.
    order = np.argsort(specificity)
    spec_sorted = specificity[order]
    tpr_sorted = tpr[order]
    sens_at_target = float(np.interp(target_spec, spec_sorted, tpr_sorted))
    # Also report the achieved (threshold, spec, sens) closest to the target
    # from the actual ROC vertices, for sanity-checking the interpolation.
    idx = int(np.argmin(np.abs(specificity - target_spec)))
    return dict(
        sensitivity_at_target_spec=sens_at_target,
        nearest_vertex_spec=float(specificity[idx]),
        nearest_vertex_sens=float(tpr[idx]),
        nearest_vertex_threshold=float(thresholds[idx]),
        n_positive=int(np.sum(np.asarray(y_true) == 1)),
        n_negative=int(np.sum(np.asarray(y_true) == 0)),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache_dir", required=True)
    p.add_argument("--held_out_json", required=True)
    p.add_argument("--outer_fold", type=int, required=True)
    p.add_argument("--n_outer", type=int, default=5)
    p.add_argument("--n_inner", type=int, default=4)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--target_specificity", type=float, default=0.60)
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

    results = {}
    for name in ["evolved_best", "prostnfound_plus", "topk_mil"]:
        genome = decode(held_out[name]["genome"])
        model = train_probe(genome, cache, fold["outer_train"], device=args.device,
                             epochs=args.epochs, seed=0)
        test_cores = cache.load_many(fold["outer_test"], device=args.device)
        df = score_cores(model, test_cores)
        df = df[df.max_density.notna()]

        y = (df.grade_group.to_numpy() >= 2).astype(int)
        res = sensitivity_at_specificity(y, df.max_density.to_numpy(), args.target_specificity)
        res["reported_auc_cspca_max"] = held_out[name]["metrics"].get("auc_cspca_max")
        results[name] = res
        print(f"[fold {args.outer_fold}] {name}: sensitivity@{args.target_specificity:.0%}spec="
              f"{res['sensitivity_at_target_spec']:.3f}  (n_pos={res['n_positive']}, "
              f"n_neg={res['n_negative']}, AUROC={res['reported_auc_cspca_max']:.3f})", flush=True)

    out_path = os.path.join(args.out_dir, f"sensitivity_at_spec{int(args.target_specificity*100)}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"-> {out_path}", flush=True)


if __name__ == "__main__":
    main()
