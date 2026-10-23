#!/usr/bin/env python
"""Run NSGA-II loss search for one outer fold, then evaluate the Pareto
front's best-csPCa genome against the two named fixed points
(genome.PROSTNFOUND_PLUS, genome.TOPK_MIL) on that fold's held-out test
patients -- H1's controlled comparison.

Usage:
    python scripts/run_search.py --cache_dir cache/regions --exp_dir experiments/fold0 \
        --outer_fold 0 --pop_size 16 --n_gen 10

The search (NSGA-II over inner folds) never sees outer-test patients; the
held-out comparison at the end is the only thing that touches them.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prostea.fitness import evaluate_genome
from prostea.genome import PROSTNFOUND_PLUS, TOPK_MIL
from prostea.region_cache import RegionCache
from prostea.search import pareto_front_to_records, run_search
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
    p.add_argument("--detector", choices=["pnf", "linear"], default="pnf",
                    help="pnf = real ProstNFound+ mask decoder, warm-started from its trained "
                         "weights (prostea/pnf_detector.py); linear = from-scratch 1x1 conv "
                         "(prostea/model.py::LinearProbe), kept for comparison/debugging.")
    p.add_argument("--lr", type=float, default=None,
                    help="defaults to 1e-3 for --detector pnf (fine-tuning a pretrained "
                         "transformer decoder; train_probe's old 1e-2 default, tuned for "
                         "LinearProbe's from-scratch 1x1 conv, destroys the pretrained init "
                         "within a few steps) or 1e-2 for --detector linear.")
    args = p.parse_args()

    model_builder = None
    if args.detector == "pnf":
        from prostea.pnf_detector import build_pnf_detector
        model_builder = build_pnf_detector
    lr = args.lr if args.lr is not None else (1e-3 if args.detector == "pnf" else 1e-2)

    os.makedirs(args.exp_dir, exist_ok=True)
    cache = RegionCache(args.cache_dir)
    nested = nested_patient_folds(cache.meta, n_outer=args.n_outer, n_inner=args.n_inner)
    fold = nested[args.outer_fold]

    print(f"[search] outer fold {args.outer_fold}: "
          f"{len(fold['outer_train'])} train / {len(fold['outer_test'])} test cores, "
          f"{len(fold['inner'])} inner folds", flush=True)

    res = run_search(
        cache, fold["inner"],
        pop_size=args.pop_size, n_gen=args.n_gen, device=args.device,
        epochs=args.epochs, max_oop_activation=args.max_oop_activation, seed=args.seed,
        model_builder=model_builder, lr=lr,
    )
    pareto = pareto_front_to_records(res)
    pareto.sort(key=lambda r: r["auc_cspca_max"], reverse=True)
    with open(os.path.join(args.exp_dir, "pareto_front.json"), "w") as f:
        json.dump(pareto, f, indent=2)
    print(f"[search] {len(pareto)} Pareto-optimal genomes -> "
          f"{args.exp_dir}/pareto_front.json", flush=True)

    if not pareto:
        print("[search] empty Pareto front (all individuals infeasible under "
              "max_oop_activation); nothing to evaluate on held-out test.", flush=True)
        return

    from prostea.genome import decode
    best_genome = decode(pareto[0]["genome"])

    candidates = {
        "evolved_best": best_genome,
        "prostnfound_plus": PROSTNFOUND_PLUS,
        "topk_mil": TOPK_MIL,
    }
    held_out = {}
    for name, genome in candidates.items():
        metrics = evaluate_genome(
            genome, cache, fold["outer_train"], fold["outer_test"],
            device=args.device, epochs=args.epochs, seed=0, model_builder=model_builder, lr=lr,
        )
        held_out[name] = dict(genome=genome.as_dict(), metrics=metrics)
        print(f"[held-out test] {name}: auc_cspca_max={metrics.get('auc_cspca_max'):.3f} "
              f"localization_consistency={metrics.get('localization_consistency')}", flush=True)

    with open(os.path.join(args.exp_dir, "held_out_test.json"), "w") as f:
        json.dump(held_out, f, indent=2)


if __name__ == "__main__":
    main()
