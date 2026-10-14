"""NSGA-II over the genome space (prostea/genome.py), objectives:
    maximize  auc_cspca_max              (H1's target metric)
    maximize  localization_consistency   (the proxy in prostea/metrics.py)
under one constraint:
    out_of_prostate_activation <= max_oop_activation

pymoo minimizes by convention, so both objectives are negated in `_evaluate`.
Requires pymoo>=0.6 (`pip install pymoo`; not a dependency of either sibling
project -- see requirements.txt).
"""

from __future__ import annotations

import numpy as np

from prostea.fitness import evaluate_genome_nested
from prostea.genome import BOUNDS, GG_TARGET_MODES, decode
from prostea.pooling import POOLING_CHOICES


def problem_vars():
    from pymoo.core.variable import Choice, Real

    return {
        "pooling": Choice(options=POOLING_CHOICES),
        "pooling_param": Real(bounds=BOUNDS["pooling_param"]),
        "gg_target_mode": Choice(options=GG_TARGET_MODES),
        "w_weak": Real(bounds=BOUNDS["w_weak"]),
        "w_negative": Real(bounds=BOUNDS["w_negative"]),
        "w_unlabeled": Real(bounds=BOUNDS["w_unlabeled"]),
        "pu_prior": Real(bounds=BOUNDS["pu_prior"]),
        "pos_weight": Real(bounds=BOUNDS["pos_weight"]),
    }


def _make_problem(cache, inner_folds, device, epochs, max_oop_activation):
    from pymoo.core.problem import ElementwiseProblem

    class LossSearchProblem(ElementwiseProblem):
        def __init__(self):
            super().__init__(vars=problem_vars(), n_obj=2, n_ieq_constr=1)

        def _evaluate(self, x, out, *args, **kwargs):
            genome = decode(x)
            metrics = evaluate_genome_nested(
                genome, cache, inner_folds, device=device, epochs=epochs
            )
            cspca = metrics["auc_cspca_max"]
            loc = metrics["localization_consistency"]
            # Undefined-metric fallbacks: a fold with a single class present
            # gives NaN AUROC (chance-level credit); too few graded cancer
            # cores gives NaN correlation (no consistency credit). Both keep
            # the objective well-defined without rewarding degenerate genomes.
            cspca = 0.5 if np.isnan(cspca) else cspca
            loc = 0.0 if np.isnan(loc) else loc
            out["F"] = [-cspca, -loc]
            oop = metrics["out_of_prostate_activation"]
            oop = 1.0 if np.isnan(oop) else oop  # no out-of-prostate data -> treat as worst-case
            out["G"] = [oop - max_oop_activation]
            out["metrics"] = metrics

    return LossSearchProblem()


def run_search(
    cache,
    inner_folds,
    pop_size: int = 16,
    n_gen: int = 10,
    device: str = "cpu",
    epochs: int = 8,
    max_oop_activation: float = 0.15,
    seed: int = 1,
    verbose: bool = True,
):
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.core.mixed import (
        MixedVariableDuplicateElimination,
        MixedVariableMating,
        MixedVariableSampling,
    )
    from pymoo.optimize import minimize

    problem = _make_problem(cache, inner_folds, device, epochs, max_oop_activation)
    algorithm = NSGA2(
        pop_size=pop_size,
        sampling=MixedVariableSampling(),
        mating=MixedVariableMating(eliminate_duplicates=MixedVariableDuplicateElimination()),
        eliminate_duplicates=MixedVariableDuplicateElimination(),
    )
    return minimize(problem, algorithm, ("n_gen", n_gen), seed=seed, verbose=verbose)


def pareto_front_to_records(res) -> list[dict]:
    """res: the return value of run_search(). One record per Pareto-optimal
    individual: its genome, and the (un-negated) objective values. Returns []
    if NSGA-II found no individual satisfying the out-of-prostate activation
    constraint (res.X is None in that case).
    """
    if res.X is None:
        return []
    xs = np.atleast_1d(res.X)
    fs = np.atleast_2d(res.F)
    records = []
    for x, f in zip(xs, fs):
        genome = decode(x)
        records.append(dict(
            genome=genome.as_dict(),
            auc_cspca_max=float(-f[0]),
            localization_consistency=float(-f[1]),
        ))
    return records


def non_dominated_front(
    records: list[dict],
    objective_keys: tuple[str, str] = ("auc_cspca_max", "localization_consistency"),
) -> list[dict]:
    """Plain O(n^2) non-dominated filter (both objectives maximized), used by
    scripts/random_search_baseline.py, which samples genomes directly rather
    than going through pymoo's own NSGA2 loop.
    """
    def dominates(a, b):
        better_or_eq = all(a[k] >= b[k] for k in objective_keys)
        strictly_better = any(a[k] > b[k] for k in objective_keys)
        return better_or_eq and strictly_better

    return [r for r in records if not any(dominates(o, r) for o in records if o is not r)]
