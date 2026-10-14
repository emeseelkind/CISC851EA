"""Nested, patient-grouped cross-validation on top of
../prototypeProstate/protoprostate/splits.py::patient_grouped_folds.

Outer folds are held out completely from the search (used only to report the
final H1 comparison against ProstNFound+'s loss). Each genome's fitness
during NSGA-II search (prostea/fitness.py) is the mean over *inner* folds of
the corresponding outer-train set, so the search never sees outer-test
patients -- guarding against the search itself overfitting a single split,
which nested CV over LogisticRegressionCV/RidgeCV's built-in inner loop
(as ../prototypeProstate/scripts/baseline_mean_pool.py does) doesn't need,
but a hand-rolled genome search does.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from prostea import _paths  # noqa: F401
from protoprostate.splits import patient_grouped_folds


def nested_patient_folds(meta: pd.DataFrame, n_outer: int = 5, n_inner: int = 4) -> list[dict]:
    """meta: must be indexed 0..len(meta)-1 and have a `patient_id` column,
    matching the row order region_cache.RegionCache.meta uses.

    Returns one dict per outer fold:
        outer_train, outer_test    row-index arrays into `meta`
        inner                      list of (train, val) row-index array pairs,
                                    also into `meta`, drawn only from
                                    outer_train's patients
    """
    outer_folds = patient_grouped_folds(meta, n_splits=n_outer)
    nested = []
    for outer_train_idx, outer_test_idx in outer_folds:
        inner_meta = meta.iloc[outer_train_idx].reset_index(drop=True)
        inner_folds_local = patient_grouped_folds(inner_meta, n_splits=n_inner)
        inner_folds = [
            (outer_train_idx[tr], outer_train_idx[va]) for tr, va in inner_folds_local
        ]
        nested.append(
            dict(outer_train=outer_train_idx, outer_test=outer_test_idx, inner=inner_folds)
        )
    return nested
