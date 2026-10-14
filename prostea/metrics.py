"""Classification metrics (reused from ../prototypeProstate) plus the one
metric that doesn't exist anywhere yet: a localization-*consistency* proxy.

No pixel-level histopathology exists for this data, so there is no ground
truth to score localization *accuracy* against. What the needle biopsy does
give us is involvement (% cancer in the core) -- if a genome's heatmap is
spatially consistent with pathology, the fraction of the needle trace it
lights up should track involvement across cores. That correlation is the
`localization_consistency` proxy below. Never call this "localization
accuracy" -- see the project README's "claims you can't make" section.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from prostea import _paths  # noqa: F401  (adds ../prototypeProstate to sys.path)
from protoprostate.metrics import core_metrics, per_center_metrics, safe_auc  # noqa: F401


def localization_consistency(df: pd.DataFrame) -> dict:
    """df needs: grade_group, involvement, needle_activated_fraction (the
    fraction of needle-region patches with predicted prob above a fixed
    threshold -- see prostea/fitness.py::score_cores). Restricted to cancer
    cores with known involvement, since benign cores have involvement=0 by
    construction and would trivially inflate the correlation.
    """
    cancer = df[(df.grade_group > 0) & df.involvement.notna()]
    if len(cancer) < 2 or cancer.involvement.std() == 0:
        return {"localization_consistency": float("nan"), "n_localization": int(len(cancer))}
    corr = np.corrcoef(cancer.needle_activated_fraction, cancer.involvement)[0, 1]
    return {"localization_consistency": float(corr), "n_localization": int(len(cancer))}


def out_of_prostate_activation(df: pd.DataFrame) -> float:
    """Mean predicted probability outside the prostate mask, averaged over
    cores that have any out-of-prostate patches. This is the quantity the
    proposal's "out-of-prostate activation constraint" bounds.
    """
    known = df.negative_mean_prob.notna()
    if known.sum() == 0:
        return float("nan")
    return float(df.negative_mean_prob[known].mean())
