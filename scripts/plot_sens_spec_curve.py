#!/usr/bin/env python
"""Render the figure for scripts/sens_spec_curve.py's output. Separate step
because the proside-dino-pca venv (needed for torch/prostea to compute the
curve) has no matplotlib; this only reads the saved .npz/.json, so it can run
under any venv that has matplotlib (e.g. ../prostNfound/.venv).

Usage:
    python scripts/plot_sens_spec_curve.py --in_dir experiments/h1/fold0 --outer_fold 0
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

METHODS = ["evolved_best", "prostnfound_plus", "topk_mil"]
COLORS = {"evolved_best": "#1baf7a", "prostnfound_plus": "#eb6834", "topk_mil": "#2a78d6"}
LABELS = {
    "evolved_best": "Evolved (NSGA-II)",
    "prostnfound_plus": "ProstNFound+ loss",
    "topk_mil": "Top-k MIL",
}
OFFSETS = {"evolved_best": (8, 8), "prostnfound_plus": (10, 2), "topk_mil": (-46, -14)}
OFFSETS_NEAR_TOP = {"evolved_best": (-50, 2), "prostnfound_plus": (10, 6), "topk_mil": (10, -14)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True)
    ap.add_argument("--outer_fold", type=int, required=True)
    args = ap.parse_args()

    npz = np.load(os.path.join(args.in_dir, "sens_spec_curve.npz"))
    with open(os.path.join(args.in_dir, "sens_spec_curve.json")) as f:
        summary = json.load(f)

    fig, ax = plt.subplots(figsize=(6.4, 5.2), dpi=150)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")

    for name in METHODS:
        spec, sens = npz[f"{name}_spec"], npz[f"{name}_sens"]
        color = COLORS[name]
        ax.plot(spec * 100, sens * 100, color=color, linewidth=2, label=LABELS[name], zorder=3)

        s60 = summary[name]["sens_at_60spec"] * 100
        offsets = OFFSETS_NEAR_TOP if s60 > 95 else OFFSETS
        ax.scatter([60], [s60], color=color, s=48, zorder=4, edgecolor="#fcfcfb", linewidth=1.2)
        ax.annotate(
            f"{s60:.0f}%",
            xy=(60, s60),
            xytext=offsets[name],
            textcoords="offset points",
            fontsize=9,
            color=color,
            fontweight="bold",
        )

    ax.axvline(60, color="#898781", linestyle="--", linewidth=1, zorder=1)
    ax.set_xlim(100, 0)
    ax.set_ylim(0, 102)
    ax.set_xlabel("Specificity (%)", color="#0b0b0b")
    ax.set_ylabel("Sensitivity (%)", color="#0b0b0b")
    n_pos = summary[METHODS[0]]["n_positive"]
    n_neg = summary[METHODS[0]]["n_negative"]
    ax.set_title(
        f"csPCa sensitivity vs. specificity -- ProstEA H1 (fixed), outer fold {args.outer_fold}\n"
        f"evolved loss vs. ProstNFound+ loss vs. top-k MIL (n={n_pos}+/{n_neg}-)",
        fontsize=11,
        color="#0b0b0b",
    )
    ax.grid(True, color="#e1e0d9", linewidth=0.8, zorder=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#c3c2b7")
    ax.tick_params(colors="#52514e")
    ax.legend(frameon=False, loc="lower left", labelcolor="#0b0b0b")

    fig.tight_layout()
    out_path = os.path.join(args.in_dir, "sens_spec_curve.png")
    fig.savefig(out_path, facecolor=fig.get_facecolor())
    print(f"Saved figure to {out_path}")


if __name__ == "__main__":
    main()
