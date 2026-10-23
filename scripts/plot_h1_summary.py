#!/usr/bin/env python
"""Cross-fold summary for H1: does the evolved loss beat ProstNFound+'s loss
by more than fold-to-fold variance (README's own bar)? Reads each outer
fold's experiments/h1_fixed/fold{0..4}/sens_spec_curve.json (written by
scripts/sens_spec_curve.py) and plots per-method AUROC and sens@60%spec as
a strip of per-fold points plus a mean +/- std marker.

Also draws a reference line for the REAL, deployed pnf_plus_final.pth
checkpoint run zero-shot on this same cohort (no retraining, no CV needed --
the checkpoint's own saved args show cohort_selection_mode="train_only"
within medAI/datasets/nct2013/cohort_selection.py, i.e. it was trained
entirely on NCT2013 and never saw OPTIMUM/UA, so every core here is equally
unseen to it). This is NOT the same thing as the "prostnfound_plus" bar: that
bar is a frozen-encoder + freshly-trained linear probe using an *approximation*
of PNF+'s loss (see prostea/genome.py's PROSTNFOUND_PLUS docstring for what
the approximation can't represent), matched in capacity/budget to
evolved_best/topk_mil so the loss-function ablation is controlled. The
checkpoint line answers a different question -- "how does the actual paper
model do here" -- and is included for context, not as a 4th ablation arm.
Source: ../prostNfound/experiments/pnf_plus_prospective_repro_v2/metrics.json
(all_cores -> csPCa_core), from the corrected v2 reproduction (see memory
note prostnfound_plus_repro).

Usage:
    python scripts/plot_h1_summary.py --exp_dir experiments/h1_fixed --n_folds 5 \
        --pnf_plus_checkpoint_json \
        ../prostNfound/experiments/pnf_plus_prospective_repro_v2/metrics.json
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
REF_COLOR = "#4a3aa7"
LABELS = {
    "evolved_best": "Evolved\n(NSGA-II)",
    "prostnfound_plus": "ProstNFound+\nloss",
    "topk_mil": "Top-k\nMIL",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_dir", required=True)
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--pnf_plus_checkpoint_json", default=None,
                     help="../prostNfound's metrics.json, for the deployed-checkpoint reference line")
    args = ap.parse_args()

    ref = None
    if args.pnf_plus_checkpoint_json:
        with open(args.pnf_plus_checkpoint_json) as fh:
            m = json.load(fh)["all_cores"]["csPCa_core"]
        ref = dict(auroc=m["auroc"], sens60=m["sens_at_60spec"], n=m["n"])

    data = {m: {"auroc": [], "sens60": []} for m in METHODS}
    for f in range(args.n_folds):
        with open(os.path.join(args.exp_dir, f"fold{f}", "sens_spec_curve.json")) as fh:
            d = json.load(fh)
        for m in METHODS:
            data[m]["auroc"].append(d[m]["auroc"])
            data[m]["sens60"].append(d[m]["sens_at_60spec"])

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 5.0), dpi=150)
    fig.patch.set_facecolor("#fcfcfb")

    panels = [("auroc", "AUROC", axes[0]), ("sens60", "Sensitivity @ 60% specificity", axes[1])]
    x_positions = {m: i for i, m in enumerate(METHODS)}
    rng = np.random.default_rng(0)

    for key, ylabel, ax in panels:
        ax.set_facecolor("#fcfcfb")
        for m in METHODS:
            vals = np.array(data[m][key])
            x0 = x_positions[m]
            jitter = rng.uniform(-0.08, 0.08, size=len(vals))
            ax.scatter(
                np.full(len(vals), x0) + jitter, vals,
                color=COLORS[m], alpha=0.55, s=36, zorder=2, edgecolor="none",
            )
            mean, std = vals.mean(), vals.std()
            ax.errorbar(
                [x0], [mean], yerr=[std], fmt="o", color=COLORS[m],
                markersize=9, markeredgecolor="#fcfcfb", markeredgewidth=1.2,
                elinewidth=2, capsize=5, capthick=2, zorder=3,
            )
            ax.annotate(
                f"{mean:.2f}±{std:.2f}",
                xy=(x0, mean + std),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                fontsize=8.5,
                color=COLORS[m],
                fontweight="bold",
            )

        if ref is not None:
            ref_val = ref[key]
            ax.axhline(ref_val, color=REF_COLOR, linestyle="--", linewidth=1.5, zorder=1)
            ax.annotate(
                f"deployed checkpoint: {ref_val:.2f}",
                xy=(2.5, ref_val),
                xytext=(-4, 5),
                textcoords="offset points",
                ha="right",
                fontsize=8.5,
                color=REF_COLOR,
                fontweight="bold",
            )

        ax.set_xticks(list(x_positions.values()))
        ax.set_xticklabels([LABELS[m] for m in METHODS], color="#0b0b0b", fontsize=9)
        ax.set_xlim(-0.5, 2.5)
        ax.set_ylim(0, 1.08)
        ax.set_ylabel(ylabel, color="#0b0b0b")
        ax.grid(True, axis="y", color="#e1e0d9", linewidth=0.8, zorder=0)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color("#c3c2b7")
        ax.tick_params(colors="#52514e")

    subtitle = "dots = each fold; bold marker = mean ± 1 std (csPCa, GG≥3)"
    if ref is not None:
        subtitle += f"\ndashed line = real pnf_plus_final.pth checkpoint, zero-shot (n={ref['n']}, not a 4th ablation arm)"
    fig.suptitle(
        f"H1 across {args.n_folds} outer folds: evolved loss vs. ProstNFound+ loss vs. top-k MIL\n"
        f"{subtitle}",
        fontsize=11, color="#0b0b0b",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    out_path = os.path.join(args.exp_dir, "h1_summary_across_folds.png")
    fig.savefig(out_path, facecolor=fig.get_facecolor())
    print(f"Saved figure to {out_path}")

    print("\nmean +/- std:")
    for key, label, _ in panels:
        print(f"  {label}:")
        for m in METHODS:
            vals = np.array(data[m][key])
            print(f"    {m:<18} {vals.mean():.3f} +/- {vals.std():.3f}  (per-fold: {[round(v,3) for v in vals]})")
        if ref is not None:
            print(f"    {'deployed checkpoint':<18} {ref[key]:.3f}  (n={ref['n']}, zero-shot, no CV)")


if __name__ == "__main__":
    main()
