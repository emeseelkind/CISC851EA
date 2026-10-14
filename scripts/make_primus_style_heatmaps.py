#!/usr/bin/env python
"""Render our evolved genome's heatmaps in the visual style of ProstNFound+'s
own publication figure (Fig. 3: grayscale + prostate outline, next to a
jet-colormap heatmap with the needle trace outlined, labeled with PRI-MUS /
score / grade).

We do NOT attempt to reproduce ProstNFound+'s own heatmap here -- this
checkpoint's mask-decoder upsampling head doesn't load into the currently
installed medAI code (11 tensors fail to match; see the project README), so
a full-model forward pass would run an untrained, random upsampling layer.
This script only renders OUR OWN trained linear probe's heatmap, in that
figure's visual language. Where the reference figure shows "Model Score
1-5" (an internally-calibrated ProstNFound+ scale we don't have the
definition for), this shows our raw predicted probability instead, plus an
informal 1-5 bin of it -- captioned as informal, not a validated scale.

Usage matches scripts/make_heatmap_examples.py.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from prostea import _paths  # noqa: F401,E402

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import FancyBboxPatch

from protoprostate import optimum_data
from prostea.fitness import train_probe
from prostea.genome import decode
from prostea.region_cache import RegionCache
from prostea.splits import nested_patient_folds

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_heatmap_examples import GG_NAMES, pick_examples  # noqa: E402


def label_box(ax, lines, xy=(0.03, 0.97)):
    text = "\n".join(lines)
    ax.text(
        xy[0], xy[1], text, transform=ax.transAxes, ha="left", va="top",
        fontsize=10.5, fontweight="bold", color="white", linespacing=1.5,
        bbox=dict(boxstyle="square,pad=0.35", facecolor="black", alpha=0.55, edgecolor="none"),
    )


def informal_score(prob):
    return int(np.clip(np.floor(prob * 5), 0, 4)) + 1


def render_pair(ax_gray, ax_heat, image, heat, needle, prostate, labels, annotate=False):
    box_aspect = image.shape[0] / image.shape[1]  # lock each cell to the image's own
    # aspect ratio (these B-mode frames are ~1.65:1 wide) so subplots pack
    # tightly instead of leaving each row's cell letterboxed with blank space.
    ax_gray.imshow(image, cmap="gray", vmin=0, vmax=255)
    if prostate is not None:
        ax_gray.contour(prostate.astype(float), levels=[0.5], colors="white", linewidths=1.6)
    ax_gray.set_xticks([]); ax_gray.set_yticks([])
    ax_gray.set_box_aspect(box_aspect)
    label_box(ax_gray, labels)

    heat_masked = np.ma.masked_where(~prostate, heat) if prostate is not None else heat
    ax_heat.imshow(image, cmap="gray", vmin=0, vmax=255)
    cmap = matplotlib.colormaps["jet"].copy()
    cmap.set_bad(alpha=0)
    im = ax_heat.imshow(heat_masked, cmap=cmap, vmin=0, vmax=1, alpha=0.85)
    ax_heat.contour(needle.astype(float), levels=[0.5], colors="white", linewidths=2.2)
    ax_heat.set_xticks([]); ax_heat.set_yticks([])
    ax_heat.set_box_aspect(box_aspect)

    if annotate:
        h, w = image.shape
        ax_gray.annotate(
            "Prostate\nboundary", xy=(w * 0.30, h * 0.55), xytext=(w * 0.02, h * 0.85),
            color="white", fontsize=9, fontweight="bold",
            arrowprops=dict(facecolor="white", edgecolor="none", width=3, headwidth=8, shrink=0.05),
        )
        ys, xs = np.where(needle)
        if len(xs):
            nx, ny = xs.mean(), ys.mean()
            ax_heat.annotate(
                "Needle trace", xy=(nx, ny), xytext=(w * 0.02, h * 0.85),
                color="white", fontsize=9, fontweight="bold",
                arrowprops=dict(facecolor="white", edgecolor="none", width=3, headwidth=8, shrink=0.05),
            )
    return im


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache_dir", required=True)
    p.add_argument("--root", default=optimum_data.DEFAULT_ROOT)
    p.add_argument("--genome_json", required=True)
    p.add_argument("--genome_key", default="evolved_best")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--outer_fold", type=int, default=0)
    p.add_argument("--n_outer", type=int, default=5)
    p.add_argument("--n_inner", type=int, default=4)
    p.add_argument("--n_examples", type=int, default=5)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    with open(args.genome_json) as f:
        genome = decode(json.load(f)[args.genome_key]["genome"])
    print(f"[1/4] genome ({args.genome_key}): {genome.as_dict()}", flush=True)

    cache = RegionCache(args.cache_dir)
    nested = nested_patient_folds(cache.meta, n_outer=args.n_outer, n_inner=args.n_inner)
    fold = nested[args.outer_fold]

    print(f"[2/4] training probe on {len(fold['outer_train'])} outer-train cores "
          f"({args.epochs} epochs)", flush=True)
    model = train_probe(genome, cache, fold["outer_train"], device=args.device, epochs=args.epochs)
    model.eval()

    example_idx = pick_examples(cache.meta, fold["outer_test"], args.n_examples)
    core_ids = [cache.meta.iloc[i].core_id for i in example_idx]
    print(f"[3/4] examples: {core_ids}", flush=True)

    refs_by_id = {ref.core_id: ref for ref in optimum_data.list_cores(args.root)}

    print("[4/4] rendering Fig-3-style overlays", flush=True)
    n = len(example_idx)
    fig, axes = plt.subplots(
        n, 2, figsize=(8.6, 2.9 * n), constrained_layout=True,
        gridspec_kw=dict(hspace=0.06, wspace=0.03),
    )
    if n == 1:
        axes = axes[None, :]
    fig.patch.set_facecolor("black")

    for row, (ax_gray, ax_heat), row_idx in zip(range(n), axes, example_idx):
        item = cache.load(row_idx, device=args.device)
        with torch.no_grad():
            logits = model(item["feats"].unsqueeze(0))[0]
        prob = torch.sigmoid(logits)

        ref = refs_by_id[item["core_id"]]
        raw = optimum_data.load_core_item(ref)
        image = np.array(raw["image"].convert("L"))
        needle = np.array(raw["needle_mask"]) > 0
        prostate = np.array(raw.get("prostate_mask")) > 0 if raw.get("prostate_mask") is not None else np.ones_like(needle, dtype=bool)

        heat = torch.nn.functional.interpolate(
            prob[None, None], size=image.shape, mode="bilinear", align_corners=False,
        )[0, 0].numpy()

        weak_mask_t = (item["needle_mask"] & item["prostate_mask"]).cpu().numpy()
        needle_prob = float(prob.numpy()[weak_mask_t].mean()) if weak_mask_t.any() else float("nan")

        pri_mus_row = cache.meta.iloc[row_idx]
        pri_mus = pri_mus_row.get("pri_mus", float("nan"))
        gg = int(item["grade_group"])
        gg_line = "Benign" if gg == 0 else f"{GG_NAMES.get(gg, gg).upper()} ({item['involvement']*100:.0f}%)"

        labels = [f"PRI-MUS {int(pri_mus)}" if not np.isnan(pri_mus) else "PRI-MUS n/a"]
        if not np.isnan(needle_prob):
            labels.append(f"Our score {informal_score(needle_prob)}/5  (P={needle_prob:.2f})")
        else:
            labels.append("Our score n/a")
        labels.append(gg_line)

        render_pair(ax_gray, ax_heat, image, heat, needle, prostate, labels, annotate=(row == 0))

    fig.suptitle(
        "Our evolved genome's heatmap, ProstNFound+ figure style\n"
        "white outline = prostate boundary / needle trace   ·   jet = our predicted P(cancer), masked to prostate\n"
        "\"Our score\" is an informal 1–5 bin of our own probability — not ProstNFound+'s Model Score, "
        "and not a validated scale",
        fontsize=10.5, color="white",
    )
    out_path = os.path.join(args.out_dir, "primus_style_grid.png")
    fig.savefig(out_path, dpi=160, facecolor="black")
    plt.close(fig)
    print(f"      -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
