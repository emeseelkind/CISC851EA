#!/usr/bin/env python
"""Overlay a trained genome's predicted cancer-likelihood heatmap on the real
B-mode image for a handful of held-out test cores.

Only needs the region cache (prostea/region_cache.py) to retrain the linear
probe and the raw OPTIMUM PNGs (for display) -- no GPU, no medAI, no frozen
encoder forward pass, since the embeddings are already cached.

Usage:
    python scripts/make_heatmap_examples.py \
        --cache_dir cache/regions --genome_json experiments/search/held_out_test.json \
        --genome_key evolved_best --out_dir experiments/search/heatmap_examples --n_examples 5
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

from protoprostate import optimum_data
from prostea.fitness import train_probe
from prostea.genome import decode
from prostea.region_cache import RegionCache
from prostea.splits import nested_patient_folds

GG_NAMES = {0: "benign", 1: "GG1", 2: "GG2", 3: "GG3", 4: "GG4", 5: "GG5"}


def draw_overlay(ax, image, heat, needle, prostate, title):
    ax.imshow(image, cmap="gray", vmin=0, vmax=255)
    heat_masked = np.ma.masked_where(~prostate, heat) if prostate is not None else heat
    cmap = matplotlib.colormaps["jet"].copy()
    cmap.set_bad(alpha=0)
    im = ax.imshow(heat_masked, cmap=cmap, alpha=0.75, vmin=0, vmax=1)
    ax.contour(needle.astype(float), levels=[0.5], colors="white", linewidths=1.8, linestyles="--")
    if prostate is not None:
        ax.contour(prostate.astype(float), levels=[0.5], colors="white", linewidths=1.3)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=9)
    return im


def pick_examples(meta, test_idx, n_examples):
    """1 benign (largest needle region, for a clean visual) + up to n-1 cancer
    cores spanning as many distinct grade groups as possible, highest
    involvement within each -- a representative spread, not cherry-picked
    for correctness.
    """
    sub = meta.iloc[test_idx]
    chosen = []

    benign = sub[sub.grade_group == 0].sort_values("n_needle_patches", ascending=False)
    if len(benign):
        chosen.append(benign.index[0])

    cancer = sub[sub.grade_group > 0].sort_values(
        ["grade_group", "involvement"], ascending=[False, False]
    )
    seen_gg = set()
    for idx, row in cancer.iterrows():
        if len(chosen) >= n_examples:
            break
        if row.grade_group in seen_gg:
            continue
        seen_gg.add(row.grade_group)
        chosen.append(idx)
    for idx, row in cancer.iterrows():
        if len(chosen) >= n_examples:
            break
        if idx not in chosen:
            chosen.append(idx)
    return chosen[:n_examples]


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

    print("[4/4] rendering overlays", flush=True)
    fig_grid, axes = plt.subplots(1, len(example_idx), figsize=(4.2 * len(example_idx), 4.6))
    if len(example_idx) == 1:
        axes = [axes]

    for ax, row_idx in zip(axes, example_idx):
        item = cache.load(row_idx, device=args.device)
        with torch.no_grad():
            logits = model(item["feats"].unsqueeze(0))[0]  # (Gh, Gw)
        prob = torch.sigmoid(logits)

        ref = refs_by_id[item["core_id"]]
        raw = optimum_data.load_core_item(ref)
        image = np.array(raw["image"].convert("L"))
        needle = np.array(raw["needle_mask"]) > 0
        prostate = np.array(raw.get("prostate_mask")) > 0 if raw.get("prostate_mask") is not None else None

        heat = torch.nn.functional.interpolate(
            prob[None, None], size=image.shape, mode="bilinear", align_corners=False,
        )[0, 0].numpy()

        weak_mask_np = (item["needle_mask"] & item["prostate_mask"]).cpu().numpy()
        needle_probs_np = prob.numpy()[weak_mask_np]
        needle_frac = float(needle_probs_np.mean()) if weak_mask_np.any() else float("nan")
        needle_max = float(needle_probs_np.max()) if weak_mask_np.any() else float("nan")

        gg = int(item["grade_group"])
        title = f"{item['core_id']}\n{GG_NAMES.get(gg, gg)}  involvement={item['involvement']:.2f}"
        if not np.isnan(needle_frac):
            # mean and max diverge whenever the genome's pooling operator (not
            # necessarily a plain mean -- see prostea/pooling.py) only needs a
            # subset of needle patches to be high to satisfy the training loss.
            title += f"\nP(cancer) in needle: mean={needle_frac:.2f}  max={needle_max:.2f}"

        draw_overlay(ax, image, heat, needle, prostate, title)

        fig_single, ax_single = plt.subplots(figsize=(5, 5.4))
        im_single = draw_overlay(ax_single, image, heat, needle, prostate, title)
        ax_single.set_title(title, fontsize=10)
        cbar = fig_single.colorbar(im_single, ax=ax_single, fraction=0.046, pad=0.03)
        cbar.set_label("predicted P(cancer)")
        fig_single.tight_layout()
        out_path = os.path.join(args.out_dir, f"heatmap_{item['core_id']}.png")
        fig_single.savefig(out_path, dpi=150)
        plt.close(fig_single)
        print(f"      -> {out_path}", flush=True)

    fig_grid.suptitle(
        "dashed white = needle trace  ·  solid white = prostate boundary  ·  "
        "heat (blue=benign to red=cancer, masked to prostate) = predicted cancer likelihood "
        "(evolved genome, held-out test)",
        fontsize=9, y=1.03,
    )
    fig_grid.tight_layout()
    grid_path = os.path.join(args.out_dir, "heatmap_grid.png")
    fig_grid.savefig(grid_path, dpi=150, bbox_inches="tight")
    plt.close(fig_grid)
    print(f"      -> {grid_path}", flush=True)


if __name__ == "__main__":
    main()
