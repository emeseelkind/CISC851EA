#!/usr/bin/env python
"""Cache the *full* frozen-PNF+ patch grid + both region masks for every
OPTIMUM core, so prostea/search.py never has to re-run the encoder.

This is a fork of ../prototypeProstate/scripts/extract_embeddings.py: that
script keeps only needle-region patches (discarding everything else at
extraction time), which was enough for prototypeProstate's mean-pool
baseline but throws away exactly the out-of-prostate and prostate-not-needle
information this project's region-wise loss search needs. This script keeps
the whole (Gh, Gw, D) grid plus both masks instead.

Usage:
    python scripts/cache_regions.py --out_dir cache/regions --limit 8   # smoke test
    python scripts/cache_regions.py --out_dir cache/regions             # full run (GPU node)

Output, at <out_dir>:
    npz/<core_id>.npz     feats (Gh,Gw,D) float16, needle_mask (Gh,Gw) uint8,
                           prostate_mask (Gh,Gw) uint8
    metadata.csv           core_id, patient_id, center, grade_group,
                            involvement, pri_mus, age, psa,
                            n_prostate_patches, n_needle_patches,
                            has_prostate_mask
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from prostea import _paths  # noqa: F401,E402  (adds ../prototypeProstate to sys.path)

import numpy as np
import torch

from protoprostate import optimum_data, pnf_plus


def build_transform(args_from_ckpt):
    from medAI.transforms.prostnfound_transform import ProstNFoundTransform

    return ProstNFoundTransform(
        augment="none",
        image_size=args_from_ckpt.get("image_size", 256),
        mask_size=pnf_plus.GRID_SIZE,
        mean=args_from_ckpt.get("mean", [0, 0, 0]),
        std=args_from_ckpt.get("std", [1, 1, 1]),
        return_raw_images=False,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=optimum_data.DEFAULT_ROOT)
    p.add_argument("--checkpoint", default=pnf_plus.DEFAULT_CHECKPOINT)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--limit", type=int, default=None, help="process only the first N cores (smoke test)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--skip_existing", action="store_true")
    args = p.parse_args()

    npz_dir = os.path.join(args.out_dir, "npz")
    os.makedirs(npz_dir, exist_ok=True)
    metadata_path = os.path.join(args.out_dir, "metadata.csv")

    print(f"[1/3] loading PNF+ checkpoint from {args.checkpoint} on {args.device}", flush=True)
    t0 = time.time()
    model, ckpt_args, skipped_keys = pnf_plus.load_pnf_plus(args.checkpoint, device=args.device)
    print(f"      loaded in {time.time()-t0:.1f}s; {len(skipped_keys)} incompatible "
          f"(mask-decoder-only) keys skipped", flush=True)
    transform = build_transform(ckpt_args)

    print(f"[2/3] enumerating cores under {args.root}", flush=True)
    cores = optimum_data.list_cores(args.root)
    if args.limit:
        cores = cores[: args.limit]
    print(f"      {len(cores)} cores", flush=True)
    meta_table = optimum_data.build_metadata_table(args.root).set_index("core_id")

    write_header = not os.path.isfile(metadata_path)
    metadata_file = open(metadata_path, "a", newline="")
    writer = csv.DictWriter(
        metadata_file,
        fieldnames=[
            "core_id", "patient_id", "center", "grade_group", "involvement",
            "pri_mus", "age", "psa", "n_prostate_patches", "n_needle_patches",
            "has_prostate_mask",
        ],
    )
    if write_header:
        writer.writeheader()

    print("[3/3] extracting full patch-grid embeddings + region masks", flush=True)
    n_ok, n_skip, n_err, n_no_prostate_mask, n_degenerate_prostate_mask = 0, 0, 0, 0, 0
    t_extract0 = time.time()
    for i, ref in enumerate(cores):
        npz_path = os.path.join(npz_dir, f"{ref.core_id}.npz")
        if args.skip_existing and os.path.isfile(npz_path):
            n_skip += 1
            continue
        try:
            item = optimum_data.load_core_item(ref)
            has_prostate_mask = "prostate_mask" in item
            out = transform(item)
            bmode = out["bmode"].unsqueeze(0).to(args.device)  # (1, 3, 256, 256)
            needle_mask = (out["needle_mask"].squeeze().numpy() > 0.5).astype(np.uint8)  # (64, 64)
            prostate_mask = (
                (out["prostate_mask"].squeeze().numpy() > 0.5).astype(np.uint8)
                if has_prostate_mask else None
            )
            if prostate_mask is not None and prostate_mask.sum() == 0:
                # The source mask existed but vanished under resize+threshold
                # to the 64x64 patch grid (thin/small annotation). Falling
                # through to "prostate_mask=0 everywhere" would make the
                # negative region the WHOLE grid, including the needle's own
                # cancerous patches -- silently mislabeling them known-
                # negative. Treat this exactly like "no prostate mask".
                prostate_mask = None
                has_prostate_mask = False
                n_degenerate_prostate_mask += 1
            if prostate_mask is None:
                # No usable prostate boundary for this core: treat every
                # patch as "inside the prostate" (region_cache.py's negative
                # region -- everything outside the prostate mask -- is then
                # empty for this core, so it contributes 0 to the negative
                # loss term and to the out-of-prostate-activation metric,
                # rather than silently fabricating a boundary).
                prostate_mask = np.ones_like(needle_mask)
                n_no_prostate_mask += 1

            feats = pnf_plus.frozen_encoder_forward(model, bmode)  # (1, D, 64, 64)
            feats = feats.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.float16)  # (64, 64, D)

            np.savez_compressed(
                npz_path,
                feats=feats,
                needle_mask=needle_mask,
                prostate_mask=prostate_mask,
            )

            row = meta_table.loc[ref.core_id]
            writer.writerow(dict(
                core_id=ref.core_id,
                patient_id=row["patient_id"],
                center=row["center"],
                grade_group=row["grade_group"],
                involvement=row["involvement"],
                pri_mus=row["pri_mus"],
                age=row["age"],
                psa=row["psa"],
                n_prostate_patches=int(prostate_mask.sum()),
                n_needle_patches=int(needle_mask.sum()),
                has_prostate_mask=has_prostate_mask,
            ))
            metadata_file.flush()
            n_ok += 1
        except Exception as e:
            n_err += 1
            print(f"      [ERROR] {ref.core_id}: {e}", flush=True)

        if (i + 1) % max(1, len(cores) // 20) == 0 or (i + 1) == len(cores):
            elapsed = time.time() - t_extract0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            print(f"      {i+1}/{len(cores)} ({rate:.2f} cores/s, "
                  f"ok={n_ok} skip={n_skip} err={n_err} no_prostate_mask={n_no_prostate_mask} "
                  f"degenerate_prostate_mask={n_degenerate_prostate_mask})", flush=True)

    metadata_file.close()
    print(f"done: {n_ok} ok, {n_skip} skipped (already cached), {n_err} errors, "
          f"{n_no_prostate_mask} missing a prostate mask, {n_degenerate_prostate_mask} "
          f"with a mask that vanished at 64x64 resolution. metadata -> {metadata_path}", flush=True)


if __name__ == "__main__":
    main()
