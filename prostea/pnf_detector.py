"""The real ProstNFound+ mask-decoder head, as a drop-in replacement for
model.py::LinearProbe.

LinearProbe is a from-scratch `Conv2d(D, 1, kernel_size=1)`: cheap, but it
throws away everything ProstNFound+ actually trained -- the SAM-style mask
decoder (prompt-conditioned cross-attention, not a per-patch linear map) and
the age/psa clinical prompts entirely. This module instead wraps the real
`mask_decoder` + `prompt_encoder` + age/psa prompt MLPs from
`pnf_plus_final.pth`, warm-started from their actual trained weights, and
still cheap to retrain per genome: it never touches the frozen ViT image
encoder, only the small decoder stack, operating on the same cached patch
embeddings (prostea/region_cache.py) LinearProbe already used.

Two things `protoprostate/pnf_plus.py::load_pnf_plus` (used by
scripts/cache_regions.py to build the cache in the first place) does NOT
give us, and why this module exists separately:

1. It explicitly skips the mask decoder's weights ("N incompatible
   (mask-decoder-only) keys skipped" in cache_regions.py's own log) because
   the checkpoint was trained with `output_upscaling_version='default'` but
   medAI's current MaskDecoder defaults to `'interpolate'` -- a silent
   architecture drift, not a real incompatibility. `_load_pnf_plus_full`
   below passes `backbone_kw=dict(mask_decoder_kw=dict(
   output_upscaling_version='default'))` to get a byte-exact `strict=True`
   load, the same fix already confirmed (against the paper's own Table 2)
   in ../../prostNfound/scripts/reproduce_pnf_plus_prospective_v2.py.
2. It only returns the frozen encoder, so nothing downstream of
   `image_feats` (prompt encoding, mask decoding) was ever reachable from
   ProstEA before this module.

Checkpoint config (`pnf_plus_final.pth`'s own saved `args['model_kw']`):
`{"prompts": ["psa", "age"], "use_class_decoder": true}` -- no
`use_prostate_mask_prompt`, so the mask-decoder's dense prompt is just the
prompt_encoder's data-independent default (no prostate-mask conditioning to
replicate here); `use_class_decoder` adds a separate classification head
(`ClassDecoder`) we don't need, since ProstEA's loss operates on the
per-patch heatmap `mask_decoder` already produces.

Resolution note: cached `feats` are the encoder's native 64x64 grid (same
as needle_mask/prostate_mask, see region_cache.py), but
`output_upscaling_version='default'` upsamples the decoder's output 4x
(two stride-2 ConvTranspose2d layers), i.e. to 256x256 -- matching the
real model's native mask resolution, which the real training loss then
resizes targets *up* to match (medAI/losses/prostnfound.py interpolates
the mask to `heatmap_logits.shape`). To keep every other part of ProstEA
(loss.py, metrics.py, the cached 64x64 masks) unchanged, this module
instead resizes the decoder's output back *down* to 64x64 -- mathematically
equivalent up to interpolation, and far simpler than re-caching masks at
256x256 for every downstream consumer.
"""

from __future__ import annotations

import copy
import os

import torch
from torch import nn
from torch.nn import functional as F

DEFAULT_CHECKPOINT = "/datasets/exactvu_pca/checkpoint_store/pnf_plus_final.pth"
MEDSAM_CHECKPOINT_DIR = "/datasets/exactvu_pca/checkpoint_store"
MEDSAM_CHECKPOINT = "/datasets/exactvu_pca/checkpoint_store/sam/medsam_vit_b_cpu.pth"

# medAI/transforms/prostnfound_transform.py's own fixed min-max normalization
# constants for the age/psa prompts -- the floating_point_prompt_modules MLPs
# were trained on values normalized this way, not raw age (years) / psa
# (ng/mL). NaN falls back to the training-set average, matching that file.
PSA_MIN, PSA_MAX, PSA_AVG = 0.2, 32.95, 6.821426488456866
AGE_MIN, AGE_MAX, AGE_AVG = 0, 79, 62.5816

_TEMPLATE: dict | None = None  # lazily built once per process; see _load_template


def _normalize_age(age: torch.Tensor) -> torch.Tensor:
    age = torch.where(torch.isnan(age), torch.full_like(age, AGE_AVG), age)
    return (age - AGE_MIN) / (AGE_MAX - AGE_MIN)


def _normalize_psa(psa: torch.Tensor) -> torch.Tensor:
    psa = torch.where(torch.isnan(psa), torch.full_like(psa, PSA_AVG), psa)
    return (psa - PSA_MIN) / (PSA_MAX - PSA_MIN)


def _load_template(checkpoint_path: str = DEFAULT_CHECKPOINT, device: str = "cpu") -> dict:
    """Builds the full ProstNFound model once, with a byte-exact strict=True
    weight load, then keeps only the decoder-side submodules (everything
    downstream of the frozen image encoder) as the process-wide template
    PNFDetector.__init__ copies from. Loading the ~400MB checkpoint and
    building the full SAM backbone once per process (not once per genome)
    is what keeps retraining this head per genome cheap.
    """
    global _TEMPLATE
    if _TEMPLATE is not None:
        return _TEMPLATE

    os.environ.setdefault("MEDSAM_CHECKPOINT_DIR", MEDSAM_CHECKPOINT_DIR)
    os.environ.setdefault("MEDSAM_CHECKPOINT", MEDSAM_CHECKPOINT)
    from medAI.modeling import create_model
    from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present

    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    args = state["args"]
    model_kw = dict(args["model_kw"])
    model_kw["backbone_kw"] = dict(mask_decoder_kw=dict(output_upscaling_version="default"))
    model = create_model(args["model"], **model_kw)

    sd = state["model"]
    consume_prefix_in_state_dict_if_present(sd, "model.")
    msg = model.load_state_dict(sd, strict=True)
    if msg.missing_keys or msg.unexpected_keys:
        raise RuntimeError(f"PNF+ checkpoint did not load byte-exact: {msg}")

    _TEMPLATE = dict(
        mask_decoder=copy.deepcopy(model.medsam_model.mask_decoder).to(device).eval(),
        prompt_encoder=copy.deepcopy(model.medsam_model.prompt_encoder).to(device).eval(),
        prompt_modules=copy.deepcopy(model.floating_point_prompt_modules).to(device).eval(),
        null_prompt=model.null_prompt.detach().clone().to(device),
        # ProstNFound.forward applies these to the raw mask_decoder output
        # before it's ever treated as a logit (model.temperature/model.bias,
        # not submodule-local): mask_logits / temperature + bias. 1.0/0.0 for
        # this checkpoint (verified against the loaded state dict), i.e. a
        # no-op here, but carried over for architectural completeness.
        temperature=model.temperature.detach().clone().to(device),
        bias=model.bias.detach().clone().to(device),
    )
    return _TEMPLATE


class PNFDetector(nn.Module):
    """forward(feats, age, psa) -> (B, 64, 64) logits -- same calling
    convention as LinearProbe.forward, plus the two extra per-core prompt
    tensors. Every call builds a fresh copy of the template's weights (see
    _load_template), the same "reset to a known starting point" semantics
    train_probe already relies on for LinearProbe's random init -- except
    here the starting point is the real trained decoder, not random.
    """

    def __init__(self, device: str = "cpu"):
        super().__init__()
        tmpl = _load_template(device=device)
        self.mask_decoder = copy.deepcopy(tmpl["mask_decoder"])
        self.prompt_encoder = copy.deepcopy(tmpl["prompt_encoder"])
        self.prompt_modules = copy.deepcopy(tmpl["prompt_modules"])
        self.null_prompt = nn.Parameter(tmpl["null_prompt"].clone())
        self.register_buffer("temperature", tmpl["temperature"].clone())
        self.register_buffer("bias", tmpl["bias"].clone())

    def forward(
        self,
        feats: torch.Tensor,
        age: torch.Tensor | None = None,
        psa: torch.Tensor | None = None,
    ) -> torch.Tensor:
        single = feats.dim() == 3
        x = feats.unsqueeze(0) if single else feats  # (B, Gh, Gw, D)
        image_feats = x.permute(0, 3, 1, 2)  # (B, D, Gh, Gw)
        B = image_feats.shape[0]

        sparse_embedding, dense_embedding = self.prompt_encoder(None, None, None)
        if dense_embedding.shape[-2:] != image_feats.shape[-2:]:
            dense_embedding = F.interpolate(dense_embedding, size=image_feats.shape[-2:])
        dense_embedding = dense_embedding.repeat_interleave(B, 0)
        sparse_embedding = sparse_embedding.repeat_interleave(B, 0)

        # Checkpoint's own prompts=["psa", "age"] order (args['model_kw']);
        # the exact token order wasn't independently re-verified against the
        # original training loop's forward() call, but the decoder is being
        # fine-tuned from this starting point anyway, not used frozen.
        for name, raw_value, normalize in (("psa", psa, _normalize_psa), ("age", age, _normalize_age)):
            if raw_value is None:
                prompt_embedding = self.null_prompt.repeat_interleave(B, 0)
            else:
                value = normalize(raw_value.to(image_feats.device).float()).reshape(B, 1)
                prompt_embedding = self.prompt_modules[name](value)
            sparse_embedding = torch.cat([sparse_embedding, prompt_embedding[:, None, :]], dim=1)

        pe = self.prompt_encoder.get_dense_pe()
        if pe.shape[-2:] != image_feats.shape[-2:]:
            pe = F.interpolate(pe, size=image_feats.shape[-2:])

        mask_logits, _iou = self.mask_decoder(
            image_feats, pe, sparse_embedding, dense_embedding, multimask_output=False,
        )  # (B, 1, 4*Gh, 4*Gw) under output_upscaling_version="default"
        mask_logits = mask_logits / self.temperature[None, None, None, :] + self.bias[None, None, None, :]
        logits = F.interpolate(
            mask_logits, size=image_feats.shape[-2:], mode="bilinear", align_corners=False,
        )[:, 0]
        return logits[0] if single else logits


def build_pnf_detector(d: int, device: str = "cpu") -> PNFDetector:
    """Matches LinearProbe's `model_builder(d, device)` call signature (see
    fitness.py::train_probe) even though PNFDetector ignores `d` -- its
    embedding dim is fixed by the checkpoint (D=256), not learned from the
    cache like LinearProbe's.
    """
    return PNFDetector(device=device)
