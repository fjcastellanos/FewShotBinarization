"""Optional, batch-level partial-label MixUp and four-quadrant CutMix.

The default training path never calls these functions. All randomness uses an
independent Python RNG, avoiding changes to legacy training RNG state.
"""
from __future__ import annotations

import random
import torch
import torch.nn.functional as F


def mixup_batch(image, target, valid, rng: random.Random, low=0.25, high=0.75):
    """Mix each image with a batch partner; return both supervision sources.

    The mixed input may contain unlabelled content; loss below is a weighted
    partial-label objective, NOT a claim that unknown labels are background.
    """
    if not 0 <= low <= high <= 1:
        raise ValueError("MixUp bounds must satisfy 0 <= low <= high <= 1")
    batch = image.shape[0]
    if batch < 2:
        # Sampling a second item is essential; self-mixing is a no-op.
        return image, target, valid, target, valid, torch.ones(batch, device=image.device)
    indices = list(range(batch))
    # Derangement: no sample is paired with itself.
    offset = rng.randrange(1, batch)
    partner = [(i + offset) % batch for i in indices]
    lam = torch.tensor([rng.uniform(low, high) for _ in indices],
                       device=image.device, dtype=image.dtype)
    mixed = lam[:, None, None, None] * image + (1 - lam[:, None, None, None]) * image[partner]
    return mixed, target, valid, target[partner], valid[partner], lam


def mixup_partial_loss(logits, target_a, valid_a, target_b, valid_b, lam, use_masking=True):
    """Weighted CE over *available* source labels; preserves partial masks.

    For unmasked baseline, retain its existing unknown-as-background semantics.
    """
    ce_a = F.cross_entropy(logits, target_a.long(), reduction="none")
    ce_b = F.cross_entropy(logits, target_b.long(), reduction="none")
    wa = lam[:, None, None].to(dtype=ce_a.dtype)
    wb = 1 - wa
    if use_masking:
        wa = wa * valid_a.to(dtype=wa.dtype)
        wb = wb * valid_b.to(dtype=wb.dtype)
    denom = (wa + wb).sum()
    if denom.item() == 0:
        return logits.sum() * 0.0
    return ((wa * ce_a) + (wb * ce_b)).sum() / denom


def cutmix_quadrants(image, target, valid, rng: random.Random):
    """Shuffle the four aligned quadrants among batch samples.

    On batch size 1, shuffle its own four quadrants. Each input pixel, target
    and validity bit moves together. Per-item fallback prevents empty masks.
    """
    b, _, h, w = image.shape
    if h < 2 or w < 2:
        raise ValueError("CutMix requires patch height and width >= 2")
    sections = [(slice(0, h // 2), slice(0, w // 2)),
                (slice(0, h // 2), slice(w // 2, w)),
                (slice(h // 2, h), slice(0, w // 2)),
                (slice(h // 2, h), slice(w // 2, w))]
    out_i, out_t, out_v = image.clone(), target.clone(), valid.clone()
    for i in range(b):
        # Sampling whole batch quadrants; always move image/GT/mask together.
        if b == 1:
            sources = list(range(4))
            rng.shuffle(sources)
            if sources == list(range(4)):
                sources = [1, 0, 3, 2]
            assignments = [(0, q) for q in sources]
        else:
            assignments = [(rng.randrange(b), rng.randrange(4)) for _ in range(4)]
        for q, (src, src_q) in enumerate(assignments):
            dst_r, dst_c = sections[q]
            src_r, src_c = sections[src_q]
            # Handles odd dimensions by resizing/cropping quadrant to fit.
            hh, ww = dst_r.stop-dst_r.start, dst_c.stop-dst_c.start
            for source, dest in ((image, out_i), (target, out_t), (valid, out_v)):
                tile = source[src, ..., src_r, src_c]
                if tile.shape[-2:] != (hh, ww):
                    tile = F.interpolate(tile[None, None].float() if tile.ndim == 2 else tile[None].float(),
                                         size=(hh, ww), mode="nearest")[0]
                    if source.ndim == 3:
                        tile = tile[0]
                    tile = tile.to(dtype=source.dtype)
                dest[i, ..., dst_r, dst_c] = tile
        if valid[i].any() and not out_v[i].any():
            out_i[i], out_t[i], out_v[i] = image[i], target[i], valid[i]
    return out_i, out_t, out_v
