"""Alternative samplers to annotation-guided oversampling (reviewer request).

Both strategies always start from one annotated patch, kept fixed (no random
shift as in oversampling), and are meant to be compared *against* oversampling,
never combined with it:

* ``centered_context``: the crop is centred on the annotated patch and its side
  is ``scale`` times the patch side, with ``scale ~ U(context_min, context_max)``
  (default 1x-2x). The surrounding context is real image content, but it is not
  supervised (valid=0) unless it belongs to another annotated patch. The crop
  is resized back to the patch size so the batch size is unchanged.
* ``output_masking``: the crop is exactly the annotated patch. A random fraction
  ``r ~ U(mask_min, mask_max)`` (default 0-0.95) of its supervision is removed
  in square blocks. The input image is never modified.

Only numpy/cv2 are used so that the Keras SAE and the PyTorch pipeline share
exactly the same implementation. All randomness comes from the given ``rng``
(``random.Random`` API).
"""
from __future__ import annotations

import math

import cv2
import numpy as np

SAMPLING_DEFAULT = "default"
SAMPLING_CENTERED_CONTEXT = "centered_context"
SAMPLING_OUTPUT_MASKING = "output_masking"
SAMPLING_CHOICES = (SAMPLING_DEFAULT, SAMPLING_CENTERED_CONTEXT, SAMPLING_OUTPUT_MASKING)


def crop_with_padding(arr, top, left, h, w, fill=0):
    """Crop an array and pad to exactly (h,w) if it touches an image border."""
    out_shape = (h, w) + (() if arr.ndim == 2 else (arr.shape[2],))
    out = np.full(out_shape, fill, dtype=arr.dtype)
    src_top = max(0, top)
    src_left = max(0, left)
    src_bottom = min(arr.shape[0], top + h)
    src_right = min(arr.shape[1], left + w)
    if src_bottom <= src_top or src_right <= src_left:
        return out
    dst_top = src_top - top
    dst_left = src_left - left
    out[dst_top:dst_top + (src_bottom - src_top),
        dst_left:dst_left + (src_right - src_left)] = arr[src_top:src_bottom, src_left:src_right]
    return out


def validate_context_range(context_min, context_max):
    if not 1.0 <= context_min <= context_max:
        raise ValueError("Centered context requires 1 <= context_min <= context_max")


def validate_mask_range(mask_min, mask_max, mask_block):
    if not 0.0 <= mask_min <= mask_max < 1.0:
        raise ValueError("Output masking requires 0 <= mask_min <= mask_max < 1")
    if int(mask_block) < 1:
        raise ValueError("Output masking requires mask_block >= 1")


def centered_context_sample(image, target, valid, top, left, out_h, out_w, scale, image_fill):
    """Crop ``scale`` x the patch around patch (top,left,out_h,out_w), resize to it.

    With ``scale == 1`` the result is exactly the annotated patch. Pixels outside
    the page are padded with ``image_fill`` and are never supervised.
    """
    if scale < 1.0:
        raise ValueError("Context scale must be >= 1")
    crop_h = max(out_h, int(round(out_h * scale)))
    crop_w = max(out_w, int(round(out_w * scale)))
    crop_top = top - (crop_h - out_h) // 2
    crop_left = left - (crop_w - out_w) // 2

    image_c = crop_with_padding(image, crop_top, crop_left, crop_h, crop_w, fill=image_fill)
    target_c = crop_with_padding(np.asarray(target, dtype=np.uint8), crop_top, crop_left, crop_h, crop_w, fill=0)
    valid_c = crop_with_padding(np.asarray(valid, dtype=np.uint8), crop_top, crop_left, crop_h, crop_w, fill=0)
    if (crop_h, crop_w) != (out_h, out_w):
        image_c = cv2.resize(image_c, (out_w, out_h), interpolation=cv2.INTER_AREA)
        # Nearest keeps labels binary and target/valid aligned pixel by pixel.
        target_c = cv2.resize(target_c, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
        valid_c = cv2.resize(valid_c, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
    target_c = target_c * valid_c
    return image_c, target_c, valid_c


def random_block_keep_mask(h, w, fraction, block, rng):
    """Return uint8 (h,w) mask with ~``fraction`` of square blocks set to 0."""
    block = int(block)
    rows = int(math.ceil(h / float(block)))
    cols = int(math.ceil(w / float(block)))
    n_cells = rows * cols
    n_masked = int(round(fraction * n_cells))
    keep = np.ones((h, w), dtype=np.uint8)
    for cell in rng.sample(range(n_cells), n_masked):
        r, c = divmod(cell, cols)
        keep[r * block:(r + 1) * block, c * block:(c + 1) * block] = 0
    return keep


def output_masking_sample(image, target, valid, top, left, out_h, out_w, fraction, block, rng, image_fill):
    """Return the fixed annotated patch with a random part of its labels removed."""
    image_c = crop_with_padding(image, top, left, out_h, out_w, fill=image_fill)
    target_c = crop_with_padding(np.asarray(target, dtype=np.uint8), top, left, out_h, out_w, fill=0)
    valid_c = crop_with_padding(np.asarray(valid, dtype=np.uint8), top, left, out_h, out_w, fill=0)
    keep = random_block_keep_mask(out_h, out_w, fraction, block, rng)
    valid_c = valid_c * keep
    target_c = target_c * valid_c
    return image_c, target_c, valid_c


def sample_annotated_patch(strategy, image, target, valid, top, left, out_h, out_w, rng,
                           image_fill, context_min=1.0, context_max=2.0,
                           mask_min=0.0, mask_max=0.95, mask_block=32):
    """Draw one sample of ``strategy`` from the annotated patch at (top,left)."""
    if strategy == SAMPLING_CENTERED_CONTEXT:
        scale = rng.uniform(context_min, context_max)
        return centered_context_sample(image, target, valid, top, left, out_h, out_w, scale, image_fill)
    if strategy == SAMPLING_OUTPUT_MASKING:
        fraction = rng.uniform(mask_min, mask_max)
        return output_masking_sample(image, target, valid, top, left, out_h, out_w,
                                     fraction, mask_block, rng, image_fill)
    raise ValueError("Unknown alternative sampling strategy: %s" % strategy)
