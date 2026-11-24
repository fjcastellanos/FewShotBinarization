"""Optional alternative samplers for the original Keras SAE (reviewer request).

They replace -- never combine with -- annotation-guided oversampling:

* ``centered_context``: fixed annotated patch with a random larger centred
  context (1x-2x the patch side by default), resized to the window size.
* ``output_masking``: fixed annotated patch with a random fraction (0-95% by
  default) of its labels removed in square blocks.

In the historical SAE, unsupervised pixels are hidden by setting the *input*
to -1. Here the input image is always shown unmasked and unsupervised pixels
are excluded from the loss with a per-pixel Keras ``sample_weight`` of 0, so
the network actually sees the extra context / masked zones. Batches are
therefore ``(x, y, sample_weight)``. The validation generator follows the same
convention, and the model is recompiled with a *weighted* ``mse`` metric so
that ``val_mse`` (checkpointing / early stopping) only measures annotated
pixels. Nothing here is used unless ``--sae_sampling`` is given.

The sampling itself is shared with the modern pipeline
(``modern_fewshot/context_sampling.py``).
"""
import random

import numpy as np

from modern_fewshot.context_sampling import (
    SAMPLING_CHOICES,
    SAMPLING_DEFAULT,
    crop_with_padding,
    sample_annotated_patch,
    validate_context_range,
    validate_mask_range,
)

# Normalized SAE input: (255 - img) / 255, so white paper (padding) is 0.
SAE_PAD_VALUE = 0.0
# Same constants as util.extractRandomSamplesClass.
LEGACY_ROW_OFFSET = 100
LEGACY_COL_OFFSET = 50
LEGACY_MAX_TRIES = 100


def annotated_windows(gt, window_rows, window_cols, nb_sequential_patches=-1, min_rate_annotated_pixels=0.0025):
    """Top-left corners of the simulated annotated windows.

    Exact replica of the loop in ``util.calculate_mask`` (same order, same
    acceptance rule), returning the windows instead of only their union.
    """
    rows_img, cols_img = gt.shape[0], gt.shape[1]
    windows = []
    for row in range(window_rows // 2, rows_img + window_rows // 2 - 1, window_rows):
        for col in range(window_cols // 2, cols_img + window_cols // 2 - 1, window_cols):
            row = min(row, rows_img - window_rows // 2)
            col = min(col, cols_img - window_cols // 2)
            top = row - window_rows // 2
            left = col - window_cols // 2
            sample = gt[top:top + window_rows, left:left + window_cols]
            if np.sum(sample == 1) > 0:
                rate = np.sum(sample == 1) / (window_cols * window_rows)
                if nb_sequential_patches == -1 or rate >= min_rate_annotated_pixels:
                    windows.append((top, left))
                    if nb_sequential_patches not in (-1, 0) and len(windows) >= nb_sequential_patches:
                        return windows
    return windows


def load_page(page, window_rows, window_cols, nb_annotated_patches, min_rate_annotated_pixels):
    """Load one (src, gt) page: unmasked normalized image, partial GT, region mask, windows."""
    import util
    import utilIO

    gr, gt, regions_mask, _ = util.get_image_with_gt(
        page[0], page[1], nb_annotated_patches, window_rows, window_cols,
        min_rate_annotated_pixels, with_mask=False)
    full_gt = (utilIO.load_gt_image(page[1]) < 128) * 1
    windows = annotated_windows(full_gt, window_rows, window_cols, nb_annotated_patches, min_rate_annotated_pixels)
    check = np.zeros(regions_mask.shape, dtype=bool)
    for top, left in windows:
        check[top:top + window_rows, left:left + window_cols] = True
    if not np.array_equal(check, regions_mask > 0):
        raise AssertionError('Annotated windows do not reproduce util.calculate_mask for %s' % page[1])
    return {
        'image': gr,
        'target': (gt > 0).astype(np.uint8),
        'valid': (regions_mask > 0).astype(np.uint8),
        'windows': windows,
    }


def _to_batch(samples):
    x = np.array([s[0] for s in samples])
    y = np.array([s[1] for s in samples], dtype=np.float32)[..., np.newaxis]
    w = np.array([s[2] for s in samples], dtype=np.float32)
    return x, y, w


def sample_batch(pages, batch_size, strategy, window_rows, window_cols, rng,
                 context_min=1.0, context_max=2.0, mask_min=0.0, mask_max=0.95, mask_block=32):
    """Batch of alternative samples drawn uniformly over all annotated windows."""
    candidates = [(p, w) for p in pages for w in p['windows']]
    if not candidates:
        raise ValueError('No annotated patch satisfies the requested ink rate')
    samples = []
    for _ in range(batch_size):
        page, (top, left) = candidates[rng.randrange(len(candidates))]
        samples.append(sample_annotated_patch(
            strategy, page['image'], page['target'], page['valid'], top, left,
            window_rows, window_cols, rng, image_fill=SAE_PAD_VALUE,
            context_min=context_min, context_max=context_max,
            mask_min=mask_min, mask_max=mask_max, mask_block=mask_block))
    return _to_batch(samples)


def oversampling_crop(page, window_rows, window_cols, min_rate_annotated_pixels, rng):
    """Crop location of the historical oversampling (util.extractRandomSamplesClass)."""
    target = page['target']
    ys, xs = np.where(target == 1)
    rows_img, cols_img = target.shape
    top = left = 0
    tries = 0
    while True:
        j = rng.randrange(len(ys))
        row = max(window_rows // 2 + 1, int(ys[j]) - LEGACY_ROW_OFFSET)
        col = max(window_cols // 2 + 1, int(xs[j]) - LEGACY_COL_OFFSET)
        row = min(rows_img - window_rows // 2 - 1, row)
        col = min(cols_img - window_cols // 2 - 1, col)
        top = row - window_rows // 2
        left = col - window_cols // 2
        crop = crop_with_padding(target, top, left, window_rows, window_cols, fill=0)
        rate = np.sum(crop == 1) / float(window_rows * window_cols)
        if rate >= min_rate_annotated_pixels or tries > LEGACY_MAX_TRIES:
            return top, left
        tries += 1


def validation_batch(pages, batch_size, window_rows, window_cols, min_rate_annotated_pixels, rng):
    """Historical oversampling crops, unmasked input, weight = annotated regions."""
    eligible = [p for p in pages if np.any(p['target'] == 1)]
    if not eligible:
        raise ValueError('No annotated validation pixels')
    page = eligible[rng.randrange(len(eligible))]
    samples = []
    for _ in range(batch_size):
        top, left = oversampling_crop(page, window_rows, window_cols, min_rate_annotated_pixels, rng)
        samples.append((
            crop_with_padding(page['image'], top, left, window_rows, window_cols, fill=SAE_PAD_VALUE),
            crop_with_padding(page['target'], top, left, window_rows, window_cols, fill=0),
            crop_with_padding(page['valid'], top, left, window_rows, window_cols, fill=0),
        ))
    return _to_batch(samples)


def create_train_generator(data_pages, batch_size, window_shape, nb_annotated_patches, min_rate_annotated_pixels,
                           strategy, context_min=1.0, context_max=2.0, mask_min=0.0, mask_max=0.95,
                           mask_block=32, rng=None):
    """Infinite (x, y, sample_weight) generator. Pages are loaded lazily on first use."""
    if strategy == SAMPLING_DEFAULT or strategy not in SAMPLING_CHOICES:
        raise ValueError('Invalid SAE alternative sampling: %s' % strategy)
    rng = random.Random() if rng is None else rng
    rows, cols = window_shape[0], window_shape[1]
    pages = [load_page(p, rows, cols, nb_annotated_patches, min_rate_annotated_pixels) for p in data_pages]
    while True:
        yield sample_batch(pages, batch_size, strategy, rows, cols, rng,
                           context_min, context_max, mask_min, mask_max, mask_block)


def create_val_generator(data_pages, batch_size, window_shape, nb_annotated_patches, min_rate_annotated_pixels, rng=None):
    """Infinite (x, y, sample_weight) validation generator, identical for both strategies."""
    rng = random.Random() if rng is None else rng
    rows, cols = window_shape[0], window_shape[1]
    pages = [load_page(p, rows, cols, nb_annotated_patches, min_rate_annotated_pixels) for p in data_pages]
    while True:
        yield validation_batch(pages, batch_size, rows, cols, min_rate_annotated_pixels, rng)


def compile_with_weighted_mse(model):
    """Recompile so that 'mse' (and 'val_mse') is weighted by the per-pixel sample_weight.

    Same optimizer object and loss as CNNmodel.get_model; only the metric becomes
    weighted. The metric keeps the name 'mse' so CNNmodel.train still monitors
    'val_mse'.
    """
    import tensorflow as tf
    model.compile(optimizer=model.optimizer, loss='binary_crossentropy',
                  weighted_metrics=[tf.keras.metrics.MeanSquaredError(name='mse')])
    return model
