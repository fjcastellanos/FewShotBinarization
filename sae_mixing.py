"""Optional, batch-level augmentation for the original Keras SAE.

The SAE encodes unavailable supervision as ``-1`` in its *input*.
No change to the historical loss, architecture or validation generator is needed.
"""
import numpy as np

MASK_VALUE = -1.0


def augment_batch(x, y, method='none', mixup_min=0.25, mixup_max=0.75, rng=None):
    """Return augmented (x, y) without modifying the original arrays.

    MixUp mixes two *different* members of the batch where both inputs are
    unmasked. Elsewhere the input is masked (-1), preventing unsupported
    supervision. CutMix constructs each sample from four equal quadrants,
    preserving aligned x/y and the original masking sentinel.
    """
    if method == 'none':
        return x, y
    if method not in ('mixup', 'cutmix'):
        raise ValueError('Unknown SAE mixing method: %s' % method)
    # The historical SAE generator returns X as (N,H,W), Y as (N,H,W,1).
    # Accept both that format and aligned 4D tensors without altering its API.
    restore_x = (x.ndim == 3)
    if restore_x:
        x = x[..., np.newaxis]
    if x.ndim != 4 or y.ndim != 4 or x.shape[:3] != y.shape[:3]:
        raise ValueError('Expected spatially aligned X=(N,H,W,Cx) and Y=(N,H,W,Cy); got %s and %s' % (x.shape, y.shape))
    if x.shape[0] == 0:
        return x, y
    rng = np.random if rng is None else rng
    n, h, w, _ = x.shape
    if method == 'mixup':
        if not 0 <= mixup_min <= mixup_max <= 1:
            raise ValueError('MixUp bounds must satisfy 0 <= min <= max <= 1')
        if n < 2:
            return (x[..., 0] if restore_x else x), y  # no independent donor in this batch
        # Nonzero cyclic shift guarantees the donor is a different batch member.
        shift = int(rng.randint(1, n))
        donors = np.roll(np.arange(n), shift)
        weights = rng.uniform(mixup_min, mixup_max, size=(n, 1, 1, 1))
        # Require all input channels to be available at each supervised pixel.
        # Shape (N,H,W,1) broadcasts across Cx for the input and Cy for labels.
        valid = np.all(x != MASK_VALUE, axis=-1, keepdims=True) & np.all(x[donors] != MASK_VALUE, axis=-1, keepdims=True)
        mixed_x = weights * x + (1 - weights) * x[donors]
        mixed_y = weights * y + (1 - weights) * y[donors]
        mixed_x = np.where(valid, mixed_x, MASK_VALUE).astype(x.dtype)
        mixed_y = np.where(valid, mixed_y, 0).astype(y.dtype)
        return (mixed_x[..., 0] if restore_x else mixed_x), mixed_y

    if h % 2 or w % 2:
        raise ValueError('CutMix 2x2 requires even patch height and width')
    result_x = np.empty_like(x)
    result_y = np.empty_like(y)
    locations = [(slice(0, h // 2), slice(0, w // 2)),
                 (slice(0, h // 2), slice(w // 2, w)),
                 (slice(h // 2, h), slice(0, w // 2)),
                 (slice(h // 2, h), slice(w // 2, w))]
    for i in range(n):
        # Four distinct source locations even when batch_size=1.
        source_quadrants = rng.permutation(4)
        if np.array_equal(source_quadrants, np.arange(4)):
            source_quadrants = np.roll(source_quadrants, 1)
        # With more than one example, independently draw donor image per tile.
        donor_indices = rng.randint(0, n, size=4) if n > 1 else np.zeros(4, dtype=int)
        for dest, src_q, donor in zip(locations, source_quadrants, donor_indices):
            src = locations[int(src_q)]
            result_x[i, dest[0], dest[1], :] = x[int(donor), src[0], src[1], :]
            result_y[i, dest[0], dest[1], :] = y[int(donor), src[0], src[1], :]
    return (result_x[..., 0] if restore_x else result_x), result_y


def augment_generator(generator, method='none', mixup_min=0.25, mixup_max=0.75):
    """Decorate the *training* generator only."""
    for x, y in generator:
        yield augment_batch(x, y, method, mixup_min, mixup_max)
