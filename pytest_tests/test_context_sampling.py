import ast
import random
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modern_fewshot.context_sampling import (  # noqa: E402
    centered_context_sample,
    output_masking_sample,
    random_block_keep_mask,
    validate_context_range,
    validate_mask_range,
)
import sae_sampling  # noqa: E402


def make_arrays(rows=600, cols=700, top=200, left=300, size=256):
    rng = np.random.RandomState(0)
    image = rng.randint(0, 256, size=(rows, cols, 3)).astype(np.uint8)
    valid = np.zeros((rows, cols), dtype=np.uint8)
    valid[top:top + size, left:left + size] = 1
    target = (rng.rand(rows, cols) > 0.8).astype(np.uint8) * valid
    return image, target, valid


def test_centered_context_scale_one_is_exactly_the_patch():
    image, target, valid = make_arrays()
    i, t, v = centered_context_sample(image, target, valid, 200, 300, 256, 256, 1.0, 255)
    np.testing.assert_array_equal(i, image[200:456, 300:556])
    np.testing.assert_array_equal(t, target[200:456, 300:556])
    assert v.all()


def test_centered_context_larger_crop_is_centred_resized_and_context_unsupervised():
    image, target, valid = make_arrays()
    i, t, v = centered_context_sample(image, target, valid, 200, 300, 256, 256, 2.0, 255)
    assert i.shape == (256, 256, 3) and t.shape == (256, 256) and v.shape == (256, 256)
    # The annotated patch occupies the central half of the output, context is unsupervised.
    assert v[64:192, 64:192].all()
    assert v.sum() == 128 * 128
    assert not v[:60].any() and not v[:, :60].any() and not v[200:].any() and not v[:, 200:].any()
    assert set(np.unique(t)) <= {0, 1}
    assert not np.any(t[v == 0])
    # Real image context, not a masked sentinel.
    expected = image[72:584, 172:684]
    assert i[:60].mean() == pytest.approx(expected[:120].mean(), abs=3)


def test_centered_context_pads_outside_page_without_supervision():
    image, target, valid = make_arrays(rows=256, cols=256, top=0, left=0)
    i, t, v = centered_context_sample(image, target, valid, 0, 0, 256, 256, 2.0, 255)
    assert v.sum() == 128 * 128
    assert (i[:50, :50] == 255).all()


def test_output_masking_keeps_input_and_masks_requested_fraction():
    image, target, valid = make_arrays()
    rng = random.Random(4)
    i, t, v = output_masking_sample(image, target, valid, 200, 300, 256, 256, 0.5, 32, rng, 255)
    np.testing.assert_array_equal(i, image[200:456, 300:556])
    assert v.mean() == pytest.approx(0.5)
    assert not np.any(t[v == 0])
    np.testing.assert_array_equal(t[v == 1], target[200:456, 300:556][v == 1])


@pytest.mark.parametrize("fraction,expected", [(0.0, 1.0), (0.95, 3.0 / 64.0)])
def test_output_masking_range_endpoints(fraction, expected):
    keep = random_block_keep_mask(256, 256, fraction, 32, random.Random(1))
    assert keep.mean() == pytest.approx(expected)


def test_block_mask_handles_non_divisible_sizes():
    keep = random_block_keep_mask(100, 70, 0.5, 32, random.Random(2))
    assert keep.shape == (100, 70)
    assert 0 < keep.mean() < 1


def test_invalid_ranges_raise():
    with pytest.raises(ValueError):
        validate_context_range(0.5, 2.0)
    with pytest.raises(ValueError):
        validate_context_range(2.0, 1.5)
    with pytest.raises(ValueError):
        validate_mask_range(0.0, 1.0, 32)
    with pytest.raises(ValueError):
        validate_mask_range(0.0, 0.95, 0)


# --------------------------------------------------------------------------- modern
def _modern_page():
    from modern_fewshot.data import PartialPage, build_legacy_annotation_mask

    image = np.full((768, 768, 3), 255, dtype=np.uint8)
    image[300:400, 300:500] = 30
    gt = np.zeros((768, 768), dtype=np.uint8)
    gt[300:400, 300:500] = 1
    valid, windows = build_legacy_annotation_mask(gt, 256, 256, 1, 0.001)
    return PartialPage("src", "gt", image, gt, valid, gt * valid, windows)


@pytest.mark.parametrize("sampling", ["centered_context", "output_masking"])
def test_modern_alternative_samplers_shapes_budget_and_reproducibility(sampling):
    from modern_fewshot.data import PartialPatchDataset

    page = _modern_page()
    ds = PartialPatchDataset([page], 256, 256, 64, False, 0.001, ("none",), seed=3, sampling=sampling)
    ds2 = PartialPatchDataset([page], 256, 256, 64, False, 0.001, ("none",), seed=3, sampling=sampling)
    assert len(ds) == 64
    for idx in range(8):
        image, target, valid = ds[idx]
        image2, target2, valid2 = ds2[idx]
        assert image.shape == (3, 256, 256) and target.shape == (256, 256) and valid.shape == (256, 256)
        assert int(valid.sum()) > 0
        assert not bool(target[~valid].any())
        assert bool((image == image2).all() and (target == target2).all() and (valid == valid2).all())


def test_modern_output_masking_varies_fraction_and_never_masks_input():
    from modern_fewshot.data import PartialPatchDataset

    page = _modern_page()
    top, left, _, _ = page.annotated_windows[0]
    ds = PartialPatchDataset([page], 256, 256, 64, False, 0.001, ("none",), seed=3, sampling="output_masking")
    expected = page.image[top:top + 256, left:left + 256].transpose(2, 0, 1) / 255.0
    ratios = []
    for idx in range(32):
        image, _, valid = ds[idx]
        np.testing.assert_allclose(image.numpy(), expected, atol=1e-6)
        ratios.append(float(valid.float().mean()))
    assert min(ratios) >= 0.04 and max(ratios) <= 1.0 and max(ratios) - min(ratios) > 0.3


def test_modern_centered_context_varies_supervised_area():
    from modern_fewshot.data import PartialPatchDataset

    page = _modern_page()
    ds = PartialPatchDataset([page], 256, 256, 64, False, 0.001, ("none",), seed=3, sampling="centered_context")
    areas = [float(ds[i][2].float().mean()) for i in range(32)]
    assert min(areas) >= 0.25 - 1e-6 and max(areas) <= 1.0
    assert max(areas) - min(areas) > 0.2


def test_modern_default_sampling_is_unchanged_by_new_options():
    from modern_fewshot.data import PartialPatchDataset

    page = _modern_page()
    a = PartialPatchDataset([page], 256, 256, 16, True, 0.001, ("none",), seed=5)
    b = PartialPatchDataset([page], 256, 256, 16, True, 0.001, ("none",), seed=5, sampling="default",
                            context_min=1.5, mask_max=0.5)
    for idx in range(16):
        for x, y in zip(a[idx], b[idx]):
            assert bool((x == y).all())


# --------------------------------------------------------------------------- SAE
def _legacy_calculate_mask():
    """Load util.calculate_mask without importing TensorFlow."""
    source = (ROOT / "util.py").read_text(encoding="utf-8")
    node = [n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "calculate_mask"][0]
    namespace = {"np": np}
    exec(compile(ast.Module([node], []), "util.py", "exec"), namespace)
    return namespace["calculate_mask"]


@pytest.mark.parametrize("nb,rate", [(-1, 0.0025), (1, 0.02), (3, 0.0025), (32, 0.0)])
def test_sae_windows_reproduce_legacy_calculate_mask(nb, rate):
    calculate_mask = _legacy_calculate_mask()
    rng = np.random.RandomState(7)
    gt = (rng.rand(700, 900) > 0.995).astype(int)
    gt[0:300, 0:300] = (rng.rand(300, 300) > 0.9)
    mask, count = calculate_mask(gt, 256, 256, nb, rate)
    windows = sae_sampling.annotated_windows(gt, 256, 256, nb, rate)
    assert len(windows) == count
    rebuilt = np.zeros_like(mask)
    for top, left in windows:
        rebuilt[top:top + 256, left:left + 256] = 1
    np.testing.assert_array_equal(rebuilt, mask)


def _sae_page():
    rng = np.random.RandomState(3)
    image = rng.rand(600, 700, 3)
    valid = np.zeros((600, 700), dtype=np.uint8)
    valid[0:256, 256:512] = 1
    target = (rng.rand(600, 700) > 0.7).astype(np.uint8) * valid
    return {"image": image, "target": target, "valid": valid, "windows": [(0, 256)]}


@pytest.mark.parametrize("strategy", ["centered_context", "output_masking"])
def test_sae_train_batch_has_unmasked_input_and_pixel_weights(strategy):
    x, y, w = sae_sampling.sample_batch([_sae_page()], 8, strategy, 256, 256, random.Random(1))
    assert x.shape == (8, 256, 256, 3) and y.shape == (8, 256, 256, 1) and w.shape == (8, 256, 256)
    assert not np.any(x == -1)
    assert set(np.unique(w)) <= {0.0, 1.0} and w.sum() > 0
    assert not np.any(y[..., 0][w == 0])


def test_sae_validation_batch_uses_regions_as_weights():
    x, y, w = sae_sampling.validation_batch([_sae_page()], 4, 256, 256, 0.02, random.Random(1))
    assert x.shape == (4, 256, 256, 3) and y.shape == (4, 256, 256, 1) and w.shape == (4, 256, 256)
    assert not np.any(x == -1)
    assert w.sum() > 0 and not np.any(y[..., 0][w == 0])
