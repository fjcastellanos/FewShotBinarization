import numpy as np
import pytest
from sae_mixing import augment_batch


def test_mixup_only_intersection_and_no_mutation():
    x = np.array([[[[0.2], [-1.]], [[0.6], [0.1]]],
                  [[[0.8], [0.4]], [[-1.], [0.9]]]], dtype='float32')
    y = np.array([[[[1.], [0.]], [[0.], [1.]]],
                  [[[0.], [1.]], [[0.], [0.]]]], dtype='float32')
    original = x.copy()
    a, b = augment_batch(x, y, 'mixup', 0.5, 0.5, np.random.RandomState(3))
    assert a[0, 0, 0, 0] == pytest.approx(0.5)
    assert b[0, 0, 0, 0] == pytest.approx(0.5)
    assert a[0, 0, 1, 0] == -1
    assert a[0, 1, 0, 0] == -1
    np.testing.assert_array_equal(x, original)


def test_cutmix_single_sample_preserves_pixels_and_alignment():
    x = np.arange(16, dtype='float32').reshape(1, 4, 4, 1)
    y = x + 100
    a, b = augment_batch(x, y, 'cutmix', rng=np.random.RandomState(5))
    assert sorted(a.flatten()) == sorted(x.flatten())
    np.testing.assert_array_equal(b, a + 100)
    assert not np.array_equal(a, x)


def test_none_returns_originals():
    x = np.zeros((1, 2, 2, 1))
    y = x.copy()
    a, b = augment_batch(x, y)
    assert a is x and b is y
