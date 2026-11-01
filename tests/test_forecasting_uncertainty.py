"""Small, offline reference checks for paired temporal uncertainty."""

import json

import numpy as np
import pytest

from forecasting.uncertainty import block_bootstrap, block_indices


def test_contiguous_nonwrapping_blocks_and_truncated_last_block():
    indices = block_indices(7, 3, 200, rng=np.random.default_rng(11))
    assert indices.shape == (200, 7)
    assert indices.min() == 0
    assert indices.max() == 6
    for offset in (0, 3):
        assert np.all(np.diff(indices[:, offset:offset+3], axis=1) == 1)
        assert np.all(indices[:, offset] <= 4)
    # The final single observation is still the START of a full valid block.
    assert np.all(indices[:, 6] <= 4)


def test_seed_determinism_batch_independence_and_global_rng_untouched():
    sample = np.arange(35.0).reshape(7, 5)
    np.random.seed(27)
    expected_global = np.random.random(2)
    np.random.seed(27)
    first = block_bootstrap(sample, block_length=3, replicates=101, batch_size=7)
    assert block_bootstrap(sample, block_length=3, replicates=101, batch_size=31) == first
    np.testing.assert_array_equal(np.random.random(2), expected_global)
    json.dumps(first, allow_nan=False)


def test_shared_sampling_manual_reference_and_paired_sign():
    # One model has exactly 2 fewer loss units at every date.
    second = np.array([1., 3., 2., 8., 4., 5., 9.])
    first = second - 2
    differences = first - second
    values = np.column_stack((first, second, differences))
    result = block_bootstrap(values, block_length=3, replicates=97, seed=17)
    indices = block_indices(7, 3, 97, rng=np.random.default_rng(17))
    means = values[indices].mean(axis=1)
    np.testing.assert_allclose(result["mean"], values.mean(axis=0))
    np.testing.assert_allclose(result["lower"], np.quantile(means, .025, axis=0))
    np.testing.assert_allclose(result["upper"], np.quantile(means, .975, axis=0))
    assert result["lower"][2] == result["upper"][2] == -2
    assert result["lower"][1] - result["lower"][0] == pytest.approx(2)
    assert result["upper"][1] - result["upper"][0] == pytest.approx(2)


@pytest.mark.parametrize("constant", [0., 3.5, -2., np.finfo(float).max])
def test_constant_difference_has_exact_degenerate_interval(constant):
    result = block_bootstrap(np.full(13, constant), block_length=5, replicates=50)
    assert result["mean"] == result["lower"] == result["upper"] == [constant]
    assert result["blocks_per_replicate"] == 3
    assert result["candidate_blocks"] == 9
    assert result["warning"]


def test_block_as_long_as_sample_and_single_observation():
    result = block_bootstrap([1., 3., 5.], block_length=3, replicates=20)
    assert result["mean"] == result["lower"] == result["upper"] == [3.]
    assert result["warning"]
    single = block_bootstrap([4.], block_length=1, replicates=20)
    assert single["lower"] == single["upper"] == [4.]


def test_indicator_rates_and_warning_threshold():
    sample = np.tile([False, False, False, True], 25)
    result = block_bootstrap(sample, block_length=10, replicates=100)
    assert result["mean"] == [.25]
    assert 0 <= result["lower"][0] <= result["upper"][0] <= 1
    assert result["warning"] is None


@pytest.mark.parametrize("values", [[], [[]], [1, np.nan], [np.inf],
                                       [1j], "hello", 3., [[[1.]]]])
def test_invalid_samples(values):
    with pytest.raises(ValueError):
        block_bootstrap(values, block_length=1)


@pytest.mark.parametrize("options", [dict(block_length=0), dict(block_length=4),
    dict(block_length=True), dict(block_length=1.0), dict(replicates=1),
    dict(replicates=False), dict(seed=-1), dict(seed=1.5), dict(batch_size=0)])
def test_invalid_configuration(options):
    config = dict(block_length=1)
    config.update(options)
    with pytest.raises(ValueError):
        block_bootstrap([1., 2., 3.], **config)


def test_large_signed_values_remain_finite_without_overflow():
    large = np.finfo(float).max
    result = block_bootstrap([large, -large, large, -large],
                             block_length=1, replicates=30)
    assert result["mean"] == [0.]
    assert np.isfinite(result["lower"] + result["upper"]).all()
