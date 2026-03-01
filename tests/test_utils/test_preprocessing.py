"""Tests for fourdvarjax._src.utils.preprocessing."""

import numpy as np
import pytest

from fourdvarjax import Batch1D
from fourdvarjax._src.utils.masks import random_mask
from fourdvarjax._src.utils.noise import add_gaussian_noise
from fourdvarjax._src.utils.patches import extract_patches, trajectory_to_xr_dataset
from fourdvarjax._src.utils.preprocessing import (
    interpolate_initial_condition,
    train_test_split,
    xr_to_batch1d,
)


def _make_full_ds(n_patches=30, n_timesteps=20, n_features=3, seed=0):
    rng = np.random.default_rng(seed)
    states = rng.standard_normal((500, n_features)).astype(np.float32)
    time_coords = np.arange(500, dtype=float)
    ds = trajectory_to_xr_dataset(states, time_coords)
    ds = extract_patches(ds, n_patches=n_patches, n_timesteps=n_timesteps, seed=seed)
    ds = random_mask(ds, missing_rate=0.3, seed=seed)
    ds = add_gaussian_noise(ds, sigma=0.1, seed=seed)
    return ds


class TestTrainTestSplit:
    def test_correct_counts(self):
        ds = _make_full_ds()
        ds_train, ds_test = train_test_split(ds, n_train=20, n_test=8)
        assert ds_train.sizes["patch"] == 20
        assert ds_test.sizes["patch"] == 8

    def test_non_overlapping(self):
        ds = _make_full_ds()
        ds_train, ds_test = train_test_split(ds, n_train=20, n_test=8)
        train_patches = set(ds_train.coords["patch"].values.tolist())
        test_patches = set(ds_test.coords["patch"].values.tolist())
        assert train_patches.isdisjoint(test_patches)

    def test_too_many_raises(self):
        ds = _make_full_ds(n_patches=10)
        with pytest.raises(ValueError):
            train_test_split(ds, n_train=8, n_test=5)

    def test_deterministic(self):
        ds = _make_full_ds()
        t1, _e1 = train_test_split(ds, n_train=20, n_test=8, seed=0)
        t2, _e2 = train_test_split(ds, n_train=20, n_test=8, seed=0)
        np.testing.assert_array_equal(t1["state"].values, t2["state"].values)


class TestXrToBatch1d:
    def test_returns_batch1d(self):
        ds = _make_full_ds()
        batch = xr_to_batch1d(ds)
        assert isinstance(batch, Batch1D)

    def test_shapes(self):
        n_patches, n_timesteps, n_features = 10, 20, 3
        ds = _make_full_ds(
            n_patches=n_patches, n_timesteps=n_timesteps, n_features=n_features
        )
        batch = xr_to_batch1d(ds)
        assert batch.input.shape == (n_patches, n_timesteps, n_features)
        assert batch.mask.shape == (n_patches, n_timesteps, n_features)
        assert batch.target.shape == (n_patches, n_timesteps, n_features)

    def test_jax_arrays(self):
        ds = _make_full_ds()
        batch = xr_to_batch1d(ds)
        # Check all fields are JAX arrays
        assert hasattr(batch.input, "device")
        assert hasattr(batch.mask, "device")
        assert hasattr(batch.target, "device")


class TestInterpolateInitialCondition:
    def test_x_init_exists(self):
        ds = _make_full_ds()
        result = interpolate_initial_condition(ds)
        assert "x_init" in result

    def test_x_init_shape(self):
        ds = _make_full_ds()
        result = interpolate_initial_condition(ds)
        assert result["x_init"].shape == ds["obs"].shape

    def test_fills_gaps(self):
        """Observed points should be reproduced; gaps should be filled."""
        ds = _make_full_ds()
        result = interpolate_initial_condition(ds, fillna=0.0)
        x_init = result["x_init"].values
        # x_init should not be entirely zero where mask is 1
        mask = ds["mask"].values.astype(bool)
        obs = ds["obs"].values
        # At observed positions, x_init ≈ obs
        np.testing.assert_allclose(x_init[mask], obs[mask], atol=1e-5)
