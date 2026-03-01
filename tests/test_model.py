"""Tests for fourdvarjax._src.model."""

import jax.numpy as jnp
import pytest

from fourdvarjax import FourDVarNet1D, FourDVarNet2D


class TestFourDVarNet1D:
    def test_output_shape(self, rng, batch_1d):
        B, T, N = batch_1d.input.shape
        model = FourDVarNet1D(
            state_dim=N,
            n_time=T,
            latent_dim=8,
            hidden_dim=16,
            n_solver_steps=2,
        )
        params = model.init(rng, batch_1d)["params"]
        out = model.apply({"params": params}, batch_1d)
        assert out.shape == (B, T, N)

    @pytest.mark.slow
    def test_output_changes_with_different_masks(self, rng, batch_1d):
        from fourdvarjax import Batch1D

        _, T, N = batch_1d.input.shape
        model = FourDVarNet1D(
            state_dim=N,
            n_time=T,
            latent_dim=8,
            hidden_dim=16,
            n_solver_steps=2,
        )
        params = model.init(rng, batch_1d)["params"]
        out1 = model.apply({"params": params}, batch_1d)
        # All-zero mask
        batch_no_obs = Batch1D(
            input=batch_1d.input,
            mask=jnp.zeros_like(batch_1d.mask),
            target=batch_1d.target,
        )
        out2 = model.apply({"params": params}, batch_no_obs)
        assert not jnp.allclose(out1, out2)


class TestFourDVarNet2D:
    def test_output_shape(self, rng, batch_2d):
        B, T, H, W = batch_2d.input.shape
        model = FourDVarNet2D(
            n_time=T,
            latent_dim=8,
            hidden_dim=8,
            n_solver_steps=2,
        )
        params = model.init(rng, batch_2d)["params"]
        out = model.apply({"params": params}, batch_2d)
        assert out.shape == (B, T, H, W)
