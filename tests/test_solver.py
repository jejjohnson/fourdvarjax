"""Tests for fourdvarjax._src.solver."""

import flax.nnx as nnx
import jax
import jax.numpy as jnp

from fourdvarjax import (
    LSTMState1D,
    SolverState1D,
    fp_solver_step_1d,
    init_solver_state_1d,
    init_solver_state_2d,
    solve_4dvarnet_1d_fixedpoint,
)
from fourdvarjax._src.solver import solver_step_1d


class TestInitSolverState1D:
    def test_shape(self, batch_1d):
        state = init_solver_state_1d(batch_1d, hidden_dim=16)
        B, T, N = batch_1d.input.shape
        assert state.x.shape == (B, T, N)
        assert state.step == 0

    def test_initial_state_is_masked_input(self, batch_1d):
        state = init_solver_state_1d(batch_1d, hidden_dim=16)
        expected = batch_1d.input * batch_1d.mask
        assert jnp.allclose(state.x, expected)


class TestInitSolverState2D:
    def test_shape(self, batch_2d):
        state = init_solver_state_2d(batch_2d, hidden_dim=8)
        B, T, H, W = batch_2d.input.shape
        assert state.x.shape == (B, T, H, W)
        assert state.step == 0


class TestSolverStep1D:
    def test_step_increments(self, rng, batch_1d):
        from fourdvarjax import BilinAEPrior1D, ConvLSTMGradMod1D

        B, T, N = batch_1d.input.shape
        hidden_dim = 16
        k1, k2 = jax.random.split(rng)
        prior = BilinAEPrior1D(
            state_dim=N, latent_dim=8, n_time=T, rngs=nnx.Rngs(k1)
        )
        grad_mod = ConvLSTMGradMod1D(
            state_channels=T, hidden_dim=hidden_dim, rngs=nnx.Rngs(k2)
        )

        x0 = batch_1d.input * batch_1d.mask
        lstm = LSTMState1D.zeros(B, hidden_dim, N)

        state = SolverState1D(x=x0, lstm=lstm, step=0)
        new_state = solver_step_1d(state, batch_1d, prior, grad_mod)
        assert new_state.step == 1

    def test_state_changes_after_step(self, rng, batch_1d):
        from fourdvarjax import BilinAEPrior1D, ConvLSTMGradMod1D

        B, T, N = batch_1d.input.shape
        hidden_dim = 16
        k1, k2 = jax.random.split(rng)
        prior = BilinAEPrior1D(
            state_dim=N, latent_dim=8, n_time=T, rngs=nnx.Rngs(k1)
        )
        grad_mod = ConvLSTMGradMod1D(
            state_channels=T, hidden_dim=hidden_dim, rngs=nnx.Rngs(k2)
        )

        x0 = batch_1d.input * batch_1d.mask
        lstm = LSTMState1D.zeros(B, hidden_dim, N)

        state = SolverState1D(x=x0, lstm=lstm, step=0)
        new_state = solver_step_1d(state, batch_1d, prior, grad_mod)
        # State should differ after the step
        assert not jnp.allclose(state.x, new_state.x)


class TestFpSolverStep1D:
    def test_output_shape(self, batch_1d):
        x = batch_1d.input * batch_1d.mask
        identity_fn = lambda x_: x_
        x_new = fp_solver_step_1d(x, batch_1d, identity_fn)
        assert x_new.shape == x.shape

    def test_observed_locations_equal_input(self, batch_1d):
        """At observed locations (mask==1), result should equal the input."""
        x = jnp.zeros_like(batch_1d.input)
        identity_fn = lambda x_: x_
        x_new = fp_solver_step_1d(x, batch_1d, identity_fn)
        mask = batch_1d.mask.astype(bool)
        assert jnp.allclose(x_new[mask], batch_1d.input[mask])


class TestSolve4dvarnet1dFixedpoint:
    def test_output_shape(self, rng, batch_1d):
        from fourdvarjax import BilinAEPrior1D

        B, T, N = batch_1d.input.shape
        prior = BilinAEPrior1D(state_dim=N, latent_dim=4, n_time=T, rngs=nnx.Rngs(rng))

        result = solve_4dvarnet_1d_fixedpoint(batch_1d, prior, n_fp_steps=5)
        assert result.shape == (B, T, N)

    def test_zero_steps_returns_masked_input(self, rng, batch_1d):
        from fourdvarjax import BilinAEPrior1D

        B, T, N = batch_1d.input.shape
        prior = BilinAEPrior1D(state_dim=N, latent_dim=4, n_time=T, rngs=nnx.Rngs(rng))
        x0 = batch_1d.input * batch_1d.mask

        result = solve_4dvarnet_1d_fixedpoint(batch_1d, prior, n_fp_steps=0)
        assert result.shape == (B, T, N)
        assert jnp.allclose(result, x0)
