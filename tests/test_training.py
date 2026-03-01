"""Tests for fourdvarjax._src.training."""

import jax
import jax.numpy as jnp
import pytest

from fourdvarjax import FourDVarNet1D, reconstruction_loss
from fourdvarjax._src.training import eval_step, train_loss_fn, train_step
from flax.training import train_state
import optax


@pytest.fixture
def model_and_state(rng, batch_1d):
    B, T, N = batch_1d.input.shape
    model = FourDVarNet1D(
        state_dim=N,
        obs_dim=N,
        n_time=T,
        latent_dim=8,
        hidden_dim=16,
        n_solver_steps=2,
    )
    params = model.init(rng, batch_1d)["params"]
    tx = optax.adam(1e-3)
    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
    )
    return model, state


class TestReconstructionLoss:
    def test_zero_for_identical(self):
        x = jnp.ones((2, 5, 16))
        loss = reconstruction_loss(x, x)
        assert float(loss) == pytest.approx(0.0)

    def test_positive(self):
        pred = jnp.ones((2, 5, 16))
        target = jnp.zeros((2, 5, 16))
        loss = reconstruction_loss(pred, target)
        assert float(loss) > 0.0

    def test_scalar(self):
        loss = reconstruction_loss(jnp.ones((2, 3)), jnp.zeros((2, 3)))
        assert loss.ndim == 0


class TestTrainLossFn:
    def test_returns_scalar(self, rng, batch_1d, model_and_state):
        model, state = model_and_state
        k1, k2 = jax.random.split(rng)
        loss = train_loss_fn(state.params, model, batch_1d, {"dropout": k2})
        assert loss.ndim == 0
        assert float(loss) >= 0.0


class TestTrainStep:
    def test_params_change(self, rng, batch_1d, model_and_state):
        model, state = model_and_state
        k1, k2 = jax.random.split(rng)
        new_state, loss = train_step(state, model, batch_1d, k2)
        # At least some parameter leaves should differ
        flat_old = jax.tree_util.tree_leaves(state.params)
        flat_new = jax.tree_util.tree_leaves(new_state.params)
        changed = any(not jnp.allclose(a, b) for a, b in zip(flat_old, flat_new))
        assert changed

    def test_loss_is_finite(self, rng, batch_1d, model_and_state):
        model, state = model_and_state
        _, k2 = jax.random.split(rng)
        _, loss = train_step(state, model, batch_1d, k2)
        assert jnp.isfinite(loss)


class TestEvalStep:
    def test_returns_scalar(self, batch_1d, model_and_state):
        model, state = model_and_state
        loss = eval_step(state, model, batch_1d)
        assert loss.ndim == 0
        assert jnp.isfinite(loss)
