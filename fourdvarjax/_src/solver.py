"""4DVarNet iterative solver.

Implements the gradient-descent-like solver that unrolls fixed iterations of
the variational cost minimisation, guided by the learned gradient modulator.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from ._types import Batch1D, Batch2D, LSTMState1D, LSTMState2D

# ---------------------------------------------------------------------------
# Solver state containers
# ---------------------------------------------------------------------------


class SolverState1D(NamedTuple):
    """Mutable solver state for 1-D problems.

    Attributes:
        x: Current state estimate of shape ``(B, T, N)``.
        lstm: Current LSTM hidden/cell state for the gradient modulator.
        step: Current iteration index.
    """

    x: Float[Array, "B T N"]
    lstm: LSTMState1D
    step: int


class SolverState2D(NamedTuple):
    """Mutable solver state for 2-D problems.

    Attributes:
        x: Current state estimate of shape ``(B, T, H, W)``.
        lstm: Current LSTM hidden/cell state for the gradient modulator.
        step: Current iteration index.
    """

    x: Float[Array, "B T H W"]
    lstm: LSTMState2D
    step: int


# ---------------------------------------------------------------------------
# Initialisation helpers
# ---------------------------------------------------------------------------


def init_solver_state_1d(
    batch: Batch1D,
    hidden_dim: int,
) -> SolverState1D:
    """Initialise a 1-D solver state from a batch.

    Args:
        batch: Input batch.  The initial state is set to the masked input
            (zeros where unobserved).
        hidden_dim: Hidden dimension of the ConvLSTM gradient modulator.

    Returns:
        Zero-initialised :class:`SolverState1D`.
    """
    b, _, n = batch.input.shape
    x0 = batch.input * batch.mask
    lstm = LSTMState1D.zeros(b, hidden_dim, n)
    return SolverState1D(x=x0, lstm=lstm, step=0)


def init_solver_state_2d(
    batch: Batch2D,
    hidden_dim: int,
) -> SolverState2D:
    """Initialise a 2-D solver state from a batch.

    Args:
        batch: Input batch.  The initial state is set to the masked input.
        hidden_dim: Hidden dimension of the ConvLSTM gradient modulator.

    Returns:
        Zero-initialised :class:`SolverState2D`.
    """
    b, _, h, w = batch.input.shape
    x0 = batch.input * batch.mask
    lstm = LSTMState2D.zeros(b, hidden_dim, h, w)
    return SolverState2D(x=x0, lstm=lstm, step=0)


# ---------------------------------------------------------------------------
# Single solver step
# ---------------------------------------------------------------------------


def solver_step_1d(
    solver_state: SolverState1D,
    batch: Batch1D,
    prior_fn: Any,
    grad_mod_fn: Any,
    alpha: float = 1.0,
) -> SolverState1D:
    """Perform a single 1-D solver iteration.

    Computes the gradient of the variational cost, then passes it through
    the learned gradient modulator to obtain a state update.

    Args:
        solver_state: Current solver state.
        batch: Observed data batch.
        prior_fn: Callable ``x -> x_prior`` (prior autoencoder forward pass).
        grad_mod_fn: Callable ``(grad, x, lstm) -> (update, new_lstm)``.
        alpha: Step-size scaling factor.

    Returns:
        Updated :class:`SolverState1D`.
    """
    x = solver_state.x

    def cost_fn(x_):
        x_prior = prior_fn(x_)
        obs_diff = batch.mask * (x_ - batch.input)
        j_obs = jnp.sum(obs_diff**2)
        j_prior = jnp.sum((x_ - x_prior) ** 2)
        return j_obs + j_prior

    grad = jax.grad(cost_fn)(x)
    update, new_lstm = grad_mod_fn(grad, x, solver_state.lstm)
    x_new = x - alpha * update

    return SolverState1D(x=x_new, lstm=new_lstm, step=solver_state.step + 1)


def solver_step_2d(
    solver_state: SolverState2D,
    batch: Batch2D,
    prior_fn: Any,
    grad_mod_fn: Any,
    alpha: float = 1.0,
) -> SolverState2D:
    """Perform a single 2-D solver iteration.

    Args:
        solver_state: Current solver state.
        batch: Observed data batch.
        prior_fn: Callable ``x -> x_prior``.
        grad_mod_fn: Callable ``(grad, x, lstm) -> (update, new_lstm)``.
        alpha: Step-size scaling factor.

    Returns:
        Updated :class:`SolverState2D`.
    """
    x = solver_state.x

    def cost_fn(x_):
        x_prior = prior_fn(x_)
        obs_diff = batch.mask * (x_ - batch.input)
        j_obs = jnp.sum(obs_diff**2)
        j_prior = jnp.sum((x_ - x_prior) ** 2)
        return j_obs + j_prior

    grad = jax.grad(cost_fn)(x)
    update, new_lstm = grad_mod_fn(grad, x, solver_state.lstm)
    x_new = x - alpha * update

    return SolverState2D(x=x_new, lstm=new_lstm, step=solver_state.step + 1)


# ---------------------------------------------------------------------------
# Full unrolled solver
# ---------------------------------------------------------------------------


def solve_4dvarnet_1d(
    batch: Batch1D,
    prior_fn: Any,
    grad_mod_fn: Any,
    n_steps: int,
    hidden_dim: int,
    alpha: float = 1.0,
) -> Float[Array, "B T N"]:
    """Run the full 1-D 4DVarNet solver for ``n_steps`` iterations.

    Args:
        batch: Observed data batch.
        prior_fn: Callable ``x -> x_prior``.
        grad_mod_fn: Callable ``(grad, x, lstm) -> (update, new_lstm)``.
        n_steps: Number of gradient-descent steps to unroll.
        hidden_dim: Hidden dimension of the ConvLSTM gradient modulator.
        alpha: Step-size scaling factor.

    Returns:
        Final state estimate of shape ``(B, T, N)``.
    """
    state = init_solver_state_1d(batch, hidden_dim)
    for _ in range(n_steps):
        state = solver_step_1d(state, batch, prior_fn, grad_mod_fn, alpha)
    return state.x


def solve_4dvarnet_2d(
    batch: Batch2D,
    prior_fn: Any,
    grad_mod_fn: Any,
    n_steps: int,
    hidden_dim: int,
    alpha: float = 1.0,
) -> Float[Array, "B T H W"]:
    """Run the full 2-D 4DVarNet solver for ``n_steps`` iterations.

    Args:
        batch: Observed data batch.
        prior_fn: Callable ``x -> x_prior``.
        grad_mod_fn: Callable ``(grad, x, lstm) -> (update, new_lstm)``.
        n_steps: Number of gradient-descent steps to unroll.
        hidden_dim: Hidden dimension of the ConvLSTM gradient modulator.
        alpha: Step-size scaling factor.

    Returns:
        Final state estimate of shape ``(B, T, H, W)``.
    """
    state = init_solver_state_2d(batch, hidden_dim)
    for _ in range(n_steps):
        state = solver_step_2d(state, batch, prior_fn, grad_mod_fn, alpha)
    return state.x
