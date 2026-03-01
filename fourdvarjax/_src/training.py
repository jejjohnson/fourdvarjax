"""Training utilities for 4DVarNet.

Provides loss functions, per-step training / evaluation functions, and a
high-level ``fit`` loop compatible with Flax ``linen`` and ``optax``.
"""

from __future__ import annotations

from typing import Any

from flax.training import train_state
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float
import optax

from ._types import Batch1D, Batch2D

# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------


def reconstruction_loss(
    pred: Float[Array, ...],
    target: Float[Array, ...],
) -> Float[Array, ""]:
    """Mean-squared reconstruction loss.

    Args:
        pred: Model predictions, arbitrary shape.
        target: Ground-truth targets, same shape as ``pred``.

    Returns:
        Scalar mean-squared error.
    """
    return jnp.mean((pred - target) ** 2)


def train_loss_fn(
    params: Any,
    model: Any,
    batch: Batch1D | Batch2D,
    rngs: dict,
) -> Float[Array, ""]:
    """Compute the training loss for a single batch.

    Args:
        params: Model parameters (Flax param dict).
        model: Flax ``linen`` module.
        batch: Training batch.
        rngs: Random number generators dict (e.g. ``{"dropout": key}``).

    Returns:
        Scalar reconstruction loss.
    """
    pred = model.apply({"params": params}, batch, rngs=rngs)
    return reconstruction_loss(pred, batch.target)


# ---------------------------------------------------------------------------
# Training / evaluation steps
# ---------------------------------------------------------------------------


def train_step(
    state: train_state.TrainState,
    model: Any,
    batch: Batch1D | Batch2D,
    rng: Any,
) -> tuple[train_state.TrainState, Float[Array, ""]]:
    """Perform a single training step (forward + backward + update).

    Args:
        state: Current Flax ``TrainState`` holding params and optimizer state.
        model: Flax ``linen`` module.
        batch: Training batch.
        rng: JAX random key for stochastic operations (e.g. dropout).

    Returns:
        Tuple of (updated train state, scalar training loss).
    """
    rngs = {"dropout": rng}
    loss, grads = jax.value_and_grad(train_loss_fn)(state.params, model, batch, rngs)
    state = state.apply_gradients(grads=grads)
    return state, loss


def eval_step(
    state: train_state.TrainState,
    model: Any,
    batch: Batch1D | Batch2D,
) -> Float[Array, ""]:
    """Compute the evaluation loss for a single batch (no gradient).

    Args:
        state: Current Flax ``TrainState``.
        model: Flax ``linen`` module.
        batch: Evaluation batch.

    Returns:
        Scalar reconstruction loss.
    """
    pred = model.apply({"params": state.params}, batch, rngs={})
    return reconstruction_loss(pred, batch.target)


# ---------------------------------------------------------------------------
# High-level fit loop
# ---------------------------------------------------------------------------


def fit(
    model: Any,
    train_batches: list[Batch1D | Batch2D],
    *,
    rng: Any,
    lr: float = 1e-3,
    n_epochs: int = 10,
    val_batches: list[Batch1D | Batch2D] | None = None,
    verbose: bool = True,
) -> tuple[train_state.TrainState, list[float], list[float]]:
    """Train a 4DVarNet model for multiple epochs.

    Initialises the model on the first batch, then iterates over epochs and
    batches, logging train (and optional validation) losses.

    Args:
        model: Flax ``linen`` module.
        train_batches: List of training batches.
        rng: Initial JAX random key.
        lr: Learning rate for the Adam optimiser.
        n_epochs: Number of training epochs.
        val_batches: Optional list of validation batches.
        verbose: Whether to print per-epoch losses.

    Returns:
        Tuple of (final train state, train loss history, val loss history).
    """
    # Initialise model on first batch
    rng, init_rng = jax.random.split(rng)
    first_batch = train_batches[0]
    variables = model.init(init_rng, first_batch)
    params = variables["params"]

    tx = optax.adam(lr)
    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
    )

    train_losses: list[float] = []
    val_losses: list[float] = []

    for epoch in range(n_epochs):
        epoch_train_losses = []
        for batch in train_batches:
            rng, step_rng = jax.random.split(rng)
            state, loss = train_step(state, model, batch, step_rng)
            epoch_train_losses.append(float(loss))

        mean_train = float(jnp.mean(jnp.array(epoch_train_losses)))
        train_losses.append(mean_train)

        mean_val = float("nan")
        if val_batches is not None:
            epoch_val_losses = [float(eval_step(state, model, b)) for b in val_batches]
            mean_val = float(jnp.mean(jnp.array(epoch_val_losses)))
        val_losses.append(mean_val)

        if verbose:
            print(
                f"Epoch {epoch + 1}/{n_epochs} — "
                f"train loss: {mean_train:.6f}"
                + (f"  val loss: {mean_val:.6f}" if val_batches else "")
            )

    return state, train_losses, val_losses
