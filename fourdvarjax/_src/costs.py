"""Cost functions for 4DVarNet variational data assimilation."""

import jax.numpy as jnp
from jaxtyping import Array, Float


def obs_cost_1d(
    state: Float[Array, "B T N"],
    obs: Float[Array, "B T N"],
    mask: Float[Array, "B T N"],
) -> Float[Array, ""]:
    """Observation cost for 1-D data.

    Computes the masked mean-squared error between the state and observations:

    .. math::

        J_{obs} = \\frac{1}{|\\Omega|} \\sum_{i \\in \\Omega} (x_i - y_i)^2

    where :math:`\\Omega` is the set of observed locations (``mask == 1``).

    Args:
        state: Current state estimate of shape ``(B, T, N)``.
        obs: Observations of shape ``(B, T, N)``.
        mask: Binary observation mask of shape ``(B, T, N)``.
            A value of ``1`` indicates an observed location.

    Returns:
        Scalar observation cost.
    """
    diff = mask * (state - obs)
    return jnp.mean(diff**2)


def obs_cost_2d(
    state: Float[Array, "B T H W"],
    obs: Float[Array, "B T H W"],
    mask: Float[Array, "B T H W"],
) -> Float[Array, ""]:
    """Observation cost for 2-D data.

    Computes the masked mean-squared error between the state and observations:

    .. math::

        J_{obs} = \\frac{1}{|\\Omega|} \\sum_{i \\in \\Omega} (x_i - y_i)^2

    where :math:`\\Omega` is the set of observed locations (``mask == 1``).

    Args:
        state: Current state estimate of shape ``(B, T, H, W)``.
        obs: Observations of shape ``(B, T, H, W)``.
        mask: Binary observation mask of shape ``(B, T, H, W)``.

    Returns:
        Scalar observation cost.
    """
    diff = mask * (state - obs)
    return jnp.mean(diff**2)


def prior_cost(
    state: Float[Array, "..."],
    prior_reconstruction: Float[Array, "..."],
) -> Float[Array, ""]:
    """Prior cost based on learned autoencoder reconstruction.

    Computes the mean-squared error between the state and its reconstruction
    through the learned prior (autoencoder):

    .. math::

        J_{prior} = \\|x - \\varphi(x)\\|^2

    Args:
        state: Current state estimate of arbitrary shape.
        prior_reconstruction: Autoencoder reconstruction of the state,
            same shape as ``state``.

    Returns:
        Scalar prior cost.
    """
    return jnp.mean((state - prior_reconstruction) ** 2)
