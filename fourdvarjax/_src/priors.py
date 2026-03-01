"""Learned prior models for 4DVarNet.

All priors are implemented as Flax NNX modules with an ``encode`` /
``decode`` interface (bilinear autoencoder) or a simple forward pass that
returns the autoencoder reconstruction.
"""

from __future__ import annotations

import flax.linen as nn
import jax
from jaxtyping import Array, Float

# ---------------------------------------------------------------------------
# Helper layers
# ---------------------------------------------------------------------------


class _BilinearBlock(nn.Module):
    """Single bilinear block: relu(Ax) * tanh(Bx)."""

    features: int

    @nn.compact
    def __call__(self, x: Array) -> Array:
        a = nn.Dense(self.features)(x)
        b = nn.Dense(self.features)(x)
        return jax.nn.relu(a) * jax.nn.tanh(b)


# ---------------------------------------------------------------------------
# 1-D priors
# ---------------------------------------------------------------------------


class BilinAEPrior1D(nn.Module):
    """Bilinear autoencoder prior for 1-D data.

    The encoder maps the input to a low-dimensional latent code; the decoder
    reconstructs the original space.  The prior cost is
    ``||x - decode(encode(x))||^2``.

    Attributes:
        state_dim: Spatial size of the input (``N``).
        latent_dim: Dimensionality of the latent code.
        n_time: Number of time steps (``T``).
    """

    state_dim: int
    latent_dim: int
    n_time: int = 1

    def setup(self):
        self._bilin = _BilinearBlock(self.latent_dim)
        self._decode_dense = nn.Dense(self.n_time * self.state_dim)

    def __call__(self, x: Float[Array, "B T N"]) -> Float[Array, "B T N"]:
        b, t, n = x.shape
        x_flat = x.reshape(b, t * n)
        z = self._bilin(x_flat)
        out = self._decode_dense(z)
        return out.reshape(b, t, n)

    def encode(self, x: Float[Array, "B T N"]) -> Float[Array, "B Z"]:
        """Encode input to latent space."""
        b, t, n = x.shape
        x_flat = x.reshape(b, t * n)
        return self._bilin(x_flat)

    def decode(self, z: Float[Array, "B Z"]) -> Float[Array, "B T N"]:
        """Decode latent code to state space."""
        out = self._decode_dense(z)
        return out.reshape(-1, self.n_time, self.state_dim)


class MLPAEPrior1D(nn.Module):
    """MLP autoencoder prior for 1-D data.

    Attributes:
        state_dim: Spatial size of the input (``N``).
        latent_dim: Dimensionality of the latent code.
        hidden_dim: Hidden layer width.
        n_time: Number of time steps (``T``).
    """

    state_dim: int
    latent_dim: int
    hidden_dim: int = 64
    n_time: int = 1

    @nn.compact
    def __call__(self, x: Float[Array, "B T N"]) -> Float[Array, "B T N"]:
        b, t, n = x.shape
        x_flat = x.reshape(b, t * n)
        z = nn.relu(nn.Dense(self.hidden_dim)(x_flat))
        z = nn.Dense(self.latent_dim)(z)
        h = nn.relu(nn.Dense(self.hidden_dim)(z))
        out = nn.Dense(t * n)(h)
        return out.reshape(b, t, n)


class BilinAEPrior2D(nn.Module):
    """Bilinear autoencoder prior for 2-D data.

    Attributes:
        latent_dim: Dimensionality of the latent code.
        n_time: Number of time steps (``T``).
    """

    latent_dim: int
    n_time: int = 1

    @nn.compact
    def __call__(self, x: Float[Array, "B T H W"]) -> Float[Array, "B T H W"]:
        b, t, h, w = x.shape
        x_flat = x.reshape(b, t * h * w)
        z = _BilinearBlock(self.latent_dim)(x_flat)
        out = nn.Dense(t * h * w)(z)
        return out.reshape(b, t, h, w)


class BilinAEPrior2DMultivar(nn.Module):
    """Bilinear autoencoder prior for 2-D multivariate data.

    Attributes:
        latent_dim: Dimensionality of the latent code.
        n_time: Number of time steps (``T``).
    """

    latent_dim: int
    n_time: int = 1

    @nn.compact
    def __call__(self, x: Float[Array, "B T C H W"]) -> Float[Array, "B T C H W"]:
        b, t, c, h, w = x.shape
        x_flat = x.reshape(b, t * c * h * w)
        z = _BilinearBlock(self.latent_dim)(x_flat)
        out = nn.Dense(t * c * h * w)(z)
        return out.reshape(b, t, c, h, w)


# ---------------------------------------------------------------------------
# Lorenz-63 prior
# ---------------------------------------------------------------------------


class L63Prior(nn.Module):
    """Learned prior for the Lorenz-63 system.

    A simple MLP autoencoder designed for the 3-dimensional Lorenz-63
    attractor.  The state is treated as a flat vector of length ``3``.

    Attributes:
        latent_dim: Dimensionality of the latent code (default ``3``).
        hidden_dim: Hidden layer width.
    """

    latent_dim: int = 3
    hidden_dim: int = 32

    @nn.compact
    def __call__(self, x: Float[Array, "B N"]) -> Float[Array, "B N"]:
        n = x.shape[-1]
        z = nn.tanh(nn.Dense(self.hidden_dim)(x))
        z = nn.Dense(self.latent_dim)(z)
        h = nn.tanh(nn.Dense(self.hidden_dim)(z))
        return nn.Dense(n)(h)
