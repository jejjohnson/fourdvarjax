"""ConvLSTM gradient modulators for 4DVarNet.

The gradient modulator takes the current gradient of the variational cost
(with respect to the state) and the current LSTM hidden state, and outputs
a modulated gradient update plus the new LSTM state.
"""

from __future__ import annotations

import flax.linen as nn
import jax.numpy as jnp
from jaxtyping import Array, Float

from ._types import LSTMState1D, LSTMState2D


# ---------------------------------------------------------------------------
# 1-D ConvLSTM gradient modulator
# ---------------------------------------------------------------------------


class ConvLSTMGradMod1D(nn.Module):
    """1-D ConvLSTM-based gradient modulator.

    Accepts the concatenation of the current state and its gradient as input
    and produces a modulated gradient update (and updated LSTM state).

    Attributes:
        state_channels: Number of channels in the state / gradient.
        hidden_dim: Number of hidden channels in the LSTM.
        kernel_size: 1-D convolution kernel size.
    """

    state_channels: int
    hidden_dim: int
    kernel_size: int = 3

    def _conv(self, features: int) -> nn.Conv:
        return nn.Conv(
            features=features,
            kernel_size=(self.kernel_size,),
            padding="SAME",
        )

    @nn.compact
    def __call__(
        self,
        grad: Float[Array, "B T N"],
        state: Float[Array, "B T N"],
        lstm_state: LSTMState1D,
    ) -> tuple[Float[Array, "B T N"], LSTMState1D]:
        """Forward pass.

        Args:
            grad: Gradient of variational cost w.r.t. state, shape ``(B, T, N)``.
            state: Current state estimate, shape ``(B, T, N)``.
            lstm_state: Current LSTM hidden/cell state.

        Returns:
            Tuple of (modulated gradient update, new LSTM state).
        """
        # Concatenate along time axis and reshape to (B, N, C) for conv
        b, t, n = grad.shape
        # Treat T as channels for 1-D spatial conv over N
        x = jnp.concatenate([grad, state], axis=1)  # (B, 2T, N)
        x = jnp.transpose(x, (0, 2, 1))  # (B, N, 2T)

        h = jnp.transpose(lstm_state.h, (0, 2, 1))  # (B, N, H)
        c = lstm_state.c

        # LSTM gates (spatial convolution over N)
        gates_input = self._conv(4 * self.hidden_dim)(x)
        gates_hidden = self._conv(4 * self.hidden_dim)(h)
        gates = gates_input + gates_hidden  # (B, N, 4H)

        i, f, g, o = jnp.split(gates, 4, axis=-1)
        i = nn.sigmoid(i)
        f = nn.sigmoid(f)
        g = jnp.tanh(g)
        o = nn.sigmoid(o)

        c_new = f * jnp.transpose(c, (0, 2, 1)) + i * g
        h_new = o * jnp.tanh(c_new)

        # Output projection: from (B, N, H) back to (B, T, N)
        out = self._conv(t)(h_new)  # (B, N, T)
        out = jnp.transpose(out, (0, 2, 1))  # (B, T, N)

        new_lstm = LSTMState1D(
            h=jnp.transpose(h_new, (0, 2, 1)),
            c=jnp.transpose(c_new, (0, 2, 1)),
        )
        return out, new_lstm


# ---------------------------------------------------------------------------
# 2-D ConvLSTM gradient modulator
# ---------------------------------------------------------------------------


class ConvLSTMGradMod2D(nn.Module):
    """2-D ConvLSTM-based gradient modulator.

    Attributes:
        state_channels: Number of time channels in the state / gradient.
        hidden_dim: Number of hidden channels in the LSTM.
        kernel_size: 2-D convolution kernel size.
    """

    state_channels: int
    hidden_dim: int
    kernel_size: int = 3

    def _conv(self, features: int) -> nn.Conv:
        return nn.Conv(
            features=features,
            kernel_size=(self.kernel_size, self.kernel_size),
            padding="SAME",
        )

    @nn.compact
    def __call__(
        self,
        grad: Float[Array, "B T H W"],
        state: Float[Array, "B T H W"],
        lstm_state: LSTMState2D,
    ) -> tuple[Float[Array, "B T H W"], LSTMState2D]:
        """Forward pass.

        Args:
            grad: Gradient of variational cost w.r.t. state, shape ``(B, T, H, W)``.
            state: Current state estimate, shape ``(B, T, H, W)``.
            lstm_state: Current LSTM hidden/cell state.

        Returns:
            Tuple of (modulated gradient update, new LSTM state).
        """
        b, t, h_sz, w_sz = grad.shape

        # Reshape to (B, H, W, C) for Conv
        x = jnp.concatenate([grad, state], axis=1)  # (B, 2T, H, W)
        x = jnp.transpose(x, (0, 2, 3, 1))  # (B, H, W, 2T)

        hh = jnp.transpose(lstm_state.h, (0, 2, 3, 1))  # (B, H, W, H_dim)
        cc = jnp.transpose(lstm_state.c, (0, 2, 3, 1))  # (B, H, W, H_dim)

        gates_input = self._conv(4 * self.hidden_dim)(x)
        gates_hidden = self._conv(4 * self.hidden_dim)(hh)
        gates = gates_input + gates_hidden

        i, f, g, o = jnp.split(gates, 4, axis=-1)
        i = nn.sigmoid(i)
        f = nn.sigmoid(f)
        g = jnp.tanh(g)
        o = nn.sigmoid(o)

        c_new = f * cc + i * g
        h_new = o * jnp.tanh(c_new)

        # Output projection back to (B, T, H, W)
        out = self._conv(t)(h_new)  # (B, H, W, T)
        out = jnp.transpose(out, (0, 3, 1, 2))  # (B, T, H, W)

        new_lstm = LSTMState2D(
            h=jnp.transpose(h_new, (0, 3, 1, 2)),
            c=jnp.transpose(c_new, (0, 3, 1, 2)),
        )
        return out, new_lstm
