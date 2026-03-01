"""Tests for fourdvarjax._src.priors."""

import jax.numpy as jnp

from fourdvarjax import (
    BilinAEPrior1D,
    BilinAEPrior2D,
    BilinAEPrior2DMultivar,
    L63Prior,
    MLPAEPrior1D,
)


class TestBilinAEPrior1D:
    def test_output_shape(self, rng, batch_1d):
        model = BilinAEPrior1D(state_dim=16, latent_dim=8, n_time=5)
        params = model.init(rng, batch_1d.input)["params"]
        out = model.apply({"params": params}, batch_1d.input)
        assert out.shape == batch_1d.input.shape

    def test_encode_decode_shapes(self, rng, batch_1d):
        model = BilinAEPrior1D(state_dim=16, latent_dim=8, n_time=5)
        params = model.init(rng, batch_1d.input)["params"]
        z = model.apply({"params": params}, batch_1d.input, method=model.encode)
        assert z.shape == (batch_1d.input.shape[0], 8)
        decoded = model.apply({"params": params}, z, method=model.decode)
        assert decoded.shape == batch_1d.input.shape


class TestMLPAEPrior1D:
    def test_output_shape(self, rng, batch_1d):
        model = MLPAEPrior1D(state_dim=16, latent_dim=8, hidden_dim=32, n_time=5)
        params = model.init(rng, batch_1d.input)["params"]
        out = model.apply({"params": params}, batch_1d.input)
        assert out.shape == batch_1d.input.shape


class TestBilinAEPrior2D:
    def test_output_shape(self, rng, batch_2d):
        model = BilinAEPrior2D(latent_dim=16, n_time=3)
        params = model.init(rng, batch_2d.input)["params"]
        out = model.apply({"params": params}, batch_2d.input)
        assert out.shape == batch_2d.input.shape


class TestBilinAEPrior2DMultivar:
    def test_output_shape(self, rng, batch_2d_multivar):
        model = BilinAEPrior2DMultivar(latent_dim=16, n_time=3)
        params = model.init(rng, batch_2d_multivar.input)["params"]
        out = model.apply({"params": params}, batch_2d_multivar.input)
        assert out.shape == batch_2d_multivar.input.shape


class TestL63Prior:
    def test_output_shape(self, rng):
        model = L63Prior(latent_dim=3, hidden_dim=16)
        x = jnp.ones((4, 3))
        params = model.init(rng, x)["params"]
        out = model.apply({"params": params}, x)
        assert out.shape == (4, 3)


class TestIdentityPrior:
    def test_output_equals_input(self, rng, batch_1d):
        from fourdvarjax import IdentityPrior

        model = IdentityPrior()
        variables = model.init(rng, batch_1d.input)
        out = model.apply(variables, batch_1d.input)
        assert jnp.allclose(out, batch_1d.input)

    def test_output_shape(self, rng, batch_1d):
        from fourdvarjax import IdentityPrior

        model = IdentityPrior()
        variables = model.init(rng, batch_1d.input)
        out = model.apply(variables, batch_1d.input)
        assert out.shape == batch_1d.input.shape
