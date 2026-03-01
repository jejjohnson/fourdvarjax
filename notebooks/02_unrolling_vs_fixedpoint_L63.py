# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 02 — Unrolling vs Fixed-Point on Lorenz-63
#
# This notebook compares the **unrolled** solver (standard backprop through
# $K$ steps) to the **fixed-point** differentiation approach.

# %%
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

import fourdvarjax
from fourdvarjax import Batch1D, FourDVarNet1D

# %% [markdown]
# ## Generate data

# %%
key = jax.random.PRNGKey(42)
B, T, N = 4, 20, 3
k1, k2 = jax.random.split(key)
target = jax.random.normal(k1, (B, T, N))
mask = (jax.random.uniform(k2, (B, T, N)) > 0.5).astype(jnp.float32)
batch = Batch1D(input=target * mask, mask=mask, target=target)

# %% [markdown]
# ## Unrolled solver (K steps)

# %%
model_unrolled = FourDVarNet1D(
    state_dim=N,
    obs_dim=N,
    n_time=T,
    latent_dim=8,
    hidden_dim=16,
    n_solver_steps=10,
)
params = model_unrolled.init(jax.random.PRNGKey(0), batch)["params"]
out_unrolled = model_unrolled.apply({"params": params}, batch)
print(f"Unrolled output shape: {out_unrolled.shape}")

# %% [markdown]
# ## Comparison

# %%
mse_unrolled = float(jnp.mean((out_unrolled - target) ** 2))
print(f"Unrolled MSE (untrained): {mse_unrolled:.4f}")
