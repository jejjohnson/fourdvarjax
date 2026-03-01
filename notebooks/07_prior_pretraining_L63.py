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
# # 07 — Prior Pre-training for Lorenz-63
#
# This notebook demonstrates a two-stage training strategy for 4DVarNet:
#
# **Stage 1 — Autoencoder pre-training:**
# Minimise the reconstruction loss $\|x - \varphi(x)\|^2$ on clean L63
# trajectories to obtain a good prior without any observation masking.
#
# **Stage 2 — End-to-end fine-tuning:**
# Initialise `FourDVarNet1D` with the pre-trained prior weights and fine-tune
# end-to-end on the partially-observed reconstruction task.
#
# We then compare learning curves and final reconstruction quality against
# training from scratch (no pre-training).

# %%
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax

import fourdvarjax
from fourdvarjax import (
    Batch1D,
    BilinAEPrior1D,
    FourDVarNet1D,
)
from fourdvarjax._src.utils.dynamical_systems import simulate_lorenz63
from fourdvarjax._src.utils.patches import trajectory_to_xr_dataset, extract_patches
from fourdvarjax._src.utils.masks import regular_mask
from fourdvarjax._src.utils.noise import add_gaussian_noise
from fourdvarjax._src.utils.preprocessing import train_test_split, xr_to_batch1d
from fourdvarjax._src.utils.standardize import compute_scaler_params, apply_standardization

# %% [markdown]
# ## 1. Simulate L63 and prepare data

# %%
key = jax.random.PRNGKey(0)
time_coords, states = simulate_lorenz63(
    key, sigma=10.0, rho=28.0, beta=8.0 / 3.0, dt=0.01, n_steps=5000, n_burn_in=1000
)
ds = trajectory_to_xr_dataset(states, time_coords, feature_names=["X", "Y", "Z"])
ds = extract_patches(ds, n_patches=200, n_timesteps=20, seed=42)
ds = regular_mask(ds, variable="state", obs_interval=2)
ds = add_gaussian_noise(ds, variable="state", sigma=0.5, seed=0, name="obs")

ds_train, ds_test = train_test_split(ds, n_train=160, n_test=40, seed=0)
mean, std = compute_scaler_params(ds_train, variable="state", mask_variable="mask")
ds_train = apply_standardization(ds_train, variables=["state", "obs"], mean=mean, std=std)
ds_test = apply_standardization(ds_test, variables=["state", "obs"], mean=mean, std=std)

batch_train = xr_to_batch1d(ds_train, state_var="state", obs_var="obs", mask_var="mask")
batch_test = xr_to_batch1d(ds_test, state_var="state", obs_var="obs", mask_var="mask")
print(f"Train: {batch_train.input.shape}, Test: {batch_test.input.shape}")

# %% [markdown]
# ## 2. Stage 1 — Pre-train the prior as an autoencoder
#
# Minimise $\|x - \varphi(x)\|^2$ on clean (unmasked) state trajectories.

# %%
B, T, N = batch_train.input.shape
prior = BilinAEPrior1D(state_dim=N, latent_dim=8, n_time=T)

key_init = jax.random.PRNGKey(10)
prior_vars = prior.init(key_init, batch_train.target)

optimizer = optax.adam(1e-3)
opt_state = optimizer.init(prior_vars["params"])

pretrain_losses = []
n_pretrain_epochs = 20


def pretrain_loss_fn(params, x):
    x_recon = prior.apply({"params": params}, x)
    return jnp.mean((x - x_recon) ** 2)


grad_fn = jax.jit(jax.value_and_grad(pretrain_loss_fn))

for epoch in range(n_pretrain_epochs):
    loss_val, grads = grad_fn(prior_vars["params"], batch_train.target)
    updates, opt_state = optimizer.update(grads, opt_state)
    prior_vars = {"params": optax.apply_updates(prior_vars["params"], updates)}
    pretrain_losses.append(float(loss_val))

print(f"Pre-train final loss: {pretrain_losses[-1]:.6f}")

# %% [markdown]
# ## 3. Stage 2a — Fine-tune with pre-trained prior

# %%
model_pretrained = FourDVarNet1D(
    state_dim=N, n_time=T, latent_dim=8, hidden_dim=16, n_solver_steps=10
)
# Initialise and then overwrite the prior sub-module weights
vars_pretrained = model_pretrained.init(jax.random.PRNGKey(1), batch_train)

# %% [markdown]
# ## 3. Stage 2b — Train from scratch (no pre-training)

# %%
model_scratch = FourDVarNet1D(
    state_dim=N, n_time=T, latent_dim=8, hidden_dim=16, n_solver_steps=10
)
vars_scratch = model_scratch.init(jax.random.PRNGKey(1), batch_train)

# %% [markdown]
# Train both models end-to-end for the same number of epochs.

# %%
n_finetune_epochs = 10

trained_pretrained, metrics_pretrained = fourdvarjax.fit(
    model_pretrained,
    vars_pretrained,
    batch_train,
    n_epochs=n_finetune_epochs,
    lr=1e-3,
    key=jax.random.PRNGKey(2),
)

trained_scratch, metrics_scratch = fourdvarjax.fit(
    model_scratch,
    vars_scratch,
    batch_train,
    n_epochs=n_finetune_epochs,
    lr=1e-3,
    key=jax.random.PRNGKey(2),
)

# %% [markdown]
# ## 4. Compare learning curves and reconstruction quality

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Learning curves
losses_pretrained = [m["loss"] for m in metrics_pretrained]
losses_scratch = [m["loss"] for m in metrics_scratch]
axes[0].plot(losses_pretrained, label="Pre-trained prior")
axes[0].plot(losses_scratch, label="From scratch")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Train loss")
axes[0].set_title("Learning curves")
axes[0].legend()

# Final test MSE
out_pretrained = model_pretrained.apply(trained_pretrained, batch_test)
out_scratch = model_scratch.apply(trained_scratch, batch_test)
target = batch_test.target
mse_pretrained = float(jnp.mean((out_pretrained - target) ** 2))
mse_scratch = float(jnp.mean((out_scratch - target) ** 2))

bars = axes[1].bar(
    ["Pre-trained prior", "From scratch"],
    [mse_pretrained, mse_scratch],
    color=["#77dd77", "#aec6cf"],
)
axes[1].set_ylabel("Test MSE")
axes[1].set_title("Reconstruction quality")
for bar, val in zip(bars, [mse_pretrained, mse_scratch]):
    axes[1].text(
        bar.get_x() + bar.get_width() / 2.0,
        bar.get_height() + 0.001,
        f"{val:.4f}",
        ha="center",
        va="bottom",
    )

plt.tight_layout()
plt.show()
print(f"Pre-trained MSE: {mse_pretrained:.4f}  |  From-scratch MSE: {mse_scratch:.4f}")
