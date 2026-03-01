# Training Objective

The model is trained end-to-end by minimising the reconstruction loss on
labelled pairs $(\mathbf{y}, \mathbf{x}^*)$:

$$\mathcal{L}(\theta, \phi) = \mathbb{E}_{(\mathbf{y}, \mathbf{x}^*)} \left[ \| \hat{\mathbf{x}} - \mathbf{x}^* \|^2 \right]$$

where $\hat{\mathbf{x}} = \text{Solver}(\mathbf{y}; \theta, \phi)$ is the
output of the unrolled solver parameterised by prior weights $\theta$ and
gradient modulator weights $\phi$.

## fourdvarjax Training API

```python
from fourdvarjax import FourDVarNet1D, fit

model = FourDVarNet1D(state_dim=64, obs_dim=64, n_time=10)
state, train_losses, val_losses = fit(
    model,
    train_batches,
    rng=jax.random.PRNGKey(0),
    lr=1e-3,
    n_epochs=50,
    val_batches=val_batches,
)
```
