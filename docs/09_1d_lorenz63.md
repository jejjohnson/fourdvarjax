# 1-D Lorenz 63

The Lorenz-63 (L63) system is a 3-dimensional chaotic dynamical system:

$$\dot{x} = \sigma(y - x)$$
$$\dot{y} = x(\rho - z) - y$$
$$\dot{z} = xy - \beta z$$

with canonical parameters $\sigma = 10$, $\rho = 28$, $\beta = 8/3$.

## Data Assimilation Setup

- **State**: $(x, y, z) \in \mathbb{R}^3$ at each time step
- **Observations**: partial/noisy observations (e.g. only $x$ observed)
- **Prior**: `L63Prior` — a small MLP autoencoder that learns the attractor manifold
- **Solver**: `FourDVarNet1D` with `state_dim=3`, `n_time=T`

## Example

```python
from fourdvarjax import FourDVarNet1D, Batch1D

model = FourDVarNet1D(
    state_dim=3,
    obs_dim=3,
    n_time=20,
    latent_dim=8,
    hidden_dim=16,
    n_solver_steps=10,
)
```

See `notebooks/03_4dvarnet_L63.py` for a full example.
