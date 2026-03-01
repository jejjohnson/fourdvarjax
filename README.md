# fourdvarjax

> Modular variational data assimilation with learned components (JAX/Flax NNX)

[![CI Tests](https://github.com/jejjohnson/fourdvarjax/actions/workflows/ci.yml/badge.svg)](https://github.com/jejjohnson/fourdvarjax/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/jejjohnson/fourdvarjax/branch/main/graph/badge.svg)](https://codecov.io/gh/jejjohnson/fourdvarjax)

`fourdvarjax` is a JAX/Flax NNX implementation of the **4DVarNet** framework — an end-to-end learnable variational data assimilation approach that combines classical 4D-Var with deep learning.

---

## Overview

The 4DVarNet framework minimizes a variational cost:

$$J(\mathbf{x}) = \|\mathbf{y} - \mathcal{H}(\mathbf{x})\|^2_{\mathbf{R}^{-1}} + \lambda \|\mathbf{x} - \varphi(\mathbf{x})\|^2$$

where:
- $\mathbf{y}$ are observations
- $\mathcal{H}$ is the observation operator
- $\varphi$ is a **learned prior** (autoencoder-based)
- The minimization uses **learned gradient steps** (ConvLSTM-based gradient modulator)

## Installation

```bash
pip install fourdvarjax
```

For development:

```bash
git clone https://github.com/jejjohnson/fourdvarjax
cd fourdvarjax
pip install -e ".[dev,test]"
```

## Quick Start

```python
import jax.numpy as jnp
import fourdvarjax

# Create a 1D FourDVarNet model
model = fourdvarjax.FourDVarNet1D(
    state_dim=64,
    n_time=10,
    hidden_dim=64,
    n_solver_steps=15,
)
```

## Architecture

- **`fourdvarjax._src._types`** — Batch and LSTM state types
- **`fourdvarjax._src.costs`** — Observation and prior cost functions
- **`fourdvarjax._src.priors`** — Bilinear autoencoder and MLP priors
- **`fourdvarjax._src.grad_mod`** — ConvLSTM gradient modulators
- **`fourdvarjax._src.solver`** — 4DVarNet iterative solver
- **`fourdvarjax._src.model`** — `FourDVarNet1D` / `FourDVarNet2D` end-to-end models
- **`fourdvarjax._src.training`** — Training utilities

## References

- Fablet, R. et al. (2021). *End-to-end learning of energy-based representations for irregularly-sampled signals and images.*
- Fablet, R. et al. (2021). *4DVarNet: End-to-end learning of variational data assimilation.*

## License

MIT
