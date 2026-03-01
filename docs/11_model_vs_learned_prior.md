# Model-Based vs Learned Prior

fourdvarjax supports two types of prior:

## Model-Based Prior

Uses a physics-based dynamical model $\mathcal{M}$ as the prior:

$$\varphi(\mathbf{x}) = \mathcal{M}(\mathbf{x})$$

This requires knowledge of the governing equations but can generalise well
outside the training distribution.

## Learned Prior

Uses a neural network autoencoder $\varphi_\theta$ trained on data.
Available options:

| Class | Description |
|---|---|
| `BilinAEPrior1D` | Bilinear AE, 1-D spatial data |
| `BilinAEPrior2D` | Bilinear AE, 2-D spatial data |
| `BilinAEPrior2DMultivar` | Bilinear AE, multivariate 2-D |
| `MLPAEPrior1D` | MLP AE, 1-D spatial data |
| `L63Prior` | Small MLP AE for Lorenz-63 |

## Hybrid

It is also possible to use a physics-informed neural network that combines
model equations with a learned correction term.
