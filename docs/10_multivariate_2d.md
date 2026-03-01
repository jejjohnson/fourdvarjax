# Multivariate 2-D

For 2-D spatiotemporal fields (e.g. sea surface height, sea surface temperature),
fourdvarjax provides:

- `Batch2D` — single-variable 2-D batch `(B, T, H, W)`
- `Batch2DMultivar` — multivariate 2-D batch `(B, T, C, H, W)`
- `BilinAEPrior2D` — 2-D bilinear autoencoder prior
- `BilinAEPrior2DMultivar` — multivariate 2-D prior
- `ConvLSTMGradMod2D` — 2-D ConvLSTM gradient modulator
- `FourDVarNet2D` — end-to-end 2-D model

## Example

```python
from fourdvarjax import FourDVarNet2D, Batch2D

model = FourDVarNet2D(
    n_time=5,
    latent_dim=64,
    hidden_dim=64,
    n_solver_steps=15,
)
```
