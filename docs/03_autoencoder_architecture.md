# Autoencoder Architecture

The learned prior $\varphi_\theta$ is implemented as an autoencoder with a
bilinear bottleneck.

## Bilinear Autoencoder

The bilinear block applies two learned linear transformations and combines
them multiplicatively:

$$z = \text{ReLU}(\mathbf{A}\mathbf{x}) \odot \tanh(\mathbf{B}\mathbf{x})$$

This non-linearity allows the model to learn complex manifold structures
while remaining differentiable.

## 1-D Prior (`BilinAEPrior1D`)

Input shape: `(B, T, N)` → flatten to `(B, T*N)` → bilinear block → linear decode → reshape.

## 2-D Prior (`BilinAEPrior2D`)

Input shape: `(B, T, H, W)` → flatten → bilinear block → linear decode → reshape.

## MLP Prior (`MLPAEPrior1D`)

A standard MLP autoencoder for 1-D data:
encoder: Dense → ReLU → Dense (latent)
decoder: Dense → ReLU → Dense (reconstruction)
