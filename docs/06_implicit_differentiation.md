# Implicit Differentiation

The 4DVarNet solver is unrolled for a fixed number of steps $K$, making it
amenable to direct backpropagation through the unrolled computation graph.

## Unrolled Backprop

Training gradients flow back through all $K$ solver steps.  The memory cost
scales as $\mathcal{O}(K)$.  For large $K$, gradient checkpointing can be used.

## Fixed-Point Differentiation

An alternative is to treat the solver as finding a fixed point
$\mathbf{x}^* = \mathcal{T}(\mathbf{x}^*, \mathbf{y})$ and apply the
implicit function theorem:

$$\frac{\partial \mathbf{x}^*}{\partial \theta} = -\left(\mathbf{I} - \frac{\partial \mathcal{T}}{\partial \mathbf{x}}\right)^{-1} \frac{\partial \mathcal{T}}{\partial \theta}$$

This can be computed efficiently using `jaxopt` or `diffrax`.

## One-Step Differentiation

A third option — introduced by Bolte, Pauwels & Vaiter (NeurIPS 2023,
[arXiv:2305.13768](https://arxiv.org/abs/2305.13768)) — is to run the solver
for $K-1$ steps **without tracking gradients**, then perform a single final
step through which the gradient flows:

$$\hat{x}_K = \mathcal{T}(\texttt{stop\_gradient}(\hat{x}_{K-1}), \theta)$$

$$\frac{\partial \hat{x}_K}{\partial \theta} \approx \frac{\partial \mathcal{T}(\hat{x}_{K-1}, \theta)}{\partial \theta}$$

This has **O(1) memory cost** (same as implicit differentiation), is trivial to
implement with `jax.lax.stop_gradient`, and is asymptotically exact for fast-
converging solvers.  The approximation error is bounded by the contraction rate
of $\mathcal{T}$ at convergence.

### Usage

```python
# Construction-time selection
model = fourdvarjax.FourDVarNet1D(
    state_dim=3, n_time=20, latent_dim=8, hidden_dim=16,
    n_solver_steps=15,
    grad_mode="one_step",   # <-- new
    rngs=nnx.Rngs(key),
)

# Or use the lower-level solver directly
x_hat = fourdvarjax.one_step_solve_4dvarnet_1d(
    batch, model.prior, model.grad_mod,
    n_steps=15, hidden_dim=16,
)
```

### Comparison of differentiation strategies

| Strategy | Memory | Accuracy | Code complexity |
|---|---|---|---|
| `"unrolled"` | O(K) | Exact | Trivial |
| `"one_step"` | O(1) | Approximate (exact at convergence) | Trivial (`stop_gradient`) |
| `"implicit"` | O(1) | Exact (at convergence) | Requires fixed-point formulation |
