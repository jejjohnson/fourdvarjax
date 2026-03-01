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
