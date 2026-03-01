# Learned Gradient Solver

The 4DVarNet solver replaces the classical L-BFGS or conjugate-gradient
minimiser with a **learned iterative scheme**.

## Algorithm

Given observations $\mathbf{y}$, mask $\mathbf{m}$, prior $\varphi_\theta$,
and gradient modulator $\Psi_\phi$:

1. Initialise $\mathbf{x}^{(0)} = \mathbf{m} \odot \mathbf{y}$
2. For $k = 1, \ldots, K$:
   a. Compute gradient: $\mathbf{g}^{(k)} = \nabla_x J(\mathbf{x}^{(k-1)})$
   b. Modulate gradient: $\mathbf{d}^{(k)}, \mathbf{s}^{(k)} = \Psi_\phi(\mathbf{g}^{(k)}, \mathbf{x}^{(k-1)}, \mathbf{s}^{(k-1)})$
   c. Update state: $\mathbf{x}^{(k)} = \mathbf{x}^{(k-1)} - \alpha \mathbf{d}^{(k)}$
3. Return $\mathbf{x}^{(K)}$

The entire solver is **differentiable** (unrolled through time), allowing
end-to-end training via backpropagation.
