# Problem Setting

## Data Assimilation

Data assimilation (DA) is the process of combining prior knowledge (model
forecasts) with observational data to produce an optimal estimate of the state
of a dynamical system.

Given:
- A state vector $\mathbf{x} \in \mathbb{R}^n$
- Observations $\mathbf{y} \in \mathbb{R}^p$ with $p \ll n$
- An observation operator $\mathcal{H}: \mathbb{R}^n \to \mathbb{R}^p$
- Observation error covariance $\mathbf{R} \in \mathbb{R}^{p \times p}$
- Background error covariance $\mathbf{B} \in \mathbb{R}^{n \times n}$

The 4DVar problem seeks the state trajectory $\mathbf{x}(t)$ that minimises:

$$J(\mathbf{x}) = \frac{1}{2} \|\mathbf{x}_0 - \mathbf{x}_b\|^2_{\mathbf{B}^{-1}} + \frac{1}{2} \sum_{i=0}^{N} \|\mathbf{y}_i - \mathcal{H}_i(\mathbf{x}_i)\|^2_{\mathbf{R}^{-1}}$$

where $\mathbf{x}_b$ is the background (prior) state.

## 4DVarNet Reformulation

In the 4DVarNet framework, the classical background term is replaced by a
**learned prior** $\varphi_\theta: \mathbb{R}^n \to \mathbb{R}^n$:

$$J(\mathbf{x}) = \|\mathbf{m} \odot (\mathbf{x} - \mathbf{y})\|^2 + \lambda \|\mathbf{x} - \varphi_\theta(\mathbf{x})\|^2$$

where $\mathbf{m}$ is the binary observation mask.
