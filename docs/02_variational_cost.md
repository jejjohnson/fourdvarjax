# Variational Cost

The variational (or energy) cost in fourdvarjax has the form:

$$J(\mathbf{x}) = J_{obs}(\mathbf{x}) + \lambda \, J_{prior}(\mathbf{x})$$

## Observation Cost

$$J_{obs}(\mathbf{x}) = \|\mathbf{m} \odot (\mathbf{x} - \mathbf{y})\|^2$$

This penalises deviations from observed values at masked locations only.

## Prior Cost

$$J_{prior}(\mathbf{x}) = \|\mathbf{x} - \varphi_\theta(\mathbf{x})\|^2$$

where $\varphi_\theta$ is the learned autoencoder prior.  When
$\varphi_\theta(\mathbf{x}) = \mathbf{x}$ (perfect reconstruction), the prior
cost vanishes.

## Implementation

In `fourdvarjax`:

```python
from fourdvarjax import obs_cost_1d, prior_cost

j_obs = obs_cost_1d(state, obs, mask)
j_prior = prior_cost(state, prior_model(state))
j_total = j_obs + lam * j_prior
```
