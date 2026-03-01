"""fourdvarjax — Modular variational data assimilation with learned components.

Public API
----------
All public symbols are re-exported from the private ``_src`` subpackage so
that user code imports from the top-level namespace:

.. code-block:: python

    import fourdvarjax

    model = fourdvarjax.FourDVarNet1D(...)
"""

from fourdvarjax._src._types import (
    Batch1D,
    Batch2D,
    Batch2DMultivar,
    LSTMState1D,
    LSTMState2D,
)
from fourdvarjax._src.costs import (
    obs_cost_1d,
    obs_cost_2d,
    prior_cost,
)
from fourdvarjax._src.grad_mod import (
    ConvLSTMGradMod1D,
    ConvLSTMGradMod2D,
)
from fourdvarjax._src.model import (
    FourDVarNet1D,
    FourDVarNet2D,
)
from fourdvarjax._src.priors import (
    BilinAEPrior1D,
    BilinAEPrior2D,
    BilinAEPrior2DMultivar,
    ConvAEPrior1D,
    L63Prior,
    L96Prior,
    MLPAEPrior1D,
)
from fourdvarjax._src.solver import (
    SolverState1D,
    SolverState2D,
    init_solver_state_1d,
    init_solver_state_2d,
    solve_4dvarnet_1d,
    solve_4dvarnet_2d,
    solver_step_1d,
    solver_step_2d,
)
from fourdvarjax._src.training import (
    eval_step,
    fit,
    reconstruction_loss,
    train_loss_fn,
    train_step,
)

__all__ = [
    # Types
    "Batch1D",
    "Batch2D",
    "Batch2DMultivar",
    "LSTMState1D",
    "LSTMState2D",
    # Costs
    "obs_cost_1d",
    "obs_cost_2d",
    "prior_cost",
    # Priors
    "BilinAEPrior1D",
    "BilinAEPrior2D",
    "BilinAEPrior2DMultivar",
    "ConvAEPrior1D",
    "L63Prior",
    "L96Prior",
    "MLPAEPrior1D",
    # Gradient modulators
    "ConvLSTMGradMod1D",
    "ConvLSTMGradMod2D",
    # Solver
    "SolverState1D",
    "SolverState2D",
    "init_solver_state_1d",
    "init_solver_state_2d",
    "solve_4dvarnet_1d",
    "solve_4dvarnet_2d",
    "solver_step_1d",
    "solver_step_2d",
    # Model
    "FourDVarNet1D",
    "FourDVarNet2D",
    # Training
    "eval_step",
    "fit",
    "reconstruction_loss",
    "train_loss_fn",
    "train_step",
]
