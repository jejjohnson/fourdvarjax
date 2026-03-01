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
    decomposed_loss,
    obs_cost_1d,
    obs_cost_2d,
    prior_cost,
    variational_cost,
    variational_cost_grad,
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
    IdentityPrior,
    L63Prior,
    L96Prior,
    MLPAEPrior1D,
)
from fourdvarjax._src.solver import (
    GradMode,
    SolverState1D,
    SolverState2D,
    fp_solver_step_1d,
    init_solver_state_1d,
    init_solver_state_2d,
    one_step_solve_4dvarnet_1d,
    one_step_solve_4dvarnet_2d,
    solve_4dvarnet_1d,
    solve_4dvarnet_1d_fixedpoint,
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
from fourdvarjax._src.utils.dynamical_systems import simulate_lorenz96
from fourdvarjax._src.utils.viz import (
    plot_l96_grid,
    plot_l96_trajectories,
    plot_reconstruction_comparison,
)

__all__ = [
    # Types
    "Batch1D",
    "Batch2D",
    "Batch2DMultivar",
    "LSTMState1D",
    "LSTMState2D",
    # Costs
    "decomposed_loss",
    "obs_cost_1d",
    "obs_cost_2d",
    "prior_cost",
    "variational_cost",
    "variational_cost_grad",
    # Priors
    "BilinAEPrior1D",
    "BilinAEPrior2D",
    "BilinAEPrior2DMultivar",
    "ConvAEPrior1D",
    "IdentityPrior",
    "L63Prior",
    "L96Prior",
    "MLPAEPrior1D",
    # Gradient modulators
    "ConvLSTMGradMod1D",
    "ConvLSTMGradMod2D",
    # Solver
    "GradMode",
    "SolverState1D",
    "SolverState2D",
    "fp_solver_step_1d",
    "init_solver_state_1d",
    "init_solver_state_2d",
    "one_step_solve_4dvarnet_1d",
    "one_step_solve_4dvarnet_2d",
    "solve_4dvarnet_1d",
    "solve_4dvarnet_1d_fixedpoint",
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
    # Dynamical systems
    "simulate_lorenz96",
    # Visualization
    "plot_l96_grid",
    "plot_l96_trajectories",
    "plot_reconstruction_comparison",
]
