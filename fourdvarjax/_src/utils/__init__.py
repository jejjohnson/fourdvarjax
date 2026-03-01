"""Utility subpackage for fourdvarjax.

Submodules
----------
dynamical_systems : L63 ODE model and simulation helper.
patches           : xarray patch extraction from trajectories.
masks             : Observation mask generators.
noise             : Gaussian noise injection.
standardize       : Mean/std standardization and inverse transform.
preprocessing     : xarray → JAX conversion and train/test split.
viz               : Visualization helpers (soft matplotlib dependency).
"""

from fourdvarjax._src.utils import (
    dynamical_systems,
    masks,
    noise,
    patches,
    preprocessing,
    standardize,
    viz,
)

__all__ = [
    "dynamical_systems",
    "masks",
    "noise",
    "patches",
    "preprocessing",
    "standardize",
    "viz",
]
