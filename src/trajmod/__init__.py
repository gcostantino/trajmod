"""trajmod: Simple GNSS trajectory modeling.

A Python library for modeling GNSS displacement time series with support for
earthquakes, slow slip events, and advanced multi-tier postseismic filtering.
"""

__version__ = "0.1.0"
__author__ = "Giuseppe Costantino"
__license__ = "MIT"

# Import main classes for convenient access
from trajmod.model.model import TrajectoryModel
from trajmod.config.modelconfig import ModelConfig
from trajmod.model.results import ModelResults

__all__ = [
    "TrajectoryModel",
    "ModelConfig",
    "ModelResults",
    "__version__",
]