"""GNSS Trajectory Modeling Library - Refactored."""

__version__ = "2.0.0"
__author__ = "Giuseppe Costantino"

from src.trajmod.config.modelconfig import ModelConfig
from src.trajmod.model.model import TrajectoryModel
from src.trajmod.events.events import SSEEvent, EarthquakeEvent, SSECatalog, EarthquakeCatalog
from src.trajmod.model.results import ModelResults
from src.trajmod.strategies.fitting import OLSFitter, LassoFitter, ElasticNetFitter, IterativeRefinementFitter

__all__ = [
    "ModelConfig",
    "TrajectoryModel",
    "SSEEvent",
    "EarthquakeEvent",
    "SSECatalog",
    "EarthquakeCatalog",
    "ModelResults",
    "OLSFitter",
    "LassoFitter",
    "ElasticNetFitter",
    "IterativeRefinementFitter",
]