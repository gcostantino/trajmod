"""Configuration management for GNSS trajectory modeling."""
from dataclasses import dataclass, field
from typing import List

import numpy as np


@dataclass
class ModelConfig:
    """Configuration for TrajectoryModel parameters."""
    d_param: float = 1.0
    include_seasonal: bool = True
    use_envelope_basis: bool = False
    postseismic_mag_threshold: float = 6.5
    tau_grid: List[int] = field(default_factory=lambda: [7, 14, 30, 60, 90, 180, 1800])
    merge_close_sse: bool = False
    merge_earthquakes_same_day: bool = False
    acceleration_term: bool = True
    envelope_periods: np.ndarray = field(
        default_factory=lambda: np.array([365.25, 730.5, 1095.75, 1461.0, 1826.25, 2191.5])
    )
    gap_merge_threshold: int = 5
    bspline_knots: int = 20
    bspline_degree: int = 3

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.d_param <= 0:
            raise ValueError(f"d_param must be positive, got {self.d_param}")
        if self.postseismic_mag_threshold < 0:
            raise ValueError("postseismic_mag_threshold must be non-negative")
        if any(tau <= 0 for tau in self.tau_grid):
            raise ValueError("All tau_grid values must be positive")
