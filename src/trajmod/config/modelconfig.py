"""Configuration management for GNSS trajectory modeling."""
from dataclasses import dataclass, field
from typing import List, Optional, Any

import numpy as np


@dataclass
class ModelConfig:
    """Configuration for TrajectoryModel parameters.

    Attributes:
        d_param: Empirical scaling parameter in radius law
        include_seasonal: Whether to include annual & semi-annual terms
        use_envelope_basis: Whether to use envelope basis functions
        postseismic_mag_threshold: Minimum magnitude for post-seismic template
        tau_grid: Candidate tau values (days) for post-seismic decay
        merge_close_sse: Whether to merge close SSE templates
        merge_earthquakes_same_day: Whether to merge earthquakes on same day
        acceleration_term: Whether to include acceleration term
        envelope_periods: Periods (days) for envelope basis functions
        gap_merge_threshold: Threshold (days) for merging close events
        bspline_knots: Number of knots for B-spline basis
        bspline_degree: Degree of B-spline basis
    """
    d_param: Optional[float] = 1.0
    include_seasonal: bool = True
    use_envelope_basis: bool = False
    postseismic_mag_threshold: float = 6.5
    tau_grid: List[int] = field(default_factory=lambda: [7, 14, 30, 60, 90, 180, 1800])


    fit_postseismic_decay: bool = True
    postseismic_selection_criterion: str = 'aic'  # 'aic', 'bic', 'ftest', 'always'
    postseismic_selection_threshold: float = -2  # ΔAIC/ΔBIC < -2 or p < 0.05
    fit_best_postseismic_tau: bool = True
    enforce_postseismic_sign_consistency: bool = True
    postseismic_min_step_amplitude: float = 5.0  # mm


    merge_close_sse: bool = False
    merge_earthquakes_same_day: bool = True
    acceleration_term: bool = True

    # envelope periods (60-365 days, not 1-6 years!)
    envelope_periods: np.ndarray = field(
        default_factory=lambda: np.array([
            365.25 / 6,  # ~60.875 days (2 months)
            365.25 / 5,  # ~73.05 days
            365.25 / 4,  # ~91.3125 days (3 months)
            365.25 / 3,  # ~121.75 days (4 months)
            365.25 / 2,  # ~182.625 days (6 months)
            365.25  # 365.25 days (1 year)
        ])
    )

    gap_merge_threshold: int = 5
    bspline_knots: int = 20
    bspline_degree: int = 3


    def __post_init__(self):
        """Validate configuration parameters."""
        if self.d_param is not None and self.d_param <= 0:
            raise ValueError(f"d_param must be positive, got {self.d_param}")

        if self.postseismic_mag_threshold < 0:
            raise ValueError("postseismic_mag_threshold must be non-negative")

        # NEW: Validate min step amplitude
        if self.postseismic_min_step_amplitude < 0:
            raise ValueError("postseismic_min_step_amplitude must be non-negative")

        if any(tau <= 0 for tau in self.tau_grid):
            raise ValueError("All tau_grid values must be positive")

        valid_criteria = ['aic', 'bic', 'ftest', 'always']
        if self.postseismic_selection_criterion not in valid_criteria:
            raise ValueError(f"postseismic_selection_criterion must be one of {valid_criteria}")
