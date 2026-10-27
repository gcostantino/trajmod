"""Configuration management for GNSS trajectory modeling."""
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class ModelConfig:
    """Configuration for TrajectoryModel parameters.

    Attributes:
        d_param: Empirical scaling parameter in radius law
        include_seasonal: Whether to include annual & semi-annual terms
        use_envelope_basis: Whether to add extra seasonal periods (envelope_periods)
        postseismic_mag_threshold: Minimum magnitude for post-seismic template
            (only used when d_param is set)
        tau_grid: Candidate tau values (days) for post-seismic decay
        fit_postseismic_decay: Whether to fit post-seismic log-decay terms
        postseismic_selection_criterion: 'aic', 'bic', 'ftest' or 'always'
        postseismic_selection_threshold: Delta-AIC/BIC must be below this value
            (or p-value for 'ftest')
        postseismic_tau_penalty_params: Parameters counted per post-seismic term
            in the information criteria / F-test (1 = amplitude only,
            2 = amplitude + tau). Default 1 keeps the legacy behaviour.
        postseismic_backfit_passes: Max passes re-optimising each accepted tau
            with the other post-seismic terms fixed (0 disables backfitting)
        enforce_postseismic_sign_consistency: Post-seismic must have the sign of the step
        postseismic_min_step_amplitude: Minimum |step| (mm) to consider post-seismic
        merge_close_sse: Whether to merge close SSE templates
        max_overlap_merge_sse: Max gap (days) between SSEs that triggers a merge
        merge_earthquakes_same_day: Whether to merge earthquakes on the same day
        acceleration_term: Whether to include acceleration term
        envelope_periods: Extra seasonal periods (days) used when use_envelope_basis
        bspline_knots: Number of knots for B-spline basis
        bspline_degree: Degree of B-spline basis

        SSE timing optimisation (daily data, whole-day shifts):
        sse_optimize_timing: Optimise SSE start/end times within bounds around the catalog
        sse_timing_tol_days: Minimum timing tolerance delta_min (days)
        sse_timing_tol_frac: Tolerance as a fraction of the SSE duration
        sse_timing_tol_max_days: Maximum timing tolerance delta_max (days)
            Per-event tolerance = min(max_days, max(tol_days, frac * duration)).
        sse_min_duration_days: Minimum SSE duration allowed during optimisation (days)
        sse_timing_step_days: Search step (whole days, >= 1; 1 for daily GNSS)
        sse_timing_prior: Gaussian prior pulling start/end toward the catalog times
            (width = tolerance / 2)
        sse_timing_criterion: 'bic' or 'aic' penalty for accepting a changed timing
        sse_timing_max_passes: Max coordinate-descent passes over all SSEs
    """
    d_param: Optional[float] = 1.0
    include_seasonal: bool = True
    use_envelope_basis: bool = False
    postseismic_mag_threshold: float = 6.5
    tau_grid: List[int] = field(default_factory=lambda: [7, 14, 30, 60, 90, 180, 1800])

    fit_postseismic_decay: bool = True
    postseismic_selection_criterion: str = 'aic'  # 'aic', 'bic', 'ftest', 'always'
    postseismic_selection_threshold: float = -2   # dAIC/dBIC < -2 or p < 0.05
    postseismic_tau_penalty_params: int = 1       # 1 = amplitude only, 2 = amplitude + tau
    postseismic_backfit_passes: int = 2
    fit_best_postseismic_tau: bool = True         # DEPRECATED: no longer used
    enforce_postseismic_sign_consistency: bool = True
    postseismic_min_step_amplitude: float = 5.0   # mm

    merge_close_sse: bool = True
    max_overlap_merge_sse: int = 5  # days
    merge_earthquakes_same_day: bool = True
    acceleration_term: bool = True

    # Extra seasonal periods in days. The annual (365.25 d) and semiannual
    # (182.625 d) terms are already part of the seasonal block and are
    # not repeated here (they would make the design matrix rank deficient).
    envelope_periods: np.ndarray = field(
        default_factory=lambda: np.array([
            365.25 / 6,  # ~60.875 days (2 months)
            365.25 / 5,  # ~73.05 days
            365.25 / 4,  # ~91.3125 days (3 months)
            365.25 / 3,  # ~121.75 days (4 months)
        ])
    )

    bspline_knots: int = 20
    bspline_degree: int = 3

    # SSE timing optimisation (off by default: no change unless requested)
    sse_optimize_timing: bool = False
    sse_timing_tol_days: float = 5.0
    sse_timing_tol_frac: float = 0.25
    sse_timing_tol_max_days: float = 30.0
    sse_min_duration_days: int = 1
    sse_timing_step_days: int = 1
    sse_timing_prior: bool = True
    sse_timing_criterion: str = 'bic'  # 'bic' or 'aic'
    sse_timing_max_passes: int = 2

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.d_param is not None and self.d_param <= 0:
            raise ValueError(f"d_param must be positive, got {self.d_param}")

        if self.postseismic_mag_threshold < 0:
            raise ValueError("postseismic_mag_threshold must be non-negative")

        if self.postseismic_min_step_amplitude < 0:
            raise ValueError("postseismic_min_step_amplitude must be non-negative")

        if len(self.tau_grid) == 0:
            raise ValueError("tau_grid must contain at least one value")

        if any(tau <= 0 for tau in self.tau_grid):
            raise ValueError("All tau_grid values must be positive")

        valid_criteria = ['aic', 'bic', 'ftest', 'always']
        if self.postseismic_selection_criterion not in valid_criteria:
            raise ValueError(
                f"postseismic_selection_criterion must be one of {valid_criteria}")

        if self.postseismic_tau_penalty_params not in (1, 2):
            raise ValueError(
                "postseismic_tau_penalty_params must be 1 (amplitude only) "
                f"or 2 (amplitude + tau), got {self.postseismic_tau_penalty_params}")

        if self.postseismic_backfit_passes < 0:
            raise ValueError("postseismic_backfit_passes must be non-negative")

        if self.max_overlap_merge_sse < 0:
            raise ValueError("max_overlap_merge_sse must be non-negative")

        if any(p <= 0 for p in np.atleast_1d(self.envelope_periods)):
            raise ValueError("All envelope_periods must be positive")

        # --- SSE timing optimisation ---
        if self.sse_timing_tol_days < 0:
            raise ValueError("sse_timing_tol_days must be non-negative")
        if self.sse_timing_tol_frac < 0:
            raise ValueError("sse_timing_tol_frac must be non-negative")
        if self.sse_timing_tol_max_days < self.sse_timing_tol_days:
            raise ValueError(
                "sse_timing_tol_max_days must be >= sse_timing_tol_days "
                f"({self.sse_timing_tol_max_days} < {self.sse_timing_tol_days})")
        if int(self.sse_min_duration_days) != self.sse_min_duration_days \
                or self.sse_min_duration_days < 1:
            raise ValueError("sse_min_duration_days must be an integer >= 1")
        if int(self.sse_timing_step_days) != self.sse_timing_step_days \
                or self.sse_timing_step_days < 1:
            raise ValueError(
                "sse_timing_step_days must be an integer >= 1 "
                "(daily GNSS: whole-day shifts)")
        if self.sse_timing_criterion not in ('aic', 'bic'):
            raise ValueError("sse_timing_criterion must be 'aic' or 'bic'")
        if self.sse_timing_max_passes < 1:
            raise ValueError("sse_timing_max_passes must be >= 1")
