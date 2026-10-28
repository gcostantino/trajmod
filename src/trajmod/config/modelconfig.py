"""Configuration management for GNSS trajectory modeling."""
from dataclasses import dataclass, field
from typing import List, Optional, Union

import numpy as np


class _Inherit:
    """Marker: take the value from the shared d_param."""

    def __repr__(self) -> str:
        return 'INHERIT'


INHERIT = _Inherit()


@dataclass
class ModelConfig:
    """Configuration for TrajectoryModel parameters.

    Attributes:
        d_param: Shared default of the empirical scaling parameter in the radius law
            R = 10^((0.43*M - 0.7)/d). Used for d_param_sse and d_param_eq when those are
            not given. None disables the spatial filter
        d_param_sse: Radius-law parameter for SSEs (default: d_param; None = no filter)
        d_param_eq: Radius-law parameter for earthquakes (default: d_param; None = no filter)
        include_seasonal: Whether to include annual & semi-annual terms
        use_envelope_basis: Whether to add extra seasonal periods (envelope_periods)

        postseismic_mag_threshold: Minimum magnitude for any post-seismic term
            (always applied; set 0 to disable)
        tau_grid: Candidate tau values (days) for post-seismic decay
        fit_postseismic_decay: Whether to fit post-seismic decay terms (log and/or exp)
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
        postseismic_models: Post-seismic shapes allowed for a single term: 'log' (log(1 + dt/tau),
            afterslip-like) and/or 'exp' (1 - exp(-dt/tau), viscoelastic relaxation that saturates).
            Default ['log'] keeps the legacy log-only behaviour
        tau_grid_exp: Candidate tau values (days) for the exponential term, searched like tau_grid
        postseismic_allow_combined: Also test a log + exp pair for the same earthquake, with both
            taus searched jointly (needs both 'log' and 'exp' in postseismic_models). The pair is
            kept only if it beats the best single term by postseismic_selection_criterion
        postseismic_exp_min_ratio: In a log + exp pair, only combinations with
            tau_exp >= ratio * tau_log are tried, so the exponential is the slower process
            and near-duplicate pairs are avoided (must be >= 1)
        postseismic_exp_min_mag: Minimum magnitude for an exponential-only postseismic term
        postseismic_pair_min_mag: Minimum magnitude for a log+exp pair (needs postseismic_allow_combined)

        merge_close_sse: Whether to merge close SSE templates
        max_overlap_merge_sse: Max gap (days) between SSEs that triggers a merge
        merge_earthquakes_same_day: Whether to merge earthquakes on the same day
        acceleration_term: Whether to include acceleration term
        envelope_periods: Extra seasonal periods (days) used when use_envelope_basis

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
        sse_timing_max_rounds: Max coordinate-descent passes over all SSEs
        sse_timing_allow_overlap: Allow SSEs to overlap each other during timing optimisation
            (False keeps every SSE between its neighbours)

    """
    d_param: Optional[float] = 1.0
    d_param_sse: Union[float, None, _Inherit] = INHERIT
    d_param_eq: Union[float, None, _Inherit] = INHERIT

    include_seasonal: bool = True
    use_envelope_basis: bool = False
    postseismic_mag_threshold: float = 6.5
    tau_grid: List[int] = field(default_factory=lambda: [7, 14, 30, 60, 90, 180, 1800])

    fit_postseismic_decay: bool = True
    postseismic_selection_criterion: str = 'aic'  # 'aic', 'bic', 'ftest', 'always'
    postseismic_selection_threshold: float = -2  # dAIC/dBIC < -2 or p < 0.05
    postseismic_tau_penalty_params: int = 1  # 1 = amplitude only, 2 = amplitude + tau
    postseismic_backfit_passes: int = 2
    enforce_postseismic_sign_consistency: bool = True
    postseismic_min_step_amplitude: float = 5.0  # mm

    postseismic_models: List[str] = field(default_factory=lambda: ['log'])  # 'log', 'exp'
    tau_grid_exp: List[int] = field(default_factory=lambda: [180, 365, 730, 1460, 2920])
    postseismic_allow_combined: bool = False  # log + exp for the same earthquake
    postseismic_exp_min_ratio: float = 2.0  # tau_exp >= ratio * tau_log in a combined pair
    postseismic_exp_min_mag: float = 7.5  # min magnitude to allow an exp-only term
    postseismic_pair_min_mag: float = 7.5  # min magnitude to allow a log+exp pair

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

    # SSE timing optimisation (off by default: no change unless requested)
    sse_optimize_timing: bool = False
    sse_timing_tol_days: float = 5.0
    sse_timing_tol_frac: float = 0.25
    sse_timing_tol_max_days: float = 30.0
    sse_min_duration_days: int = 1
    sse_timing_step_days: int = 1
    sse_timing_prior: bool = True
    sse_timing_criterion: str = 'bic'  # 'bic' or 'aic'
    sse_timing_max_rounds: int = 2
    sse_timing_allow_overlap: bool = False

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.d_param_sse is INHERIT:
            self.d_param_sse = self.d_param
        if self.d_param_eq is INHERIT:
            self.d_param_eq = self.d_param
        for name in ('d_param', 'd_param_sse', 'd_param_eq'):
            v = getattr(self, name)
            if v is not None and v <= 0:
                raise ValueError(f"{name} must be positive, got {v}")

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

        if not set(self.postseismic_models) <= {'log', 'exp'} or not self.postseismic_models:
            raise ValueError("postseismic_models must be a non-empty subset of ['log', 'exp']")
        if len(self.tau_grid_exp) == 0 or any(t <= 0 for t in self.tau_grid_exp):
            raise ValueError("tau_grid_exp must contain positive values")
        if self.postseismic_allow_combined and not {'log', 'exp'} <= set(self.postseismic_models):
            raise ValueError("postseismic_allow_combined needs both 'log' and 'exp' in postseismic_models")
        if self.postseismic_exp_min_ratio < 1:
            raise ValueError("postseismic_exp_min_ratio must be >= 1")
        if self.postseismic_exp_min_mag < 0 or self.postseismic_pair_min_mag < 0:
            raise ValueError("postseismic_exp_min_mag and postseismic_pair_min_mag "
                             "must be non-negative")

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
        if self.sse_timing_max_rounds < 1:
            raise ValueError("sse_timing_max_rounds must be >= 1")
