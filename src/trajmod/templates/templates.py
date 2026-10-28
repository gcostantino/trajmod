"""Template functions for GNSS trajectory modeling."""
import datetime
from typing import Tuple

import numpy as np
from scipy.interpolate import BSpline


class TemplateFunctions:
    """Collection of template functions for trajectory modeling.

    Event times can be given either as datetimes (raised_cosine, step, log_decay)
    or as day offsets since t0 (raised_cosine_days, step_days, log_decay_days).
    The *_days variants are what the SSE timing optimiser uses. For date-only
    (daily) catalogs both families give identical columns.
    """

    def __init__(self, t_days: np.ndarray, t0: datetime.datetime):
        """Initialize template functions.

        Args:
            t_days: Time in days since t0
            t0: Reference time
        """
        self.t_days = t_days
        self.t0 = t0

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def day_offset(self, when: datetime.datetime) -> int:
        """Whole days between t0 and `when` (same convention as the datetime templates)."""
        return (when - self.t0).days

    # ------------------------------------------------------------------ #
    # Baseline terms
    # ------------------------------------------------------------------ #
    def offset(self) -> np.ndarray:
        """Constant offset template."""
        return np.ones_like(self.t_days)

    def trend(self) -> np.ndarray:
        """Linear trend template."""
        return self.t_days.copy()

    def acceleration(self) -> np.ndarray:
        """Quadratic acceleration template."""
        return 0.5 * self.t_days ** 2

    def seasonal_sin(self, period_days: float = 365.25) -> np.ndarray:
        """Sinusoidal seasonal template."""
        omega = 2 * np.pi / period_days
        return np.sin(omega * self.t_days)

    def seasonal_cos(self, period_days: float = 365.25) -> np.ndarray:
        """Cosinusoidal seasonal template."""
        omega = 2 * np.pi / period_days
        return np.cos(omega * self.t_days)

    # ------------------------------------------------------------------ #
    # SSE: raised cosine
    # ------------------------------------------------------------------ #
    def raised_cosine_days(self, start_num: float, end_num: float) -> np.ndarray:
        """Raised cosine ramp from 0 to 1 between two day offsets (since t0).

        Zero before start_num, 1 after end_num. Returns zeros if end_num <= start_num.
        """
        dur_days = end_num - start_num
        out = np.zeros_like(self.t_days, dtype=float)

        if dur_days <= 0:
            return out

        mask = (self.t_days >= start_num) & (self.t_days <= end_num)
        phase = (self.t_days[mask] - start_num) / dur_days
        out[mask] = 0.5 * (1.0 - np.cos(np.pi * phase))
        out[self.t_days > end_num] = 1.0

        return out

    def raised_cosine(self, start_day: datetime.datetime,
                      end_day: datetime.datetime) -> np.ndarray:
        """Raised cosine function for SSE events (datetime interface)."""
        dur_days = (end_day - start_day).days
        out = np.zeros_like(self.t_days, dtype=float)

        if dur_days <= 0:
            return out

        start_num = (start_day - self.t0).days
        end_num = (end_day - self.t0).days

        mask = (self.t_days >= start_num) & (self.t_days <= end_num)
        phase = (self.t_days[mask] - start_num) / dur_days
        out[mask] = 0.5 * (1.0 - np.cos(np.pi * phase))
        out[self.t_days > end_num] = 1.0

        return out

    # ------------------------------------------------------------------ #
    # Earthquakes: step and postseismic decay
    # ------------------------------------------------------------------ #
    def step_days(self, day_num: float) -> np.ndarray:
        """Heaviside step at a day offset (since t0)."""
        return (self.t_days >= day_num).astype(float)

    def step(self, event_day: datetime.datetime) -> np.ndarray:
        """Heaviside step function for earthquakes."""
        date_num = (event_day - self.t0).days
        return (self.t_days >= date_num).astype(float)

    def log_decay_days(self, day_num: float, tau: float) -> np.ndarray:
        """Logarithmic decay log(1 + dt/tau) for dt > 0, at a day offset (since t0)."""
        delta_t = self.t_days - day_num
        mask = delta_t > 0

        out = np.zeros_like(self.t_days, dtype=float)
        out[mask] = np.log1p(delta_t[mask] / tau)

        return out

    def log_decay(self, event_day: datetime.datetime, tau: float) -> np.ndarray:
        """Logarithmic decay for post-seismic relaxation."""
        date_num = (event_day - self.t0).days
        return self.log_decay_days(date_num, tau)

    def exp_decay_days(self, day_num: float, tau: float) -> np.ndarray:
        """Maxwell-type relaxation 1 - exp(-dt/tau) for dt > 0, at a day offset (since t0)."""
        delta_t = self.t_days - day_num
        out = np.zeros_like(self.t_days, dtype=float)
        mask = delta_t > 0
        out[mask] = -np.expm1(-delta_t[mask] / tau)
        return out

    def exp_decay(self, event_day: datetime.datetime, tau: float) -> np.ndarray:
        """Exponential (viscoelastic) relaxation after an earthquake."""
        date_num = (event_day - self.t0).days
        return self.exp_decay_days(date_num, tau)

    # ------------------------------------------------------------------ #
    # B-spline helpers (unchanged)
    # ------------------------------------------------------------------ #
    def bspline_basis(self, knot_index: int, knots: np.ndarray,
                      degree: int = 3) -> np.ndarray:
        """B-spline basis function."""
        c = np.zeros(len(knots) + degree - 1)
        c[knot_index] = 1.0

        spl = BSpline(knots, c, degree, extrapolate=False)
        result = spl(self.t_days)
        result[np.isnan(result)] = 0.0

        return result

    @staticmethod
    def create_bspline_knots(t_min: float, t_max: float,
                             n_knots: int, degree: int = 3) -> np.ndarray:
        """Create knot vector for B-splines."""
        interior_knots = np.linspace(t_min, t_max, n_knots)
        full_knots = np.concatenate([
            [interior_knots[0]] * degree,
            interior_knots,
            [interior_knots[-1]] * degree,
        ])
        return full_knots

    def seasonal_with_bspline_envelope(self, period_days: float = 365.25,
                                       n_knots: int = 20) -> Tuple[np.ndarray, np.ndarray]:
        """Seasonal component with time-varying amplitude via B-splines.

        Returns:
            Tuple of (envelope * sin(wt), envelope * cos(wt))
        """
        omega = 2 * np.pi / period_days

        knots = self.create_bspline_knots(
            self.t_days.min(), self.t_days.max(), n_knots, degree=3)

        # Create B-spline envelope (uses the first basis function as an example)
        envelope = self.bspline_basis(0, knots, degree=3)

        sin_modulated = envelope * np.sin(omega * self.t_days)
        cos_modulated = envelope * np.cos(omega * self.t_days)

        return sin_modulated, cos_modulated
