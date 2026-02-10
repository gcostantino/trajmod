"""Data preprocessing utilities for GNSS time series."""
from typing import Tuple, Optional

import numpy as np

from .time_utils import DecimalYearConverter


class TimeSeriesPreprocessor:
    """Preprocessor for GNSS time series data."""

    def __init__(self, converter: Optional[DecimalYearConverter] = None):
        self.converter = converter or DecimalYearConverter.from_algorithm()

    def fill_gaps(self, t: np.ndarray, y: np.ndarray,
                  uncertainties: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Fill temporal gaps in time series with NaN values."""
        t = np.asarray(t)
        y = np.asarray(y, dtype=float)
        uncertainties = np.asarray(uncertainties, dtype=float)

        if len(t) != len(y) or len(t) != len(uncertainties):
            raise ValueError("Input arrays must have the same length")

        t_decyr = self.converter.array_to_decimal(t)

        start_datetime, end_datetime = t.min(), t.max()
        n_days = (end_datetime - start_datetime).days
        full_time_dtime = np.linspace(start_datetime, end_datetime, n_days)
        full_time_decyr = self.converter.array_to_decimal(full_time_dtime)

        ts_full = np.full(full_time_decyr.shape, np.nan)
        unc_full = np.full(full_time_decyr.shape, np.nan)

        idx = np.searchsorted(full_time_decyr, t_decyr)
        ts_full[idx] = y
        unc_full[idx] = uncertainties

        return full_time_dtime, ts_full, unc_full, full_time_decyr

    @staticmethod
    def validate_data(t: np.ndarray, y: np.ndarray,
                      uncertainties: np.ndarray,
                      station_lat: float, station_lon: float) -> None:
        """Validate input data."""
        if len(t) != len(y) or len(t) != len(uncertainties):
            raise ValueError(f"Array length mismatch")
        if len(t) == 0:
            raise ValueError("Empty input arrays")
        if not (-90 <= station_lat <= 90):
            raise ValueError(f"Invalid latitude: {station_lat}")
        if not (-180 <= station_lon <= 180):
            raise ValueError(f"Invalid longitude: {station_lon}")
        if np.any(uncertainties[~np.isnan(uncertainties)] <= 0):
            raise ValueError("Uncertainties must be positive")

    @staticmethod
    def robust_std(data: np.ndarray, scale: float = 1.4826) -> float:
        """Compute robust standard deviation using MAD."""
        valid_data = data[np.isfinite(data)]
        if len(valid_data) == 0:
            return 0.0
        median = np.median(valid_data)
        mad = np.median(np.abs(valid_data - median))
        return scale * mad
