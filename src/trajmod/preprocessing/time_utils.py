"""Time conversion utilities for GNSS trajectory modeling."""
import datetime
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np


class DecimalYearConverter:
    """Converter between datetime and decimal year representations."""

    def __init__(self, lookup_table: Optional[Dict[Tuple[int, int, int], float]] = None):
        """Initialize converter with optional lookup table.

        Args:
            lookup_table: Optional (year, month, day) -> decimal year mapping
        """
        self._lookup = lookup_table
        self._inv_lookup = None
        if lookup_table:
            self._inv_lookup = {v: k for k, v in lookup_table.items()}

    @classmethod
    def from_file(cls, filepath: Path) -> 'DecimalYearConverter':
        """Load lookup table from JPL-format file.

        Args:
            filepath: Path to decimal year lookup file

        Returns:
            DecimalYearConverter instance
        """
        import re

        if not filepath.exists():
            raise FileNotFoundError(f"Lookup file not found: {filepath}")

        lookup = {}
        with open(filepath, 'r') as f:
            next(f)  # Skip header
            for line in f:
                line = re.sub(r' +', ' ', line.strip())
                parts = line.split(' ')
                if len(parts) < 5:
                    continue
                decimal = float(parts[1])
                year, month, day = int(parts[2]), int(parts[3]), int(parts[4])
                lookup[(year, month, day)] = decimal

        return cls(lookup)

    @classmethod
    def from_algorithm(cls) -> 'DecimalYearConverter':
        """Create converter using algorithmic conversion (no lookup table)."""
        return cls(lookup_table=None)

    def datetime_to_decimal(self, dt: datetime.datetime) -> float:
        """Convert datetime to decimal year.

        Args:
            dt: Datetime to convert

        Returns:
            Decimal year
        """
        if self._lookup is not None:
            key = (dt.year, dt.month, dt.day)
            if key in self._lookup:
                return self._lookup[key]

        # Algorithmic fallback
        year_start = datetime.datetime(dt.year, 1, 1)
        year_end = datetime.datetime(dt.year + 1, 1, 1)
        year_elapsed = (dt - year_start).total_seconds()
        year_duration = (year_end - year_start).total_seconds()
        return dt.year + (year_elapsed / year_duration)

    def array_to_decimal(self, dt_array: np.ndarray) -> np.ndarray:
        """Convert array of datetimes to decimal years."""
        return np.array([self.datetime_to_decimal(dt) for dt in dt_array])

    def decimal_to_datetime(self, decimal: float) -> datetime.datetime:
        """Convert decimal year to datetime."""
        if self._inv_lookup is not None and decimal in self._inv_lookup:
            year, month, day = self._inv_lookup[decimal]
            return datetime.datetime(year, month, day)

        # Algorithmic fallback
        year = int(decimal)
        remainder = decimal - year
        year_start = datetime.datetime(year, 1, 1)
        year_end = datetime.datetime(year + 1, 1, 1)
        year_duration = (year_end - year_start).total_seconds()
        elapsed = datetime.timedelta(seconds=remainder * year_duration)
        return year_start + elapsed
