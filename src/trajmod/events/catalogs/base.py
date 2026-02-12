"""Abstract base class for earthquake catalog adapters."""

import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Dict

logger = logging.getLogger(__name__)


class CatalogFetchError(Exception):
    """Raised when fetching earthquake catalog fails."""
    pass


class CatalogAdapter(ABC):
    """Abstract base class for earthquake catalog adapters.

    All catalog adapters must implement:
    - fetch(): Retrieve events from the catalog API
    - to_trajmod_format(): Convert catalog-specific format to standard format

    Standard trajmod format:
        {
            'date': datetime,           # Event time
            'lat': float,              # Latitude (degrees)
            'lon': float,              # Longitude (degrees)
            'magnitude': float,        # Magnitude (Mw preferred)
            'depth_km': float,         # Depth (km)
            'magnitude_type': str,     # 'Mw', 'Ms', 'Mb', 'ML'
            'event_id': str,           # Catalog ID
            'source': str,             # 'USGS', 'NIED', etc.
            'event_type': str,         # 'earthquake', default
        }
    """

    @abstractmethod
    def fetch(self, **kwargs) -> List[Dict]:
        """Fetch earthquake events from catalog.

        Args:
            **kwargs: Catalog-specific query parameters

        Returns:
            List of event dictionaries in catalog's native format

        Raises:
            CatalogFetchError: If fetch fails
        """
        pass

    @abstractmethod
    def to_trajmod_format(self, event: Dict) -> Dict:
        """Convert catalog event to trajmod standard format.

        Args:
            event: Event dict in catalog's native format

        Returns:
            Event dict in trajmod standard format
        """
        pass

    @staticmethod
    def _validate_dates(start_date: str, end_date: str) -> None:
        """Validate date strings.

        Args:
            start_date: Start date (ISO format: YYYY-MM-DD)
            end_date: End date (ISO format: YYYY-MM-DD)

        Raises:
            ValueError: If dates are invalid
        """
        try:
            start = datetime.fromisoformat(start_date)
            end = datetime.fromisoformat(end_date)
        except ValueError as e:
            raise ValueError(f"Invalid date format. Use YYYY-MM-DD: {e}")

        if start > end:
            raise ValueError(f"start_date ({start_date}) must be before end_date ({end_date})")

    @staticmethod
    def _validate_coordinates(lat: float, lon: float) -> None:
        """Validate latitude and longitude.

        Args:
            lat: Latitude (degrees)
            lon: Longitude (degrees)

        Raises:
            ValueError: If coordinates are invalid
        """
        if not (-90 <= lat <= 90):
            raise ValueError(f"Latitude must be in [-90, 90], got {lat}")
        if not (-180 <= lon <= 180):
            raise ValueError(f"Longitude must be in [-180, 180], got {lon}")

    @staticmethod
    def _validate_radius(radius_km: float) -> None:
        """Validate radius parameter.

        Args:
            radius_km: Search radius (km)

        Raises:
            ValueError: If radius is invalid
        """
        if radius_km <= 0:
            raise ValueError(f"radius_km must be positive, got {radius_km}")
        if radius_km > 20000:  # Half Earth circumference
            raise ValueError(f"radius_km too large (>{20000} km), got {radius_km}")

    @staticmethod
    def _validate_magnitude(magnitude: float) -> None:
        """Validate magnitude parameter.

        Args:
            magnitude: Earthquake magnitude

        Raises:
            ValueError: If magnitude is invalid
        """
        if not (-2 <= magnitude <= 10):
            raise ValueError(f"Magnitude must be in [-2, 10], got {magnitude}")
