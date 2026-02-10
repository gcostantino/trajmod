"""Geodetic calculations and utilities."""
from typing import Optional

from pyproj import Geod


class GeodesicCalculator:
    """Calculator for geodesic distances and parameters."""

    def __init__(self, ellipsoid: str = "WGS84"):
        """Initialize geodesic calculator.

        Args:
            ellipsoid: Reference ellipsoid name (default: WGS84)
        """
        self._geod = Geod(ellps=ellipsoid)

    def distance_km(self, lat1: float, lon1: float,
                    lat2: float, lon2: float) -> float:
        """Calculate geodesic distance between two points.

        Args:
            lat1, lon1: First point coordinates (degrees)
            lat2, lon2: Second point coordinates (degrees)

        Returns:
            Distance in kilometers
        """
        _, _, dist_m = self._geod.inv(lon1, lat1, lon2, lat2)
        return dist_m / 1000.0

    @staticmethod
    def radius_from_magnitude(magnitude: float, d_param: Optional[float] = 1.0) -> float:
        """Calculate radius of influence from magnitude.

        Uses empirical scaling law: R = 10^((0.43*M - 0.7) / d_param)

        Args:
            magnitude: Event magnitude
            d_param: Empirical scaling parameter

        Returns:
            Radius in kilometers
        """
        if d_param is None:  # might be handled better
            return float('inf')
        return 10.0 ** ((0.43 * magnitude - 0.7) / d_param)

    @staticmethod
    def validate_coordinates(lat: float, lon: float) -> None:
        """Validate latitude and longitude.

        Args:
            lat: Latitude in degrees
            lon: Longitude in degrees

        Raises:
            ValueError: If coordinates are invalid
        """
        if not (-90 <= lat <= 90):
            raise ValueError(f"Latitude must be in [-90, 90], got {lat}")
        if not (-180 <= lon <= 180):
            raise ValueError(f"Longitude must be in [-180, 180], got {lon}")
