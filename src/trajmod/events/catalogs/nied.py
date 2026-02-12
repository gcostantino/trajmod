"""NIED F-net (National Research Institute for Earth Science) catalog adapter.

NIED operates several networks in Japan:
- F-net: Broadband seismograph network with moment tensor solutions
- Hi-net: High-sensitivity seismograph network

Note: NIED data access may require registration/authentication.
      This adapter provides a template for integration.
"""

import logging
from typing import List, Dict, Optional

from trajmod.events.catalogs.base import CatalogAdapter, CatalogFetchError
from trajmod.events.catalogs.utils import rate_limit, CacheManager

logger = logging.getLogger(__name__)


class NIEDAdapter(CatalogAdapter):
    """Adapter for NIED F-net earthquake catalog (Japan).

    NIED F-net provides:
    - High-quality earthquake data for Japan region
    - Moment tensor solutions (focal mechanisms)
    - Broadband seismograph network (~70 stations)

    Supports two query modes:
    1. Circular region (lat, lon, radius)
    2. Rectangular box (minlat, maxlat, minlon, maxlon)

    Example:
        >>> adapter = NIEDAdapter()
        >>>
        >>> # Circular query around Tokyo
        >>> events = adapter.fetch(
        ...     lat=35.68, lon=139.69, radius_km=300,
        ...     start_date="2020-01-01", end_date="2023-12-31",
        ...     min_magnitude=5.0
        ... )
        >>>
        >>> # Box query for all Japan
        >>> events = adapter.fetch_box(
        ...     minlat=30.0, maxlat=45.0,
        ...     minlon=130.0, maxlon=145.0,
        ...     start_date="2020-01-01", end_date="2023-12-31",
        ...     min_magnitude=5.0
        ... )

    Note:
        This adapter uses NIED's earthquake catalog. Some features may
        require authentication. Visit https://hinetwww11.bosai.go.jp/
        for data access registration.
    """

    # NIED Hi-net catalog (requires authentication for full access)
    BASE_URL = "https://hinetwww11.bosai.go.jp/auth/JMA/jmalist.php"

    # Alternative: Use JMA catalog via NIED (public access)
    PUBLIC_URL = "https://www.data.jma.go.jp/svd/eqev/data/bulletin/catalog/appendix/monthly_e.csv"

    def __init__(self,
                 use_cache: bool = True,
                 cache_ttl_hours: int = 24,
                 api_key: Optional[str] = None):
        """Initialize NIED adapter.

        Args:
            use_cache: Whether to cache API responses
            cache_ttl_hours: Cache time-to-live (hours)
            api_key: NIED API key (if required for authenticated access)
        """
        self.use_cache = use_cache
        self.cache_manager = CacheManager(ttl_hours=cache_ttl_hours) if use_cache else None
        self.api_key = api_key

    @rate_limit(calls_per_second=0.5)
    def fetch(self,
              lat: float,
              lon: float,
              radius_km: float,
              start_date: str,
              end_date: str,
              min_magnitude: float = 4.0,
              max_depth_km: Optional[float] = None,
              limit: int = 10000) -> List[Dict]:
        """Fetch earthquakes from NIED using circular region.

        Args:
            lat: Center latitude (degrees)
            lon: Center longitude (degrees)
            radius_km: Search radius (km)
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            min_magnitude: Minimum magnitude
            max_depth_km: Maximum depth (km), optional
            limit: Maximum number of events

        Returns:
            List of earthquake events in NIED format

        Raises:
            CatalogFetchError: If API request fails

        Note:
            This is a placeholder implementation. NIED catalog access
            may require authentication. Consider using USGS or JMA
            for Japan earthquakes if NIED access is unavailable.
        """
        # Validate inputs
        self._validate_coordinates(lat, lon)
        self._validate_radius(radius_km)
        self._validate_dates(start_date, end_date)
        self._validate_magnitude(min_magnitude)

        # Check cache
        query_params = {
            'query_type': 'circular',
            'lat': lat, 'lon': lon, 'radius_km': radius_km,
            'start_date': start_date, 'end_date': end_date,
            'min_magnitude': min_magnitude, 'max_depth_km': max_depth_km
        }

        if self.use_cache:
            cached_events = self.cache_manager.load_from_cache('nied', **query_params)
            if cached_events is not None:
                return cached_events

        logger.warning("NIED adapter is experimental. Consider using USGS for Japan.")

        # For now, raise an error directing users to USGS
        raise CatalogFetchError(
            "NIED catalog access requires authentication. "
            "Please use CatalogFetcher.fetch_usgs() for Japan earthquakes, "
            "or register at https://hinetwww11.bosai.go.jp/ for NIED access."
        )

    @rate_limit(calls_per_second=0.5)
    def fetch_box(self,
                  minlat: float,
                  maxlat: float,
                  minlon: float,
                  maxlon: float,
                  start_date: str,
                  end_date: str,
                  min_magnitude: float = 4.0,
                  max_depth_km: Optional[float] = None,
                  limit: int = 10000) -> List[Dict]:
        """Fetch earthquakes from NIED using rectangular bounding box.

        Args:
            minlat: Minimum latitude (degrees)
            maxlat: Maximum latitude (degrees)
            minlon: Minimum longitude (degrees)
            maxlon: Maximum longitude (degrees)
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            min_magnitude: Minimum magnitude
            max_depth_km: Maximum depth (km), optional
            limit: Maximum number of events

        Returns:
            List of earthquake events in NIED format

        Raises:
            CatalogFetchError: If API request fails

        Note:
            This is a placeholder implementation. Use USGS or JMA
            for Japan earthquakes if NIED access is unavailable.
        """
        # Validate inputs
        self._validate_coordinates(minlat, minlon)
        self._validate_coordinates(maxlat, maxlon)
        self._validate_dates(start_date, end_date)
        self._validate_magnitude(min_magnitude)

        if minlat >= maxlat:
            raise ValueError(f"minlat ({minlat}) must be < maxlat ({maxlat})")
        if minlon >= maxlon:
            raise ValueError(f"minlon ({minlon}) must be < maxlon ({maxlon})")

        logger.warning("NIED adapter is experimental. Consider using USGS for Japan.")

        raise CatalogFetchError(
            "NIED catalog access requires authentication. "
            "Please use CatalogFetcher.fetch_usgs_box() for Japan earthquakes."
        )

    def to_trajmod_format(self, nied_event: Dict) -> Dict:
        """Convert NIED event to trajmod format.

        Args:
            nied_event: Event in NIED format

        Returns:
            Event in trajmod standard format
        """
        try:
            # NIED format (to be implemented based on actual API response)
            # This is a placeholder
            event = {
                'date': nied_event.get('time'),
                'lat': float(nied_event.get('latitude', 0)),
                'lon': float(nied_event.get('longitude', 0)),
                'magnitude': float(nied_event.get('magnitude', 0)),
                'depth_km': float(nied_event.get('depth', 0)),
                'magnitude_type': nied_event.get('mag_type', 'unknown'),
                'event_id': nied_event.get('id', 'unknown'),
                'source': 'NIED',
                'event_type': 'earthquake',
            }

            # Optional: Moment tensor if available
            if 'moment_tensor' in nied_event:
                event['moment_tensor'] = nied_event['moment_tensor']

            return event

        except (KeyError, TypeError, ValueError) as e:
            logger.warning(f"Failed to convert NIED event: {e}")
            raise CatalogFetchError(f"Invalid NIED event format: {e}")
