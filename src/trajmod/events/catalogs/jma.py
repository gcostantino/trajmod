"""JMA (Japan Meteorological Agency) earthquake catalog adapter.

JMA is the official source for earthquake information in Japan.
Provides real-time and historical earthquake data.

Note: JMA API access may be limited. This adapter provides integration
      when API access is available.
"""

import logging
from typing import List, Dict, Optional

from trajmod.events.catalogs.base import CatalogAdapter, CatalogFetchError
from trajmod.events.catalogs.utils import rate_limit, CacheManager

logger = logging.getLogger(__name__)


class JMAAdapter(CatalogAdapter):
    """Adapter for JMA (Japan Meteorological Agency) earthquake catalog.

    JMA provides:
    - Official Japanese earthquake catalog
    - Real-time earthquake information
    - Tsunami warnings and forecasts
    - Seismic intensity data

    Supports two query modes:
    1. Circular region (lat, lon, radius)
    2. Rectangular box (minlat, maxlat, minlon, maxlon)

    Example:
        >>> adapter = JMAAdapter()
        >>>
        >>> # Circular query
        >>> events = adapter.fetch(
        ...     lat=35.0, lon=140.0, radius_km=300,
        ...     start_date="2020-01-01", end_date="2023-12-31",
        ...     min_magnitude=5.0
        ... )
        >>>
        >>> # Box query for Japan
        >>> events = adapter.fetch_box(
        ...     minlat=30.0, maxlat=45.0,
        ...     minlon=130.0, maxlon=145.0,
        ...     start_date="2020-01-01", end_date="2023-12-31",
        ...     min_magnitude=5.0
        ... )

    Note:
        JMA data is available through various sources. This adapter
        uses publicly available JSON endpoints when possible.
        Full API may require proper authorization.
    """

    # JMA earthquake list (public JSON endpoint)
    BASE_URL = "https://www.jma.go.jp/bosai/quake/data/list.json"

    # Alternative: JMA unified catalog (requires proper parsing)
    CATALOG_URL = "https://www.data.jma.go.jp/svd/eqev/data/bulletin"

    def __init__(self, use_cache: bool = True, cache_ttl_hours: int = 24):
        """Initialize JMA adapter.

        Args:
            use_cache: Whether to cache API responses
            cache_ttl_hours: Cache time-to-live (hours)
        """
        self.use_cache = use_cache
        self.cache_manager = CacheManager(ttl_hours=cache_ttl_hours) if use_cache else None

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
        """Fetch earthquakes from JMA using circular region.

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
            List of earthquake events in JMA format

        Raises:
            CatalogFetchError: If API request fails

        Note:
            JMA adapter is experimental. For reliable Japan earthquake
            data, use USGS which provides comprehensive global coverage
            including Japan.
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
            cached_events = self.cache_manager.load_from_cache('jma', **query_params)
            if cached_events is not None:
                return cached_events

        logger.warning("JMA adapter is experimental. Consider using USGS for Japan.")

        # For now, raise an error directing users to USGS
        raise CatalogFetchError(
            "JMA catalog API access is limited. "
            "Please use CatalogFetcher.fetch_usgs() for Japan earthquakes, "
            "which provides comprehensive coverage including JMA-reported events."
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
        """Fetch earthquakes from JMA using rectangular bounding box.

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
            List of earthquake events in JMA format

        Raises:
            CatalogFetchError: If API request fails
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

        logger.warning("JMA adapter is experimental. Consider using USGS for Japan.")

        raise CatalogFetchError(
            "JMA catalog API access is limited. "
            "Please use CatalogFetcher.fetch_usgs_box() for Japan earthquakes."
        )

    def to_trajmod_format(self, jma_event: Dict) -> Dict:
        """Convert JMA event to trajmod format.

        Args:
            jma_event: Event in JMA format

        Returns:
            Event in trajmod standard format
        """
        try:
            # JMA format (to be implemented based on actual API response)
            # Placeholder implementation
            event = {
                'date': jma_event.get('time'),
                'lat': float(jma_event.get('latitude', 0)),
                'lon': float(jma_event.get('longitude', 0)),
                'magnitude': float(jma_event.get('magnitude', 0)),
                'depth_km': float(jma_event.get('depth', 0)),
                'magnitude_type': 'Mj',  # JMA magnitude
                'event_id': jma_event.get('id', 'unknown'),
                'source': 'JMA',
                'event_type': 'earthquake',
            }

            # Optional: JMA intensity scale if available
            if 'intensity' in jma_event:
                event['jma_intensity'] = jma_event['intensity']

            return event

        except (KeyError, TypeError, ValueError) as e:
            logger.warning(f"Failed to convert JMA event: {e}")
            raise CatalogFetchError(f"Invalid JMA event format: {e}")
