"""USGS ComCat earthquake catalog adapter.

API Documentation: https://earthquake.usgs.gov/fdsnws/event/1/
"""

import logging
from datetime import datetime
from typing import List, Dict, Optional

import requests

from trajmod.events.catalogs.base import CatalogAdapter, CatalogFetchError
from trajmod.events.catalogs.utils import rate_limit, CacheManager

logger = logging.getLogger(__name__)


class USGSAdapter(CatalogAdapter):
    """Adapter for USGS ComCat earthquake catalog.

    USGS ComCat provides:
    - Global coverage
    - Real-time updates
    - High-quality data
    - Well-documented API

    Supports two query modes:
    1. Circular region (lat, lon, radius_km)
    2. Rectangular box (minlat, maxlat, minlon, maxlon)

    Example:
        >>> adapter = USGSAdapter()
        >>>
        >>> # Circular query
        >>> events = adapter.fetch(
        ...     lat=35.0, lon=140.0, radius_km=500,
        ...     start_date="2020-01-01", end_date="2023-12-31",
        ...     min_magnitude=5.5
        ... )
        >>>
        >>> # Box query
        >>> events = adapter.fetch_box(
        ...     minlat=30.0, maxlat=40.0,
        ...     minlon=135.0, maxlon=145.0,
        ...     start_date="2020-01-01", end_date="2023-12-31",
        ...     min_magnitude=5.5
        ... )
        >>>
        >>> formatted = [adapter.to_trajmod_format(e) for e in events]
    """

    BASE_URL = "https://earthquake.usgs.gov/fdsnws/event/1/query"

    def __init__(self, use_cache: bool = True, cache_ttl_hours: int = 24):
        """Initialize USGS adapter.

        Args:
            use_cache: Whether to cache API responses
            cache_ttl_hours: Cache time-to-live (hours)
        """
        self.use_cache = use_cache
        self.cache_manager = CacheManager(ttl_hours=cache_ttl_hours) if use_cache else None

    @rate_limit(calls_per_second=1.0)
    def fetch(self,
              lat: float,
              lon: float,
              radius_km: float,
              start_date: str,
              end_date: str,
              min_magnitude: float = 4.0,
              max_depth_km: Optional[float] = None,
              limit: int = 20000) -> List[Dict]:
        """Fetch earthquakes from USGS ComCat using circular region.

        Args:
            lat: Center latitude (degrees)
            lon: Center longitude (degrees)
            radius_km: Search radius (km)
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            min_magnitude: Minimum magnitude
            max_depth_km: Maximum depth (km), optional
            limit: Maximum number of events (default: 20000)

        Returns:
            List of earthquake events in USGS GeoJSON format

        Raises:
            CatalogFetchError: If API request fails

        Example:
            >>> adapter = USGSAdapter()
            >>> events = adapter.fetch(
            ...     lat=35.0, lon=140.0, radius_km=500,
            ...     start_date="2020-01-01", end_date="2023-12-31",
            ...     min_magnitude=5.5
            ... )
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
            cached_events = self.cache_manager.load_from_cache('usgs', **query_params)
            if cached_events is not None:
                return cached_events

        # Build API request
        params = {
            'format': 'geojson',
            'latitude': lat,
            'longitude': lon,
            'maxradiuskm': radius_km,
            'starttime': start_date,
            'endtime': end_date,
            'minmagnitude': min_magnitude,
            'orderby': 'time-asc',
            'limit': limit,
        }

        if max_depth_km is not None:
            params['maxdepth'] = max_depth_km

        # Make API request
        try:
            logger.info(f"Fetching USGS events (circular): {start_date} to {end_date}, "
                        f"center=({lat:.2f}, {lon:.2f}), radius={radius_km}km, "
                        f"min_mag={min_magnitude}")

            response = requests.get(self.BASE_URL, params=params, timeout=30)
            response.raise_for_status()

            data = response.json()

            # Extract features (events)
            events = data.get('features', [])
            logger.info(f"Fetched {len(events)} events from USGS")

            # Cache result
            if self.use_cache:
                self.cache_manager.save_to_cache(events, 'usgs', **query_params)

            return events

        except requests.RequestException as e:
            raise CatalogFetchError(f"USGS API request failed: {e}")
        except (KeyError, ValueError) as e:
            raise CatalogFetchError(f"Failed to parse USGS response: {e}")

    @rate_limit(calls_per_second=1.0)
    def fetch_box(self,
                  minlat: float,
                  maxlat: float,
                  minlon: float,
                  maxlon: float,
                  start_date: str,
                  end_date: str,
                  min_magnitude: float = 4.0,
                  max_depth_km: Optional[float] = None,
                  limit: int = 20000) -> List[Dict]:
        """Fetch earthquakes from USGS ComCat using rectangular bounding box.

        Args:
            minlat: Minimum latitude (degrees)
            maxlat: Maximum latitude (degrees)
            minlon: Minimum longitude (degrees)
            maxlon: Maximum longitude (degrees)
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            min_magnitude: Minimum magnitude
            max_depth_km: Maximum depth (km), optional
            limit: Maximum number of events (default: 20000)

        Returns:
            List of earthquake events in USGS GeoJSON format

        Raises:
            CatalogFetchError: If API request fails

        Example:
            >>> adapter = USGSAdapter()
            >>> # Fetch events in Japan region
            >>> events = adapter.fetch_box(
            ...     minlat=30.0, maxlat=45.0,
            ...     minlon=130.0, maxlon=145.0,
            ...     start_date="2020-01-01", end_date="2023-12-31",
            ...     min_magnitude=5.0
            ... )

        Note:
            For regions crossing the International Date Line (lon=180/-180),
            use two separate queries or use fetch() with circular region.
        """
        # Validate inputs
        self._validate_coordinates(minlat, minlon)
        self._validate_coordinates(maxlat, maxlon)
        self._validate_dates(start_date, end_date)
        self._validate_magnitude(min_magnitude)

        # Validate box parameters
        if minlat >= maxlat:
            raise ValueError(f"minlat ({minlat}) must be < maxlat ({maxlat})")

        if minlon >= maxlon:
            raise ValueError(f"minlon ({minlon}) must be < maxlon ({maxlon})")

        # Check cache
        query_params = {
            'query_type': 'box',
            'minlat': minlat, 'maxlat': maxlat,
            'minlon': minlon, 'maxlon': maxlon,
            'start_date': start_date, 'end_date': end_date,
            'min_magnitude': min_magnitude, 'max_depth_km': max_depth_km
        }

        if self.use_cache:
            cached_events = self.cache_manager.load_from_cache('usgs', **query_params)
            if cached_events is not None:
                return cached_events

        # Build API request
        params = {
            'format': 'geojson',
            'minlatitude': minlat,
            'maxlatitude': maxlat,
            'minlongitude': minlon,
            'maxlongitude': maxlon,
            'starttime': start_date,
            'endtime': end_date,
            'minmagnitude': min_magnitude,
            'orderby': 'time-asc',
            'limit': limit,
        }

        if max_depth_km is not None:
            params['maxdepth'] = max_depth_km

        # Make API request
        try:
            logger.info(f"Fetching USGS events (box): {start_date} to {end_date}, "
                        f"box=[{minlat:.2f},{maxlat:.2f}]x[{minlon:.2f},{maxlon:.2f}], "
                        f"min_mag={min_magnitude}")

            response = requests.get(self.BASE_URL, params=params, timeout=30)
            response.raise_for_status()

            data = response.json()

            # Extract features (events)
            events = data.get('features', [])
            logger.info(f"Fetched {len(events)} events from USGS")

            # Cache result
            if self.use_cache:
                self.cache_manager.save_to_cache(events, 'usgs', **query_params)

            return events

        except requests.RequestException as e:
            raise CatalogFetchError(f"USGS API request failed: {e}")
        except (KeyError, ValueError) as e:
            raise CatalogFetchError(f"Failed to parse USGS response: {e}")

    def to_trajmod_format(self, usgs_event: Dict) -> Dict:
        """Convert USGS GeoJSON event to trajmod format.

        Args:
            usgs_event: Event in USGS GeoJSON format

        Returns:
            Event in trajmod standard format

        Example:
            >>> usgs_event = {'id': 'us1000abc', 'properties': {...}, 'geometry': {...}}
            >>> adapter = USGSAdapter()
            >>> trajmod_event = adapter.to_trajmod_format(usgs_event)
        """
        try:
            props = usgs_event['properties']
            coords = usgs_event['geometry']['coordinates']

            # Parse time (milliseconds since epoch)
            time_ms = props['time']
            event_time = datetime.fromtimestamp(time_ms / 1000.0)

            # Extract coordinates [lon, lat, depth]
            lon, lat, depth = coords

            # Extract magnitude
            magnitude = props.get('mag')
            mag_type = props.get('magType', 'unknown')

            # Build standard format
            event = {
                'date': event_time,
                'lat': float(lat),
                'lon': float(lon),
                'magnitude': float(magnitude) if magnitude is not None else 0.0,
                'depth_km': float(depth) if depth is not None else 0.0,
                'magnitude_type': mag_type,
                'event_id': usgs_event.get('id', 'unknown'),
                'source': 'USGS',
                'event_type': 'earthquake',
            }

            # Optional fields
            if 'place' in props:
                event['location'] = props['place']

            if 'url' in props:
                event['url'] = props['url']

            return event

        except (KeyError, TypeError, ValueError) as e:
            logger.warning(f"Failed to convert USGS event: {e}")
            raise CatalogFetchError(f"Invalid USGS event format: {e}")
