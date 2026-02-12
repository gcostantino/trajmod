"""ISC (International Seismological Centre) catalog adapter.

API Documentation: http://www.isc.ac.uk/fdsnws/event/1/
The ISC uses FDSN web services, similar to USGS.
"""

import logging
from datetime import datetime
from typing import List, Dict, Optional

import requests

from trajmod.events.catalogs.base import CatalogAdapter, CatalogFetchError
from trajmod.events.catalogs.utils import rate_limit, CacheManager

logger = logging.getLogger(__name__)


class ISCAdapter(CatalogAdapter):
    """Adapter for ISC (International Seismological Centre) catalog.

    The ISC provides:
    - Global coverage with comprehensive catalog
    - Reviewed and relocated events
    - Historical data going back decades
    - FDSN web services (similar to USGS)

    Supports two query modes:
    1. Circular region (lat, lon, radius)
    2. Rectangular box (minlat, maxlat, minlon, maxlon)

    Example:
        >>> adapter = ISCAdapter()
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

    Note:
        ISC catalog may be slower than USGS as it processes more comprehensive
        data. Results may differ from USGS due to different location/magnitude
        determinations.
    """

    BASE_URL = "http://www.isc.ac.uk/fdsnws/event/1/query"

    def __init__(self, use_cache: bool = True, cache_ttl_hours: int = 24):
        """Initialize ISC adapter.

        Args:
            use_cache: Whether to cache API responses
            cache_ttl_hours: Cache time-to-live (hours)
        """
        self.use_cache = use_cache
        self.cache_manager = CacheManager(ttl_hours=cache_ttl_hours) if use_cache else None

    @rate_limit(calls_per_second=0.5)  # ISC has lower rate limit
    def fetch(self,
              lat: float,
              lon: float,
              radius_km: float,
              start_date: str,
              end_date: str,
              min_magnitude: float = 4.0,
              max_depth_km: Optional[float] = None,
              limit: int = 10000) -> List[Dict]:
        """Fetch earthquakes from ISC using circular region.

        Args:
            lat: Center latitude (degrees)
            lon: Center longitude (degrees)
            radius_km: Search radius (km)
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            min_magnitude: Minimum magnitude
            max_depth_km: Maximum depth (km), optional
            limit: Maximum number of events (default: 10000)

        Returns:
            List of earthquake events in ISC XML format

        Raises:
            CatalogFetchError: If API request fails

        Note:
            ISC API is slower than USGS. Be patient with large queries.
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
            cached_events = self.cache_manager.load_from_cache('isc', **query_params)
            if cached_events is not None:
                return cached_events

        # Build API request
        # ISC uses maxradius in degrees, not km
        radius_deg = radius_km / 111.0

        params = {
            'format': 'json',  # ISC supports JSON
            'lat': lat,
            'lon': lon,
            'maxradius': radius_deg,  # In degrees
            'starttime': start_date,
            'endtime': end_date,
            'minmag': min_magnitude,
            'orderby': 'time-asc',
            'limit': limit,
        }

        if max_depth_km is not None:
            params['maxdepth'] = max_depth_km

        # Make API request
        try:
            logger.info(f"Fetching ISC events (circular): {start_date} to {end_date}, "
                        f"center=({lat:.2f}, {lon:.2f}), radius={radius_km}km, "
                        f"min_mag={min_magnitude}")
            logger.warning("ISC API may be slow. This could take a while...")

            response = requests.get(self.BASE_URL, params=params, timeout=120)  # Longer timeout
            response.raise_for_status()

            # ISC returns different format - might be XML or JSON
            # Try JSON first
            try:
                data = response.json()
                events = data.get('features', [])
            except:
                # If JSON fails, parse as text and return empty for now
                logger.warning("ISC returned non-JSON response. Parsing not implemented yet.")
                events = []

            logger.info(f"Fetched {len(events)} events from ISC")

            # Cache result
            if self.use_cache:
                self.cache_manager.save_to_cache(events, 'isc', **query_params)

            return events

        except requests.RequestException as e:
            raise CatalogFetchError(f"ISC API request failed: {e}")
        except (KeyError, ValueError) as e:
            raise CatalogFetchError(f"Failed to parse ISC response: {e}")

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
        """Fetch earthquakes from ISC using rectangular bounding box.

        Args:
            minlat: Minimum latitude (degrees)
            maxlat: Maximum latitude (degrees)
            minlon: Minimum longitude (degrees)
            maxlon: Maximum longitude (degrees)
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            min_magnitude: Minimum magnitude
            max_depth_km: Maximum depth (km), optional
            limit: Maximum number of events (default: 10000)

        Returns:
            List of earthquake events in ISC format

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

        # Check cache
        query_params = {
            'query_type': 'box',
            'minlat': minlat, 'maxlat': maxlat,
            'minlon': minlon, 'maxlon': maxlon,
            'start_date': start_date, 'end_date': end_date,
            'min_magnitude': min_magnitude, 'max_depth_km': max_depth_km
        }

        if self.use_cache:
            cached_events = self.cache_manager.load_from_cache('isc', **query_params)
            if cached_events is not None:
                return cached_events

        # Build API request
        params = {
            'format': 'json',
            'minlat': minlat,
            'maxlat': maxlat,
            'minlon': minlon,
            'maxlon': maxlon,
            'starttime': start_date,
            'endtime': end_date,
            'minmag': min_magnitude,
            'orderby': 'time-asc',
            'limit': limit,
        }

        if max_depth_km is not None:
            params['maxdepth'] = max_depth_km

        # Make API request
        try:
            logger.info(f"Fetching ISC events (box): {start_date} to {end_date}, "
                        f"box=[{minlat:.2f},{maxlat:.2f}]x[{minlon:.2f},{maxlon:.2f}], "
                        f"min_mag={min_magnitude}")
            logger.warning("ISC API may be slow. This could take a while...")

            response = requests.get(self.BASE_URL, params=params, timeout=120)
            response.raise_for_status()

            # Try JSON parsing
            try:
                data = response.json()
                events = data.get('features', [])
            except:
                logger.warning("ISC returned non-JSON response.")
                events = []

            logger.info(f"Fetched {len(events)} events from ISC")

            # Cache result
            if self.use_cache:
                self.cache_manager.save_to_cache(events, 'isc', **query_params)

            return events

        except requests.RequestException as e:
            raise CatalogFetchError(f"ISC API request failed: {e}")
        except (KeyError, ValueError) as e:
            raise CatalogFetchError(f"Failed to parse ISC response: {e}")

    def to_trajmod_format(self, isc_event: Dict) -> Dict:
        """Convert ISC event to trajmod format.

        Args:
            isc_event: Event in ISC format

        Returns:
            Event in trajmod standard format
        """
        try:
            # ISC format similar to USGS GeoJSON
            props = isc_event.get('properties', {})
            coords = isc_event.get('geometry', {}).get('coordinates', [])

            # Parse time
            time_str = props.get('time')
            if isinstance(time_str, (int, float)):
                event_time = datetime.fromtimestamp(time_str / 1000.0)
            else:
                event_time = datetime.fromisoformat(time_str.replace('Z', '+00:00'))

            # Extract coordinates [lon, lat, depth]
            lon, lat, depth = coords if len(coords) >= 3 else (coords + [0] * (3 - len(coords)))

            # Extract magnitude
            magnitude = props.get('mag', 0.0)
            mag_type = props.get('magType', 'unknown')

            # Build standard format
            event = {
                'date': event_time,
                'lat': float(lat),
                'lon': float(lon),
                'magnitude': float(magnitude),
                'depth_km': float(depth),
                'magnitude_type': mag_type,
                'event_id': isc_event.get('id', 'unknown'),
                'source': 'ISC',
                'event_type': 'earthquake',
            }

            # Optional fields
            if 'place' in props:
                event['location'] = props['place']

            return event

        except (KeyError, TypeError, ValueError) as e:
            logger.warning(f"Failed to convert ISC event: {e}")
            raise CatalogFetchError(f"Invalid ISC event format: {e}")
