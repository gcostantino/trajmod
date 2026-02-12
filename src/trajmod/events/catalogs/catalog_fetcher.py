"""Catalog fetcher for downloading earthquake data from various sources.

This module provides the CatalogFetcher class which handles:
- API interaction with catalog providers
- Caching of responses
- Rate limiting
- Error handling

It does NOT handle:
- Event filtering (use EventCatalog for that)
- Event manipulation (use EventCatalog for that)
"""

import logging

logger = logging.getLogger(__name__)

import logging
from typing import List, Dict, Optional

from trajmod.events.catalogs.utils import CacheManager, deduplicate_events

logger = logging.getLogger(__name__)


class CatalogFetcher:
    """Fetches earthquake catalogs from various sources.

    This class is responsible ONLY for fetching data from APIs.
    For filtering and manipulation, use EventCatalog.

    Available sources:
    - USGS: Global coverage, best API, real-time
    - ISC: Global coverage, reviewed catalog
    - NIED: Japan only, requires authentication
    - JMA: Japan only, limited API

    Example:
        >>> fetcher = CatalogFetcher()
        >>>
        >>> # USGS - Recommended for most uses
        >>> events = fetcher.fetch_usgs(
        ...     lat=35.0, lon=140.0, radius_km=500,
        ...     start_date="2020-01-01", end_date="2023-12-31",
        ...     min_magnitude=5.5
        ... )
        >>>
        >>> # Or use bounding box
        >>> events = fetcher.fetch_usgs_box(
        ...     minlat=30.0, maxlat=40.0,
        ...     minlon=135.0, maxlon=145.0,
        ...     start_date="2020-01-01", end_date="2023-12-31"
        ... )
    """

    def __init__(self, use_cache: bool = True, cache_ttl_hours: int = 24):
        """Initialize catalog fetcher.

        Args:
            use_cache: Whether to cache API responses
            cache_ttl_hours: Cache time-to-live (hours)
        """
        self.use_cache = use_cache
        self.cache_manager = CacheManager(ttl_hours=cache_ttl_hours) if use_cache else None
        logger.debug(f"CatalogFetcher initialized (cache={use_cache}, ttl={cache_ttl_hours}h)")

    # ========================================================================
    # USGS (RECOMMENDED - Global coverage, best API)
    # ========================================================================

    def fetch_usgs(self,
                   lat: float,
                   lon: float,
                   radius_km: float,
                   start_date: str,
                   end_date: str,
                   min_magnitude: float = 4.0,
                   max_depth_km: Optional[float] = None,
                   limit: int = 20000) -> List[Dict]:
        """Fetch earthquakes from USGS ComCat using circular region.

        RECOMMENDED: Use this for most applications.

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
            List of events in trajmod standard format
        """
        from trajmod.events.catalogs.usgs import USGSAdapter

        logger.info(f"Fetching USGS events (circular): {start_date} to {end_date}")

        adapter = USGSAdapter(use_cache=self.use_cache, cache_ttl_hours=24)
        raw_events = adapter.fetch(
            lat=lat, lon=lon, radius_km=radius_km,
            start_date=start_date, end_date=end_date,
            min_magnitude=min_magnitude, max_depth_km=max_depth_km, limit=limit
        )

        events = [adapter.to_trajmod_format(e) for e in raw_events]
        logger.info(f"Fetched {len(events)} events from USGS")
        return events

    def fetch_usgs_box(self,
                       minlat: float, maxlat: float,
                       minlon: float, maxlon: float,
                       start_date: str, end_date: str,
                       min_magnitude: float = 4.0,
                       max_depth_km: Optional[float] = None,
                       limit: int = 20000) -> List[Dict]:
        """Fetch earthquakes from USGS ComCat using rectangular bounding box.

        RECOMMENDED: Use this for regional queries.

        Args:
            minlat, maxlat: Latitude bounds (degrees)
            minlon, maxlon: Longitude bounds (degrees)
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            min_magnitude: Minimum magnitude
            max_depth_km: Maximum depth (km), optional
            limit: Maximum number of events

        Returns:
            List of events in trajmod standard format
        """
        from trajmod.events.catalogs.usgs import USGSAdapter

        logger.info(f"Fetching USGS events (box): {start_date} to {end_date}")

        adapter = USGSAdapter(use_cache=self.use_cache, cache_ttl_hours=24)
        raw_events = adapter.fetch_box(
            minlat=minlat, maxlat=maxlat, minlon=minlon, maxlon=maxlon,
            start_date=start_date, end_date=end_date,
            min_magnitude=min_magnitude, max_depth_km=max_depth_km, limit=limit
        )

        events = [adapter.to_trajmod_format(e) for e in raw_events]
        logger.info(f"Fetched {len(events)} events from USGS")
        return events

    # ========================================================================
    # ISC (Global coverage, reviewed catalog)
    # ========================================================================

    def fetch_isc(self,
                  lat: float,
                  lon: float,
                  radius_km: float,
                  start_date: str,
                  end_date: str,
                  min_magnitude: float = 4.0,
                  max_depth_km: Optional[float] = None,
                  limit: int = 10000) -> List[Dict]:
        """Fetch earthquakes from ISC using circular region.

        ISC provides reviewed and relocated events. Good for research.
        Note: ISC API can be slower than USGS.

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
            List of events in trajmod standard format
        """
        from trajmod.events.catalogs.isc import ISCAdapter

        logger.info(f"Fetching ISC events (circular): {start_date} to {end_date}")

        adapter = ISCAdapter(use_cache=self.use_cache, cache_ttl_hours=24)
        raw_events = adapter.fetch(
            lat=lat, lon=lon, radius_km=radius_km,
            start_date=start_date, end_date=end_date,
            min_magnitude=min_magnitude, max_depth_km=max_depth_km, limit=limit
        )

        events = [adapter.to_trajmod_format(e) for e in raw_events]
        logger.info(f"Fetched {len(events)} events from ISC")
        return events

    def fetch_isc_box(self,
                      minlat: float, maxlat: float,
                      minlon: float, maxlon: float,
                      start_date: str, end_date: str,
                      min_magnitude: float = 4.0,
                      max_depth_km: Optional[float] = None,
                      limit: int = 10000) -> List[Dict]:
        """Fetch earthquakes from ISC using rectangular bounding box.

        Args:
            minlat, maxlat: Latitude bounds (degrees)
            minlon, maxlon: Longitude bounds (degrees)
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            min_magnitude: Minimum magnitude
            max_depth_km: Maximum depth (km), optional
            limit: Maximum number of events

        Returns:
            List of events in trajmod standard format
        """
        from trajmod.events.catalogs.isc import ISCAdapter

        logger.info(f"Fetching ISC events (box): {start_date} to {end_date}")

        adapter = ISCAdapter(use_cache=self.use_cache, cache_ttl_hours=24)
        raw_events = adapter.fetch_box(
            minlat=minlat, maxlat=maxlat, minlon=minlon, maxlon=maxlon,
            start_date=start_date, end_date=end_date,
            min_magnitude=min_magnitude, max_depth_km=max_depth_km, limit=limit
        )

        events = [adapter.to_trajmod_format(e) for e in raw_events]
        logger.info(f"Fetched {len(events)} events from ISC")
        return events

    # ========================================================================
    # NIED (Japan only, requires authentication)
    # ========================================================================

    def fetch_nied(self,
                   lat: float, lon: float, radius_km: float,
                   start_date: str, end_date: str,
                   min_magnitude: float = 4.0) -> List[Dict]:
        """Fetch earthquakes from NIED (Japan).

        LIMITED: NIED requires authentication. Use USGS instead.

        Args:
            lat: Center latitude (degrees)
            lon: Center longitude (degrees)
            radius_km: Search radius (km)
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            min_magnitude: Minimum magnitude

        Returns:
            List of events in trajmod standard format

        Raises:
            CatalogFetchError: NIED access not available
        """
        from trajmod.events.catalogs.nied import NIEDAdapter

        logger.warning("NIED requires authentication. Use fetch_usgs() for Japan.")
        adapter = NIEDAdapter()
        raw_events = adapter.fetch(
            lat=lat, lon=lon, radius_km=radius_km,
            start_date=start_date, end_date=end_date,
            min_magnitude=min_magnitude
        )
        return [adapter.to_trajmod_format(e) for e in raw_events]

    def fetch_nied_box(self,
                       minlat: float, maxlat: float,
                       minlon: float, maxlon: float,
                       start_date: str, end_date: str,
                       min_magnitude: float = 4.0) -> List[Dict]:
        """Fetch earthquakes from NIED using box (Japan).

        LIMITED: Use fetch_usgs_box() for Japan instead.
        """
        from trajmod.events.catalogs.nied import NIEDAdapter

        logger.warning("NIED requires authentication. Use fetch_usgs_box() for Japan.")
        adapter = NIEDAdapter()
        raw_events = adapter.fetch_box(
            minlat=minlat, maxlat=maxlat, minlon=minlon, maxlon=maxlon,
            start_date=start_date, end_date=end_date,
            min_magnitude=min_magnitude
        )
        return [adapter.to_trajmod_format(e) for e in raw_events]

    # ========================================================================
    # JMA (Japan only, limited API)
    # ========================================================================

    def fetch_jma(self,
                  lat: float, lon: float, radius_km: float,
                  start_date: str, end_date: str,
                  min_magnitude: float = 4.0) -> List[Dict]:
        """Fetch earthquakes from JMA (Japan).

        LIMITED: JMA has limited public API. Use USGS instead.
        """
        from trajmod.events.catalogs.jma import JMAAdapter

        logger.warning("JMA has limited API. Use fetch_usgs() for Japan.")
        adapter = JMAAdapter()
        raw_events = adapter.fetch(
            lat=lat, lon=lon, radius_km=radius_km,
            start_date=start_date, end_date=end_date,
            min_magnitude=min_magnitude
        )
        return [adapter.to_trajmod_format(e) for e in raw_events]

    def fetch_jma_box(self,
                      minlat: float, maxlat: float,
                      minlon: float, maxlon: float,
                      start_date: str, end_date: str,
                      min_magnitude: float = 4.0) -> List[Dict]:
        """Fetch earthquakes from JMA using box (Japan).

        ⚠️ LIMITED: Use fetch_usgs_box() for Japan instead.
        """
        from trajmod.events.catalogs.jma import JMAAdapter

        logger.warning("JMA has limited API. Use fetch_usgs_box() for Japan.")
        adapter = JMAAdapter()
        raw_events = adapter.fetch_box(
            minlat=minlat, maxlat=maxlat, minlon=minlon, maxlon=maxlon,
            start_date=start_date, end_date=end_date,
            min_magnitude=min_magnitude
        )
        return [adapter.to_trajmod_format(e) for e in raw_events]

    # ========================================================================
    # UTILITIES
    # ========================================================================

    @staticmethod
    def merge(*catalogs: List[Dict], remove_duplicates: bool = True) -> List[Dict]:
        """Merge multiple catalogs.

        Example:
            >>> usgs_events = fetcher.fetch_usgs(...)
            >>> isc_events = fetcher.fetch_isc(...)
            >>> merged = fetcher.merge(usgs_events, isc_events)
        """
        merged = []
        for catalog in catalogs:
            merged.extend(catalog)

        logger.info(f"Merging {len(catalogs)} catalogs: {sum(len(c) for c in catalogs)} total")

        if remove_duplicates:
            merged = deduplicate_events(merged)
            logger.info(f"After deduplication: {len(merged)} events")

        return merged

    def clear_cache(self, source: Optional[str] = None) -> int:
        """Clear cached API responses.

        Args:
            source: If provided, only clear this source ('usgs', 'isc', etc.)

        Returns:
            Number of cache files deleted
        """
        if not self.use_cache:
            logger.warning("Cache is disabled")
            return 0

        return self.cache_manager.clear_cache(source)
