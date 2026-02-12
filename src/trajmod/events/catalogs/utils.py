"""Utility functions for catalog adapters.

Provides:
- Rate limiting decorator
- Cache management
- Event deduplication
- Distance calculations
"""

import hashlib
import json
import logging
import time
from datetime import datetime, timedelta
from functools import wraps
from pathlib import Path
from typing import List, Dict, Optional, Callable, Any

import numpy as np

logger = logging.getLogger(__name__)


def rate_limit(calls_per_second: float = 1.0):
    """Decorator to rate limit function calls.

    Args:
        calls_per_second: Maximum number of calls per second

    Example:
        >>> @rate_limit(calls_per_second=2)
        ... def fetch_data():
        ...     pass
    """
    min_interval = 1.0 / calls_per_second
    last_called = [0.0]

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            elapsed = time.time() - last_called[0]
            if elapsed < min_interval:
                sleep_time = min_interval - elapsed
                logger.debug(f"Rate limiting: sleeping {sleep_time:.2f}s")
                time.sleep(sleep_time)

            result = func(*args, **kwargs)
            last_called[0] = time.time()
            return result

        return wrapper

    return decorator


class CacheManager:
    """Manages caching of catalog queries.

    Caches API responses to:
    - Reduce API load
    - Avoid rate limiting
    - Enable offline work
    - Improve reproducibility
    """

    def __init__(self, cache_dir: Optional[Path] = None, ttl_hours: int = 24):
        """Initialize cache manager.

        Args:
            cache_dir: Directory for cache files (default: ~/.trajmod/cache)
            ttl_hours: Time-to-live for cache entries (hours)
        """
        if cache_dir is None:
            cache_dir = Path.home() / '.trajmod' / 'cache'

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl = timedelta(hours=ttl_hours)

    def _make_cache_key(self, **params) -> str:
        """Create cache key from parameters.

        Args:
            **params: Query parameters

        Returns:
            SHA256 hash of parameters
        """
        # Sort params for consistent hashing
        param_str = json.dumps(params, sort_keys=True, default=str)
        return hashlib.sha256(param_str.encode()).hexdigest()

    def get_cache_path(self, adapter_name: str, **params) -> Path:
        """Get cache file path for query.

        Args:
            adapter_name: Name of adapter (e.g., 'usgs')
            **params: Query parameters

        Returns:
            Path to cache file
        """
        cache_key = self._make_cache_key(**params)
        return self.cache_dir / f"{adapter_name}_{cache_key}.json"

    def is_cached(self, adapter_name: str, **params) -> bool:
        """Check if query result is cached and valid.

        Args:
            adapter_name: Name of adapter
            **params: Query parameters

        Returns:
            True if cached and not expired
        """
        cache_path = self.get_cache_path(adapter_name, **params)

        if not cache_path.exists():
            return False

        # Check if expired
        mtime = datetime.fromtimestamp(cache_path.stat().st_mtime)
        age = datetime.now() - mtime

        if age > self.ttl:
            logger.debug(f"Cache expired (age: {age})")
            cache_path.unlink()
            return False

        return True

    def load_from_cache(self, adapter_name: str, **params) -> Optional[List[Dict]]:
        """Load query result from cache.

        Args:
            adapter_name: Name of adapter
            **params: Query parameters

        Returns:
            Cached events or None if not cached
        """
        cache_path = self.get_cache_path(adapter_name, **params)

        if not self.is_cached(adapter_name, **params):
            return None

        try:
            with open(cache_path, 'r') as f:
                data = json.load(f)
                logger.info(f"Loaded {len(data)} events from cache")
                return data
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Cache read failed: {e}")
            cache_path.unlink()
            return None

    def save_to_cache(self, events: List[Dict], adapter_name: str, **params) -> None:
        """Save query result to cache.

        Args:
            events: Events to cache
            adapter_name: Name of adapter
            **params: Query parameters
        """
        cache_path = self.get_cache_path(adapter_name, **params)

        try:
            with open(cache_path, 'w') as f:
                json.dump(events, f, indent=2, default=str)
            logger.info(f"Cached {len(events)} events to {cache_path}")
        except IOError as e:
            logger.warning(f"Cache write failed: {e}")

    def clear_cache(self, adapter_name: Optional[str] = None) -> int:
        """Clear cache files.

        Args:
            adapter_name: If provided, only clear this adapter's cache

        Returns:
            Number of files deleted
        """
        if adapter_name:
            pattern = f"{adapter_name}_*.json"
        else:
            pattern = "*.json"

        deleted = 0
        for cache_file in self.cache_dir.glob(pattern):
            cache_file.unlink()
            deleted += 1

        logger.info(f"Cleared {deleted} cache files")
        return deleted


def calculate_distance_km(lat1: float, lon1: float,
                          lat2: float, lon2: float) -> float:
    """Calculate great-circle distance between two points.

    Uses Haversine formula.

    Args:
        lat1, lon1: First point (degrees)
        lat2, lon2: Second point (degrees)

    Returns:
        Distance in kilometers
    """
    # Convert to radians
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)

    # Haversine formula
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))

    # Earth radius in km
    r = 6371.0

    return r * c


def deduplicate_events(events: List[Dict],
                       time_threshold_seconds: float = 60.0,
                       distance_threshold_km: float = 10.0) -> List[Dict]:
    """Remove duplicate events based on time and location.

    Two events are considered duplicates if:
    - Time difference < time_threshold_seconds
    - Distance < distance_threshold_km

    When duplicates found, keeps the one with largest magnitude.

    Args:
        events: List of event dicts with 'date', 'lat', 'lon', 'magnitude'
        time_threshold_seconds: Maximum time difference for duplicates (seconds)
        distance_threshold_km: Maximum distance for duplicates (km)

    Returns:
        Deduplicated list of events
    """
    if len(events) <= 1:
        return events

    # Sort by time
    sorted_events = sorted(events, key=lambda e: e['date'])

    unique_events = []
    skip_indices = set()

    for i, event1 in enumerate(sorted_events):
        if i in skip_indices:
            continue

        # Check for duplicates
        duplicates = [event1]

        for j in range(i + 1, len(sorted_events)):
            if j in skip_indices:
                continue

            event2 = sorted_events[j]

            # Time check
            if isinstance(event1['date'], datetime) and isinstance(event2['date'], datetime):
                time_diff = abs((event2['date'] - event1['date']).total_seconds())
            else:
                continue

            if time_diff > time_threshold_seconds:
                break  # Events sorted by time, no more duplicates possible

            # Distance check
            dist = calculate_distance_km(
                event1['lat'], event1['lon'],
                event2['lat'], event2['lon']
            )

            if dist < distance_threshold_km:
                duplicates.append(event2)
                skip_indices.add(j)

        # Keep event with the largest magnitude
        best_event = max(duplicates, key=lambda e: e.get('magnitude', 0))
        unique_events.append(best_event)

    n_removed = len(events) - len(unique_events)
    if n_removed > 0:
        logger.info(f"Removed {n_removed} duplicate events")

    return unique_events
