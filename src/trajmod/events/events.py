"""Event catalog management for SSE and earthquake events."""
import datetime
import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Iterator

logger = logging.getLogger(__name__)


@dataclass
class SSEEvent:
    """Slow Slip Event representation."""
    start: datetime.datetime
    end: datetime.datetime
    lat: float
    lon: float
    magnitude: float
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.end < self.start:
            raise ValueError("SSE end time must be after start time")

    @property
    def duration_days(self) -> float:
        """Duration of SSE in days."""
        return (self.end - self.start).total_seconds() / 86400.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'start': self.start,
            'end': self.end,
            'lat': self.lat,
            'lon': self.lon,
            'magnitude': self.magnitude,
            'start_day': self.start,
            'end_day': self.end,
            **self.metadata
        }


@dataclass
class EarthquakeEvent:
    """Earthquake event representation."""
    date: datetime.datetime
    lat: float
    lon: float
    magnitude: float
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'date': self.date,
            'lat': self.lat,
            'lon': self.lon,
            'magnitude': self.magnitude,
            'eq_day': self.date,
            **self.metadata
        }


class EventCatalog:
    """Container for earthquake events with filtering capabilities.

    This is a lightweight container that holds events and provides
    filtering/manipulation methods. It does NOT fetch data - use
    CatalogFetcher for that.

    Responsibilities:
    - Hold events
    - Filter events (distance, magnitude, depth, date)
    - Provide utilities (summary, iteration, sorting)

    Example:
        >>> from trajmod.events import CatalogFetcher, EventCatalog
        >>>
        >>> # Fetch events
        >>> fetcher = CatalogFetcher()
        >>> events = fetcher.fetch_usgs(
        ...     lat=35.0, lon=140.0, radius_km=500,
        ...     start_date="2020-01-01", end_date="2023-12-31"
        ... )
        >>>
        >>> # Create catalog and filter
        >>> catalog = EventCatalog(events)
        >>> catalog.filter_by_distance(35.0, 140.0, max_km=300)
        >>> catalog.filter_by_magnitude(min_mag=6.0)
        >>>
        >>> # Use in model
        >>> model = TrajectoryModel(..., eq_catalog=catalog.events)
    """

    def __init__(self, events: List[Dict]):
        """Initialize event catalog.

        Args:
            events: List of event dicts in trajmod standard format
                Each event should have: date, lat, lon, magnitude, depth_km

        Example:
            >>> events = [
            ...     {'date': datetime(...), 'lat': 35.0, 'lon': 140.0,
            ...      'magnitude': 7.2, 'depth_km': 10.0}
            ... ]
            >>> catalog = EventCatalog(events)
        """
        self.events = events
        logger.debug(f"EventCatalog created with {len(events)} events")

    @staticmethod
    def merge_close_events(events: List[Dict], gap_threshold: int = 5) -> List[Dict]:
        """Merge temporally close SSE events to reduce multicollinearity.

        Args:
            events: List of SSE event dicts with 'start_day', 'end_day', 'magnitude'
            gap_threshold: Maximum gap (days) to merge events

        Returns:
            List of merged SSE events (same format as input)
        """
        if not events:
            return []

        # Sort by start time
        sorted_events = sorted(events, key=lambda e: e['start_day'])
        merged = [sorted_events[0].copy()]

        for event in sorted_events[1:]:
            last = merged[-1]

            # Calculate gap in days
            gap = (event['start_day'] - last['end_day']).days

            if gap <= gap_threshold:
                # Merge: keep event with larger magnitude
                if event.get('magnitude', 0) > last.get('magnitude', 0):
                    # Keep newer event's properties (except times)
                    last.update({k: v for k, v in event.items()
                                 if k not in ['start_day', 'end_day', 'start', 'end']})

                # Extend time range to cover both events
                last['end_day'] = max(last['end_day'], event['end_day'])
                if 'end' in last and 'end' in event:
                    last['end'] = max(last['end'], event['end'])

                logger.debug(f"Merged SSE: gap={gap}d, "
                             f"{last['start_day']} to {last['end_day']}")
            else:
                merged.append(event.copy())

        logger.info(f"Merged SSEs: {len(events)} --> {len(merged)}")
        return merged

    @staticmethod
    def merge_earthquakes_by_date(earthquakes: List[Dict]) -> List[Dict]:
        """Merge earthquakes occurring on the same day.

        Strategy: Keep the largest magnitude earthquake, discard others.

        Args:
            earthquakes: List of earthquake event dicts

        Returns:
            List of merged earthquake events (standard trajmod format)
        """
        if not earthquakes:
            return []

        from collections import defaultdict
        events_by_date = defaultdict(list)

        # Group by date
        for eq in earthquakes:
            # Handle both 'date' and 'eq_day' keys
            eq_time = eq.get('eq_day') or eq.get('date')
            if eq_time is None:
                logger.warning(f"Skipping earthquake without date: {eq}")
                continue

            date_only = eq_time.date()  # Get date part only
            events_by_date[date_only].append(eq)

        # Merge: keep event with largest magnitude per day
        merged_catalog = []
        for date_only, events in sorted(events_by_date.items()):
            if len(events) == 1:
                # Single event - keep as is
                merged_catalog.append(events[0])
            else:
                # Multiple events - keep largest
                largest = max(events, key=lambda e: e.get('magnitude', 0))

                # Update to use the date (not datetime)
                merged_event = largest.copy()
                if 'eq_day' in merged_event:
                    # Preserve eq_day but ensure it's a datetime
                    merged_event['eq_day'] = datetime.datetime.combine(
                        date_only, datetime.time()
                    )
                if 'date' in merged_event:
                    merged_event['date'] = datetime.datetime.combine(
                        date_only, datetime.time()
                    )

                merged_catalog.append(merged_event)

                logger.debug(f"Merged {len(events)} earthquakes on {date_only}, "
                             f"kept M{largest.get('magnitude', 0):.1f}")

        logger.info(f"Merged earthquakes: {len(earthquakes)} --> {len(merged_catalog)}")
        return merged_catalog

    # ========================================================================
    # FILTERING METHODS (return self for chaining)
    # ========================================================================

    def filter_by_distance(self,
                           station_lat: float,
                           station_lon: float,
                           max_km: float) -> 'EventCatalog':
        """Filter events by distance from station.

        Args:
            station_lat: Station latitude (degrees)
            station_lon: Station longitude (degrees)
            max_km: Maximum distance (km)

        Returns:
            Self (for method chaining)

        Example:
            >>> catalog.filter_by_distance(35.0, 140.0, max_km=300)
        """
        from trajmod.events.catalogs.utils import calculate_distance_km

        original_count = len(self.events)

        self.events = [
            event for event in self.events
            if calculate_distance_km(
                station_lat, station_lon,
                event['lat'], event['lon']
            ) <= max_km
        ]

        filtered_count = original_count - len(self.events)
        logger.info(f"Distance filter: kept {len(self.events)}/{original_count} events "
                    f"(removed {filtered_count})")

        return self

    def filter_by_magnitude(self,
                            min_mag: Optional[float] = None,
                            max_mag: Optional[float] = None) -> 'EventCatalog':
        """Filter events by magnitude range.

        Args:
            min_mag: Minimum magnitude (inclusive)
            max_mag: Maximum magnitude (inclusive)

        Returns:
            Self (for method chaining)

        Example:
            >>> catalog.filter_by_magnitude(min_mag=6.0)
            >>> catalog.filter_by_magnitude(min_mag=5.0, max_mag=7.0)
        """
        original_count = len(self.events)

        if min_mag is not None:
            self.events = [e for e in self.events if e.get('magnitude', 0) >= min_mag]

        if max_mag is not None:
            self.events = [e for e in self.events if e.get('magnitude', 0) <= max_mag]

        filtered_count = original_count - len(self.events)
        logger.info(f"Magnitude filter: kept {len(self.events)}/{original_count} events "
                    f"(removed {filtered_count})")

        return self

    def filter_by_depth(self,
                        min_depth_km: Optional[float] = None,
                        max_depth_km: Optional[float] = None) -> 'EventCatalog':
        """Filter events by depth range.

        Args:
            min_depth_km: Minimum depth (km, inclusive)
            max_depth_km: Maximum depth (km, inclusive)

        Returns:
            Self (for method chaining)

        Example:
            >>> catalog.filter_by_depth(max_depth_km=50)  # Shallow events only
        """
        original_count = len(self.events)

        if min_depth_km is not None:
            self.events = [e for e in self.events if e.get('depth_km', 0) >= min_depth_km]

        if max_depth_km is not None:
            self.events = [e for e in self.events if e.get('depth_km', 0) <= max_depth_km]

        filtered_count = original_count - len(self.events)
        logger.info(f"Depth filter: kept {len(self.events)}/{original_count} events")

        return self

    def filter_by_date_range(self,
                             start_date: Optional[datetime] = None,
                             end_date: Optional[datetime] = None) -> 'EventCatalog':
        """Filter events by date range.

        Args:
            start_date: Start date (inclusive)
            end_date: End date (inclusive)

        Returns:
            Self (for method chaining)

        Example:
            >>> catalog.filter_by_date_range(
            ...     start_date=datetime(2020, 1, 1),
            ...     end_date=datetime(2023, 12, 31)
            ... )
        """
        original_count = len(self.events)

        filtered_events = []
        for event in self.events:
            event_date = event.get('date')
            if not isinstance(event_date, datetime):
                continue

            if start_date is not None and event_date < start_date:
                continue

            if end_date is not None and event_date > end_date:
                continue

            filtered_events.append(event)

        self.events = filtered_events
        filtered_count = original_count - len(self.events)
        logger.info(f"Date filter: kept {len(self.events)}/{original_count} events")

        return self

    # ========================================================================
    # MANIPULATION METHODS
    # ========================================================================

    def remove_duplicates(self,
                          time_threshold_seconds: float = 60.0,
                          distance_threshold_km: float = 10.0) -> 'EventCatalog':
        """Remove duplicate events.

        Two events are considered duplicates if:
        - Time difference < time_threshold_seconds
        - Distance < distance_threshold_km

        When duplicates found, keeps the one with largest magnitude.

        Args:
            time_threshold_seconds: Max time difference for duplicates (seconds)
            distance_threshold_km: Max distance for duplicates (km)

        Returns:
            Self (for method chaining)

        Example:
            >>> catalog.remove_duplicates()
        """
        from trajmod.events.catalogs.utils import deduplicate_events

        original_count = len(self.events)
        self.events = deduplicate_events(
            self.events, time_threshold_seconds, distance_threshold_km
        )

        removed_count = original_count - len(self.events)
        if removed_count > 0:
            logger.info(f"Removed {removed_count} duplicate events")

        return self

    def sort_by_date(self, reverse: bool = False) -> 'EventCatalog':
        """Sort events by date.

        Args:
            reverse: Sort in descending order (newest first)

        Returns:
            Self (for method chaining)
        """
        self.events = sorted(self.events, key=lambda e: e.get('date'), reverse=reverse)
        logger.debug(f"Sorted by date (reverse={reverse})")
        return self

    def sort_by_magnitude(self, reverse: bool = True) -> 'EventCatalog':
        """Sort events by magnitude.

        Args:
            reverse: Sort in descending order (largest first, default)

        Returns:
            Self (for method chaining)
        """
        self.events = sorted(self.events, key=lambda e: e.get('magnitude', 0), reverse=reverse)
        logger.debug(f"Sorted by magnitude (reverse={reverse})")
        return self

    # ========================================================================
    # UTILITY METHODS
    # ========================================================================

    def summary(self) -> str:
        """Get summary statistics.

        Returns:
            Multi-line summary string

        Example:
            >>> print(catalog.summary())
        """
        if not self.events:
            return "EventCatalog: Empty"

        mags = [e.get('magnitude', 0) for e in self.events]
        depths = [e.get('depth_km', 0) for e in self.events]

        dates = [e.get('date') for e in self.events if isinstance(e.get('date'), datetime)]
        if dates:
            date_range = f"{min(dates).date()} to {max(dates).date()}"
        else:
            date_range = "Unknown"

        sources = list(set(e.get('source', 'unknown') for e in self.events))

        return f"""EventCatalog Summary:
  Events: {len(self.events)}
  Magnitude: {min(mags):.1f} - {max(mags):.1f}
  Depth: {min(depths):.1f} - {max(depths):.1f} km
  Date range: {date_range}
  Sources: {', '.join(sources)}
"""

    def copy(self) -> 'EventCatalog':
        """Create a copy of this catalog.

        Returns:
            New EventCatalog with copied events
        """
        return EventCatalog(self.events.copy())

    # ========================================================================
    # UTILITY METHODS
    # ========================================================================

    def __len__(self) -> int:
        """Return number of events."""
        return len(self.events)

    def __iter__(self) -> Iterator[Dict]:
        """Iterate over events."""
        return iter(self.events)

    def __getitem__(self, index: int) -> Dict:
        """Get event by index."""
        return self.events[index]

    def __repr__(self) -> str:
        """String representation."""
        return f"EventCatalog(n_events={len(self.events)})"

    def __str__(self) -> str:
        """String representation."""
        return self.summary()


class SSECatalog:
    """Catalog of Slow Slip Events."""

    def __init__(self, events: Optional[List[SSEEvent]] = None):
        self.events = events or []

    def add_event(self, event: SSEEvent) -> None:
        self.events.append(event)

    def to_dict_list(self) -> List[Dict]:
        return [event.to_dict() for event in self.events]

    @classmethod
    def from_dict_list(cls, dict_list: List[Dict]) -> 'SSECatalog':
        events = []
        for d in dict_list:
            events.append(SSEEvent(
                start=d['start'],
                end=d['end'],
                lat=d['lat'],
                lon=d['lon'],
                magnitude=d['magnitude'],
                metadata={k: v for k, v in d.items()
                          if k not in ['start', 'end', 'lat', 'lon', 'magnitude']}
            ))
        return cls(events)


class EarthquakeCatalog:
    """Catalog of earthquake events."""

    def __init__(self, events: Optional[List[EarthquakeEvent]] = None):
        self.events = events or []

    def add_event(self, event: EarthquakeEvent) -> None:
        self.events.append(event)

    def to_dict_list(self) -> List[Dict]:
        return [event.to_dict() for event in self.events]

    @classmethod
    def from_dict_list(cls, dict_list: List[Dict]) -> 'EarthquakeCatalog':
        events = []
        for d in dict_list:
            events.append(EarthquakeEvent(
                date=d['date'],
                lat=d['lat'],
                lon=d['lon'],
                magnitude=d['magnitude'],
                metadata={k: v for k, v in d.items()
                          if k not in ['date', 'lat', 'lon', 'magnitude']}
            ))
        return cls(events)
