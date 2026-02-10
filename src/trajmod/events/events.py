"""Event catalog management for SSE and earthquake events."""
import datetime
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict, Any, Optional


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
    """Base class for event catalogs."""

    @staticmethod
    def merge_close_events(events: List[Dict], gap_threshold: int = 5) -> List[Dict]:
        """Merge temporally close events to reduce multicollinearity."""
        if not events:
            return []

        sorted_events = sorted(events, key=lambda e: e['start_day'])
        merged = [sorted_events[0].copy()]

        for event in sorted_events[1:]:
            last = merged[-1]
            gap = (event['start_day'] - last['end_day']).days

            if gap <= gap_threshold:
                if event.get('magnitude', 0) > last.get('magnitude', 0):
                    last.update({k: v for k, v in event.items()
                                 if k not in ['start_day', 'end_day']})
                last['end_day'] = max(last['end_day'], event['end_day'])
            else:
                merged.append(event.copy())

        return merged

    @staticmethod
    def merge_earthquakes_by_date(earthquakes: List[Dict]) -> List[Dict]:
        """Merge earthquakes occurring on the same day."""
        if not earthquakes:
            return []

        events_by_date = defaultdict(list)

        for eq in earthquakes:
            date_only = eq['date'].date()
            events_by_date[date_only].append(eq)

        merged_catalog = []
        for date_only, events in sorted(events_by_date.items()):
            merged_catalog.append({
                'date': date_only,
                'latitude': [eq['lat'] for eq in events],
                'longitude': [eq['lon'] for eq in events],
                'magnitude': [eq['magnitude'] for eq in events]
            })

        return merged_catalog


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
