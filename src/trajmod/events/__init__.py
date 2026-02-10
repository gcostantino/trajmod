"""Event catalog and selection module."""

from trajmod.events.events import EventCatalog
from trajmod.events.event_selection import (
    EventSelector,
    KneeEventSelector,
    LassoEventSelector,
)

__all__ = [
    "EventCatalog",
    "EventSelector",
    "KneeEventSelector",
    "LassoEventSelector",
]