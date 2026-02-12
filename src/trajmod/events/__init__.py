"""Event catalog and selection module."""
from trajmod.events.catalogs import CatalogAdapter, USGSAdapter, CatalogFetchError
from trajmod.events.catalogs.catalog_fetcher import CatalogFetcher
from trajmod.events.catalogs.validation import validate_earthquake_catalog, validate_sse_catalog, \
    print_validation_summary
from trajmod.events.event_selection import (
    EventSelector,
    KneeEventSelector,
    LassoEventSelector, MarginalScreeningSelector, ThresholdEventSelector,
)
from trajmod.events.events import EventCatalog

__all__ = [
    # Main classes
    "CatalogFetcher",
    "EventCatalog",

    # Event selection
    "EventSelector",
    "KneeEventSelector",
    "LassoEventSelector",
    "MarginalScreeningSelector",
    "ThresholdEventSelector",

    # Advanced (catalog adapters)
    "CatalogAdapter",
    "USGSAdapter",
    "CatalogFetchError",

    # catalog validation
    "validate_earthquake_catalog",
    "validate_sse_catalog",
    "print_validation_summary"
]
