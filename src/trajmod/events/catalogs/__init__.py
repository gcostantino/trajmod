"""Catalog adapters for fetching earthquake data from various providers.

This module provides adapters for fetching earthquake catalogs from:
- USGS ComCat (global coverage)
- NIED F-net (Japan)
- JMA (Japan Meteorological Agency)
- ISC (International Seismological Centre)

Example:
    >>> from trajmod.events import EventCatalog
    >>> catalog = EventCatalog.from_usgs(
    ...     lat=35.0, lon=140.0, radius_km=500,
    ...     start_date="2020-01-01", end_date="2023-12-31",
    ...     min_magnitude=5.5
    ... )
"""

from trajmod.events.catalogs.base import CatalogAdapter, CatalogFetchError
from trajmod.events.catalogs.usgs import USGSAdapter

__all__ = [
    "CatalogAdapter",
    "CatalogFetchError",
    "USGSAdapter",
]

# Optional imports (regional catalogs may not be installed)
try:
    from trajmod.events.catalogs.nied import NIEDAdapter
    __all__.append("NIEDAdapter")
except ImportError:
    pass

try:
    from trajmod.events.catalogs.jma import JMAAdapter
    __all__.append("JMAAdapter")
except ImportError:
    pass

try:
    from trajmod.events.catalogs.isc import ISCAdapter
    __all__.append("ISCAdapter")
except ImportError:
    pass