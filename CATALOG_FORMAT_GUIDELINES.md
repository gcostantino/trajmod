
================================================================================
TRAJMOD CATALOG FORMAT GUIDELINES
================================================================================

This guide ensures your earthquake and SSE catalogs work seamlessly with
trajmod's TrajectoryModel.


================================================================================
1. EARTHQUAKE CATALOG FORMAT
================================================================================

Required Fields (all events MUST have these):
──────────────────────────────────────────────

{
    "time": datetime.datetime,     # Event datetime (UTC recommended)
    "lat": float,                  # Latitude in degrees [-90, 90]
    "lon": float,                  # Longitude in degrees [-180, 180]
    "magnitude": float             # Magnitude (any scale: Mw, ML, etc.)
}

Alternative Time Field Names (trajmod accepts):
────────────────────────────────────────────────
- "time" (preferred)
- "date"
- "datetime"
- "eq_day" (added automatically by trajmod)


Optional Fields (recommended):
───────────────────────────────

{
    "depth": float,                # Depth in km (positive down)
    "event_id": str,               # Unique event identifier
    "source": str,                 # Catalog source (e.g., "USGS", "ISC")
    "magnitude_type": str,         # Type (e.g., "mw", "mb", "ml")
    "event_type": str              # Type (e.g., "earthquake", "antenna_change")
}


Complete Example:
─────────────────

import datetime

earthquake_catalog = [
    {
        "time": datetime.datetime(2019, 7, 6, 3, 19, 53),
        "lat": 35.77,
        "lon": -117.599,
        "magnitude": 7.1,
        "depth": 8.0,
        "event_id": "ci38457511",
        "source": "USGS",
        "magnitude_type": "mw"
    },
    {
        "time": datetime.datetime(2020, 1, 28, 19, 10, 23),
        "lat": 19.410,
        "lon": -77.933,
        "magnitude": 7.7,
        "depth": 10.0,
        "event_id": "us70007b0w",
        "source": "USGS",
        "magnitude_type": "mw"
    }
]


Using USGS Catalog Fetcher (recommended):
──────────────────────────────────────────

from trajmod.events import CatalogFetcher

fetcher = CatalogFetcher()
earthquake_catalog = fetcher.fetch_usgs(
    lat=35.0, lon=140.0, radius_km=500,
    start_date="2015-01-01", end_date="2020-12-31",
    min_magnitude=5.0
)

# Already in correct format! ✓


================================================================================
2. SSE (SLOW SLIP EVENT) CATALOG FORMAT
================================================================================

Required Fields (all events MUST have these):
──────────────────────────────────────────────

{
    "start": datetime.datetime,    # SSE start time
    "end": datetime.datetime,      # SSE end time (must be > start)
    "lat": float,                  # Latitude in degrees [-90, 90]
    "lon": float,                  # Longitude in degrees [-180, 180]
    "magnitude": float             # Moment magnitude (or equivalent)
}

Alternative Time Field Names (trajmod accepts):
────────────────────────────────────────────────
- "start" (preferred) or "start_time"
- "end" (preferred) or "end_time"
- "start_day", "end_day" (added automatically by trajmod)


Optional Fields (recommended):
───────────────────────────────

{
    "event_id": str,               # Unique event identifier
    "source": str,                 # Catalog source
    "duration_days": float,        # Duration (computed automatically)
    "slip": float,                 # Slip amount (cm or m, specify in metadata)
    "area": float                  # Rupture area (km²)
}


Complete Example:
─────────────────

import datetime

sse_catalog = [
    {
        "start": datetime.datetime(2018, 5, 10),
        "end": datetime.datetime(2018, 7, 15),
        "lat": 36.5,
        "lon": 140.8,
        "magnitude": 6.3,
        "event_id": "SSE_2018_Boso",
        "source": "NIED",
        "duration_days": 66
    },
    {
        "start": datetime.datetime(2019, 11, 20),
        "end": datetime.datetime(2020, 1, 10),
        "lat": 36.6,
        "lon": 140.9,
        "magnitude": 6.1,
        "event_id": "SSE_2019_Boso",
        "source": "NIED",
        "duration_days": 51
    }
]


================================================================================
3. COMMON MISTAKES TO AVOID
================================================================================

❌ DON'T: Use strings for datetime
───────────────────────────────────

BAD:
{
    "time": "2019-07-06",          # String!
    "lat": 35.77,
    "lon": -117.599,
    "magnitude": 7.1
}

GOOD:
{
    "time": datetime.datetime(2019, 7, 6),  # datetime object!
    "lat": 35.77,
    "lon": -117.599,
    "magnitude": 7.1
}


❌ DON'T: Use date objects (use datetime)
──────────────────────────────────────────

BAD:
{
    "time": datetime.date(2019, 7, 6),     # date, not datetime!
}

GOOD:
{
    "time": datetime.datetime(2019, 7, 6), # datetime!
}


❌ DON'T: Mix coordinate conventions
─────────────────────────────────────

BAD:
{
    "latitude": 35.77,             # Wrong key name!
    "longitude": -117.599,         # Wrong key name!
}

GOOD:
{
    "lat": 35.77,                  # Correct!
    "lon": -117.599,               # Correct!
}


❌ DON'T: Use inconsistent magnitude fields
────────────────────────────────────────────

BAD (multiple magnitude fields):
{
    "magnitude": 7.1,
    "mag": 7.0,                    # Confusing!
    "mw": 7.1
}

GOOD:
{
    "magnitude": 7.1,              # Single magnitude field
    "magnitude_type": "mw"         # Specify type if needed
}


❌ DON'T: Forget timezone (use UTC)
────────────────────────────────────

BAD:
{
    "time": datetime.datetime(2019, 7, 6, 3, 19, 53),  # What timezone?
}

BETTER:
{
    "time": datetime.datetime(2019, 7, 6, 3, 19, 53,
                             tzinfo=datetime.timezone.utc),  # Explicit UTC
}


❌ DON'T: Have SSE end before start
─────────────────────────────────────

BAD:
{
    "start": datetime.datetime(2018, 7, 15),
    "end": datetime.datetime(2018, 5, 10),  # Before start!
}


================================================================================
4. DATA TYPE REFERENCE
================================================================================

Python Types:
─────────────
- datetime: Use datetime.datetime (NOT datetime.date)
- lat/lon: Use float (NOT int)
- magnitude: Use float (NOT int)
- event_id: Use str

Value Ranges:
─────────────
- Latitude: -90 to +90 (degrees)
- Longitude: -180 to +180 (degrees, or 0 to 360)
- Magnitude: Typically 0 to 10 (but no hard limit)
- Depth: Typically 0 to 700 km (positive = down)


================================================================================
5. CONVERTING FROM COMMON FORMATS
================================================================================

From Pandas DataFrame:
──────────────────────

import pandas as pd
import datetime

df = pd.read_csv("earthquakes.csv")

# Convert to trajmod format
earthquake_catalog = []
for _, row in df.iterrows():
    earthquake_catalog.append({
        "time": pd.to_datetime(row["datetime"]).to_pydatetime(),
        "lat": float(row["latitude"]),
        "lon": float(row["longitude"]),
        "magnitude": float(row["mag"]),
        "depth": float(row["depth"]) if "depth" in row else None,
        "event_id": str(row["id"]) if "id" in row else None
    })


From CSV with string dates:
────────────────────────────

import csv
import datetime

earthquake_catalog = []
with open("earthquakes.csv", "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        earthquake_catalog.append({
            "time": datetime.datetime.strptime(
                row["datetime"], "%Y-%m-%d %H:%M:%S"
            ),
            "lat": float(row["lat"]),
            "lon": float(row["lon"]),
            "magnitude": float(row["magnitude"])
        })


From QuakeML or other formats:
───────────────────────────────

# Use ObsPy for seismological formats
from obspy import read_events

catalog = read_events("events.xml")  # QuakeML
earthquake_catalog = []
for event in catalog:
    origin = event.preferred_origin() or event.origins[0]
    magnitude = event.preferred_magnitude() or event.magnitudes[0]

    earthquake_catalog.append({
        "time": origin.time.datetime,
        "lat": origin.latitude,
        "lon": origin.longitude,
        "magnitude": magnitude.mag,
        "depth": origin.depth / 1000.0,  # Convert m to km
        "event_id": str(event.resource_id)
    })


================================================================================
6. VALIDATION BEFORE USE
================================================================================

Always validate your catalog before using it:

from trajmod.events import validate_earthquake_catalog, validate_sse_catalog

# Validate earthquake catalog
is_valid, errors = validate_earthquake_catalog(earthquake_catalog)
if not is_valid:
    print("Catalog errors:")
    for error in errors:
        print(f"  - {error}")

# Validate SSE catalog
is_valid, errors = validate_sse_catalog(sse_catalog)
if not is_valid:
    print("Catalog errors:")
    for error in errors:
        print(f"  - {error}")


================================================================================
7. QUICK START EXAMPLE
================================================================================

import datetime
from trajmod import TrajectoryModel, ModelConfig

# 1. Create earthquake catalog
earthquake_catalog = [
    {
        "time": datetime.datetime(2019, 7, 6, 3, 19, 53),
        "lat": 35.77,
        "lon": -117.599,
        "magnitude": 7.1
    }
]

# 2. Create SSE catalog (if applicable)
sse_catalog = [
    {
        "start": datetime.datetime(2018, 5, 10),
        "end": datetime.datetime(2018, 7, 15),
        "lat": 36.5,
        "lon": 140.8,
        "magnitude": 6.3
    }
]

# 3. Load your GNSS time series
# t, y, sigma_y = ...

# 4. Create model
model = TrajectoryModel(
    t=t, y=y, sigma_y=sigma_y,
    station_lat=35.0,
    station_lon=140.0,
    eq_catalog=earthquake_catalog,
    sse_catalog=sse_catalog,
    config=ModelConfig()
)

# 5. Fit
results = model.fit()


================================================================================
8. TROUBLESHOOTING
================================================================================

Error: "KeyError: 'time'"
─────────────────────────
→ Make sure your catalog uses "time" or "date" key
→ Run validate_earthquake_catalog() to check

Error: "'str' object has no attribute 'year'"
──────────────────────────────────────────────
→ You're using strings instead of datetime objects
→ Convert: datetime.datetime.strptime(date_str, format)

Error: "SSE end time must be after start time"
───────────────────────────────────────────────
→ Check your start/end fields
→ Make sure end > start

Error: "Skipping earthquake without datetime"
─────────────────────────────────────────────
→ Some events have missing time field
→ Check your catalog for None values

Error: "TypeError: unsupported operand type(s)"
───────────────────────────────────────────────
→ Wrong data types (probably string dates)
→ Convert all dates to datetime.datetime


================================================================================
9. BEST PRACTICES
================================================================================

✅ DO:
- Use datetime.datetime for all times
- Use float for lat, lon, magnitude
- Include event_id for traceability
- Validate catalogs before use
- Use UTC timezone
- Document coordinate system (usually WGS84)

✅ PREFER:
- USGS CatalogFetcher for earthquakes (automatic formatting)
- Consistent magnitude scale within catalog
- Complete metadata (depth, event_id, source)

⚠️ CONSIDER:
- Filtering events by temporal range before passing to model
- Spatial filtering if using large catalogs
- Merging duplicate events


================================================================================
10. ADDITIONAL RESOURCES
================================================================================

Trajmod Documentation:
- Catalog fetching: trajmod/events/catalog_fetcher.py
- Event validation: trajmod/events/validation.py
- Examples: trajmod/examples/

External Resources:
- USGS ComCat: https://earthquake.usgs.gov/data/comcat/
- ISC Catalog: http://www.isc.ac.uk/iscbulletin/
- ObsPy (seismology): https://docs.obspy.org/


================================================================================
