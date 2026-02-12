# 📖 Trajmod Catalog Format Guidelines

> **Comprehensive guide to earthquake and SSE catalog formatting for trajmod**

Ensure your earthquake and SSE catalogs work seamlessly with trajmod's `TrajectoryModel`.

---

## 📋 Table of Contents

- [Earthquake Catalog Format](#1-earthquake-catalog-format)
- [SSE Catalog Format](#2-sse-slow-slip-event-catalog-format)
- [Common Mistakes](#3-common-mistakes-to-avoid)
- [Data Type Reference](#4-data-type-reference)
- [Format Conversion](#5-converting-from-common-formats)
- [Validation](#6-validation-before-use)
- [Quick Start](#7-quick-start-example)
- [Troubleshooting](#8-troubleshooting)
- [Best Practices](#9-best-practices)
- [Resources](#10-additional-resources)

---

## 1. Earthquake Catalog Format

### Required Fields

All earthquake events **must** include these fields:

| Field | Type | Description | Range |
|-------|------|-------------|-------|
| `time` | `datetime.datetime` | Event datetime (UTC recommended) | - |
| `lat` | `float` | Latitude in degrees | [-90, 90] |
| `lon` | `float` | Longitude in degrees | [-180, 180] |
| `magnitude` | `float` | Magnitude (any scale) | typically [0, 10] |

**Example:**
```python
{
    "time": datetime.datetime(2019, 7, 6, 3, 19, 53),
    "lat": 35.77,
    "lon": -117.599,
    "magnitude": 7.1
}
```

### Alternative Time Field Names

Trajmod accepts multiple time field names:
- `"time"` (preferred)
- `"date"`
- `"datetime"`
- `"eq_day"` (added automatically by trajmod)

### Optional Fields

| Field | Type | Description |
|-------|------|-------------|
| `depth` | `float` | Depth in km (positive down) |
| `event_id` | `str` | Unique event identifier |
| `source` | `str` | Catalog source (e.g., "USGS", "ISC") |
| `magnitude_type` | `str` | Magnitude type (e.g., "mw", "mb", "ml") |
| `event_type` | `str` | Event type (e.g., "earthquake", "antenna_change") |

### Complete Example

```python
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
```

### 🚀 Quick Method: USGS Catalog Fetcher

> **Recommended:** Use the built-in fetcher for automatic formatting!

```python
from trajmod.events import CatalogFetcher

fetcher = CatalogFetcher()
earthquake_catalog = fetcher.fetch_usgs(
    lat=35.0, 
    lon=140.0, 
    radius_km=500,
    start_date="2015-01-01", 
    end_date="2020-12-31",
    min_magnitude=5.0
)
# ✓ Already in correct format!
```

---

## 2. SSE (Slow Slip Event) Catalog Format

### Required Fields

All SSE events **must** include these fields:

| Field | Type | Description | Constraint |
|-------|------|-------------|------------|
| `start` | `datetime.datetime` | SSE start time | - |
| `end` | `datetime.datetime` | SSE end time | `end > start` |
| `lat` | `float` | Latitude in degrees | [-90, 90] |
| `lon` | `float` | Longitude in degrees | [-180, 180] |
| `magnitude` | `float` | Moment magnitude | - |

**Example:**
```python
{
    "start": datetime.datetime(2018, 5, 10),
    "end": datetime.datetime(2018, 7, 15),
    "lat": 36.5,
    "lon": 140.8,
    "magnitude": 6.3
}
```

### Alternative Time Field Names

Trajmod accepts:
- `"start"` / `"start_time"` (preferred)
- `"end"` / `"end_time"` (preferred)
- `"start_day"`, `"end_day"` (added automatically)

### Optional Fields

| Field | Type | Description |
|-------|------|-------------|
| `event_id` | `str` | Unique identifier |
| `source` | `str` | Catalog source |
| `duration_days` | `float` | Duration (auto-computed) |
| `slip` | `float` | Slip amount (specify units) |
| `area` | `float` | Rupture area (km²) |

### Complete Example

```python
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
```

---

## 3. Common Mistakes to Avoid

### ❌ Using Strings for Datetime

**BAD:**
```python
{
    "time": "2019-07-06",  # String!
    "lat": 35.77,
    "lon": -117.599,
    "magnitude": 7.1
}
```

**GOOD:**
```python
{
    "time": datetime.datetime(2019, 7, 6),  # datetime object
    "lat": 35.77,
    "lon": -117.599,
    "magnitude": 7.1
}
```

---

### ❌ Using `date` Objects

**BAD:**
```python
{
    "time": datetime.date(2019, 7, 6),  # date, not datetime!
}
```

**GOOD:**
```python
{
    "time": datetime.datetime(2019, 7, 6),  # datetime!
}
```

---

### ❌ Wrong Field Names

**BAD:**
```python
{
    "latitude": 35.77,     # Wrong!
    "longitude": -117.599  # Wrong!
}
```

**GOOD:**
```python
{
    "lat": 35.77,          # Correct
    "lon": -117.599        # Correct
}
```

---

### ❌ Inconsistent Magnitude Fields

**BAD:**
```python
{
    "magnitude": 7.1,
    "mag": 7.0,            # Confusing!
    "mw": 7.1
}
```

**GOOD:**
```python
{
    "magnitude": 7.1,           # Single field
    "magnitude_type": "mw"      # Specify type separately
}
```

---

### ❌ Missing Timezone

**BAD:**
```python
{
    "time": datetime.datetime(2019, 7, 6, 3, 19, 53)  # What timezone?
}
```

**BETTER:**
```python
{
    "time": datetime.datetime(
        2019, 7, 6, 3, 19, 53,
        tzinfo=datetime.timezone.utc  # Explicit UTC
    )
}
```

---

### ❌ SSE End Before Start

**BAD:**
```python
{
    "start": datetime.datetime(2018, 7, 15),
    "end": datetime.datetime(2018, 5, 10)  # Before start!
}
```

> **⚠️ Warning:** End time must be after start time!

---

## 4. Data Type Reference

### Python Types

| Field | Required Type | Notes |
|-------|--------------|-------|
| `time`, `start`, `end` | `datetime.datetime` | NOT `datetime.date` or `str` |
| `lat`, `lon` | `float` | NOT `int` |
| `magnitude` | `float` | NOT `int` |
| `event_id` | `str` | - |

### Value Ranges

| Field | Valid Range | Notes |
|-------|-------------|-------|
| **Latitude** | -90 to +90 | Degrees |
| **Longitude** | -180 to +180 | Degrees (0 to 360 also accepted) |
| **Magnitude** | typically 0 to 10 | No hard limit |
| **Depth** | typically 0 to 700 km | Positive = downward |

---

## 5. Converting from Common Formats

### From Pandas DataFrame

```python
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
```

### From CSV with String Dates

```python
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
```

### From QuakeML (using ObsPy)

```python
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
```

---

## 6. Validation Before Use

### Validate Your Catalogs

> **💡 Best Practice:** Always validate before using!

```python
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
```

---

## 7. Quick Start Example

```python
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
    t=t, 
    y=y, 
    sigma_y=sigma_y,
    station_lat=35.0,
    station_lon=140.0,
    eq_catalog=earthquake_catalog,
    sse_catalog=sse_catalog,
    config=ModelConfig()
)

# 5. Fit
results = model.fit()
```

---

## 8. Troubleshooting

### Common Errors and Solutions

| Error | Cause | Solution |
|-------|-------|----------|
| `KeyError: 'time'` | Missing time field | Use `"time"` or `"date"` key |
| `'str' object has no attribute 'year'` | String instead of datetime | Convert with `datetime.datetime.strptime()` |
| `SSE end time must be after start time` | End ≤ start | Verify `end > start` |
| `Skipping earthquake without datetime` | None values | Check for missing time fields |
| `TypeError: unsupported operand type(s)` | Wrong data types | Convert strings to `datetime.datetime` |

### Detailed Solutions

#### Error: `KeyError: 'time'`
```python
# ✓ Run validation to check
from trajmod.events import validate_earthquake_catalog
is_valid, errors = validate_earthquake_catalog(catalog)
```

#### Error: String dates
```python
# ✓ Convert strings to datetime
date_str = "2019-07-06 03:19:53"
dt = datetime.datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
```

#### Error: SSE timing
```python
# ✓ Verify temporal order
assert sse['end'] > sse['start'], "End must be after start!"
```

---

## 9. Best Practices

### ✅ DO

- Use `datetime.datetime` for all times
- Use `float` for lat, lon, magnitude
- Include `event_id` for traceability
- Validate catalogs before use
- Use UTC timezone consistently
- Document coordinate system (usually WGS84)

### ⭐ PREFER

- USGS `CatalogFetcher` for earthquakes (automatic formatting)
- Consistent magnitude scale within catalog
- Complete metadata (depth, event_id, source)

### ⚠️ CONSIDER

- Filtering events by temporal range before passing to model
- Spatial filtering if using large catalogs
- Merging duplicate events

---

## 10. Additional Resources

### Trajmod Documentation

- **Catalog Fetching:** `trajmod/events/catalog_fetcher.py`
- **Event Validation:** `trajmod/events/validation.py`
- **Examples:** `trajmod/examples/`

### External Resources

- [USGS ComCat](https://earthquake.usgs.gov/data/comcat/) - Earthquake catalog API
- [ISC Bulletin](http://www.isc.ac.uk/iscbulletin/) - International Seismological Centre
- [ObsPy Documentation](https://docs.obspy.org/) - Seismology tools for Python

---

## 📧 Need Help?

If you encounter issues not covered here, please:
1. Check the [trajmod documentation](https://github.com/gcostantino/trajmod)
2. Run catalog validation functions
3. Open an issue on [GitHub](https://github.com/gcostantino/trajmod/issues)

---

**Last updated:** February 2026  
**Version:** 2.0
