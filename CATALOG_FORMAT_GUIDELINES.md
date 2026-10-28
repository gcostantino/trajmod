# Trajmod Catalog Format Guidelines

How to format earthquake, SSE and station-step catalogs for `TrajectoryModel`.

## Contents

1. [Earthquake catalog](#1-earthquake-catalog)
2. [SSE catalog](#2-sse-catalog)
3. [Step catalog](#3-step-catalog)
4. [Data types and ranges](#4-data-types-and-ranges)
5. [Common mistakes](#5-common-mistakes)
6. [Converting from other formats](#6-converting-from-other-formats)
7. [Validation](#7-validation)
8. [Quick start](#8-quick-start)
9. [Troubleshooting](#9-troubleshooting)
10. [Best practices](#10-best-practices)

---

## 1. Earthquake catalog

A catalog is a list of dictionaries, one per earthquake. Earthquakes get a co-seismic step and, if selected, a log, exponential or log+exp postseismic term. Events farther than the radius given by `d_param` (or `d_param_eq`) are dropped.

### Required fields

| Field | Type | Description | Range |
|---|---|---|---|
| `time` | `datetime.datetime` | Event time (UTC recommended) | |
| `lat` | `float` | Latitude, degrees | -90 to 90 |
| `lon` | `float` | Longitude, degrees | -180 to 180 |
| `magnitude` | `float` | Magnitude (any scale) | typically 0 to 10 |

Accepted aliases: `date`, `datetime` or `eq_day` for the time (`eq_day` is added automatically by trajmod); `latitude` and `longitude` for `lat` and `lon`.

### Optional fields

| Field | Type | Description |
|---|---|---|
| `depth` | `float` | Depth in km, positive down |
| `event_id` | `str` | Unique identifier |
| `source` | `str` | Catalog source (for example "USGS") |
| `magnitude_type` | `str` | Magnitude type (for example "mw") |

### Example

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
        "magnitude_type": "mw",
    },
]
```

### Fetching from catalogs

```python
from trajmod.events import CatalogFetcher

fetcher = CatalogFetcher()
earthquake_catalog = fetcher.fetch_usgs(
    lat=35.0, lon=140.0, radius_km=500,
    start_date="2015-01-01", end_date="2020-12-31",
    min_magnitude=5.0,
)
```

Fetchers for USGS, ISC, NIED and JMA are available (`fetch_usgs`, `fetch_isc`, `fetch_nied`, `fetch_jma`, each with a `_box` variant for a rectangular region). Their output is already in the correct format.

---

## 2. SSE catalog

### Required fields

| Field | Type | Description | Constraint |
|---|---|---|---|
| `start` | `datetime.datetime` | SSE start | |
| `end` | `datetime.datetime` | SSE end | `end > start` |
| `lat` | `float` | Latitude, degrees | -90 to 90 |
| `lon` | `float` | Longitude, degrees | -180 to 180 |
| `magnitude` | `float` | Moment magnitude | |

Accepted aliases: `start_time` and `start_day` for `start`, and the matching names for `end`. `start_day` and `end_day` are added automatically.

### Optional fields

| Field | Type | Description |
|---|---|---|
| `event_id` | `str` | Unique identifier |
| `source` | `str` | Catalog source |
| `duration_days` | `float` | Duration (computed automatically) |

### Example

```python
sse_catalog = [
    {
        "start": datetime.datetime(2018, 5, 10),
        "end": datetime.datetime(2018, 7, 15),
        "lat": 36.5,
        "lon": 140.8,
        "magnitude": 6.3,
        "event_id": "SSE_2018_Boso",
    },
]
```

### Timing optimisation

If `sse_optimize_timing=True`, the start and end of each SSE are optimised within a tolerance around the catalog values (`sse_timing_tol_days`, `sse_timing_tol_frac`, `sse_timing_tol_max_days`), starting from the catalog times each time. The catalog times should therefore be your best prior estimate, since the tolerance is centred on them.

---

## 3. Step catalog

Station-specific discontinuities that are not earthquakes (antenna, receiver, radome or monument changes) go in a separate list passed as `step_catalog`. They get one step column each. They are never filtered by distance, never get postseismic terms, and are not pruned unless you ask for it with `select_events(selector, event_type='step')`.

### Fields

| Field | Type | Description |
|---|---|---|
| `date` | `datetime.datetime` | Time of the step (required) |
| `event_type` | `str` | Free-text label, default `"step"` (for example `"antenna_change"`) |

Accepted aliases for `date`: `time`, `step_day`. `date` objects and `numpy.datetime64` are converted to `datetime`. Duplicates (same day and same label) are dropped.

### Example

```python
step_catalog = [
    {"date": datetime.datetime(2013, 5, 2), "event_type": "antenna_change"},
    {"date": datetime.datetime(2016, 8, 14), "event_type": "receiver_change"},
]

model = TrajectoryModel(..., eq_catalog=earthquake_catalog, step_catalog=step_catalog)
```

Columns are named `STEP_{i}_{date}_{label}`.

### Notes

- Two steps on the same day, or a step on the day of an earthquake, give identical columns. Merge them or drop one.
- Rows with `event_type` equal to `antenna_change`, `equipment_change` or `monument_change` found in `eq_catalog` are moved to the step catalog automatically, with a warning. Put them in `step_catalog` yourself.
- A step catalog is typically derived from a station log or a network-wide step file (one file for all stations, filtered by station code).

---

## 4. Data types and ranges

| Field | Type | Notes |
|---|---|---|
| `time`, `start`, `end`, `date` | `datetime.datetime` | Not `str`. `date` objects and strings are converted with a warning when validation is on |
| `lat`, `lon`, `magnitude` | `float` | |
| `event_id`, `source` | `str` | |
| `depth` | `float` | km, positive down |

Keep all catalog times and the time series `t` in the same convention. Naive datetimes in UTC are the safe choice.

---

## 5. Common mistakes

**String instead of datetime.** `"time": "2019-07-06"` is parsed with a warning if validation is on, but fix it at the source:

```python
{"time": datetime.datetime(2019, 7, 6), ...}
```

**`date` instead of `datetime`.** `datetime.date(2019, 7, 6)` is converted to midnight with a warning. Use `datetime.datetime`.

**Several magnitude fields.** `magnitude`, `mag` and `mw` together are ambiguous. Use `magnitude` and `magnitude_type`.

**SSE end before start.** `end` must be after `start`. The validator reports this as an error.

**Antenna changes in the earthquake catalog.** Use the step catalog (section 3).

---

## 6. Converting from other formats

### pandas DataFrame

```python
import pandas as pd

df = pd.read_csv("earthquakes.csv")
earthquake_catalog = [
    {
        "time": pd.to_datetime(row["datetime"]).to_pydatetime(),
        "lat": float(row["latitude"]),
        "lon": float(row["longitude"]),
        "magnitude": float(row["mag"]),
    }
    for _, row in df.iterrows()
]
```

### CSV with string dates

```python
import csv
import datetime

earthquake_catalog = []
with open("earthquakes.csv") as f:
    for row in csv.DictReader(f):
        earthquake_catalog.append({
            "time": datetime.datetime.strptime(row["datetime"], "%Y-%m-%d %H:%M:%S"),
            "lat": float(row["lat"]),
            "lon": float(row["lon"]),
            "magnitude": float(row["magnitude"]),
        })
```

### Step file for one station

```python
import numpy as np

steps = np.loadtxt("steps.txt", dtype=str)          # adapt to your file layout
station_steps = steps[steps[:, 0] == "STATION"]
step_catalog = [
    {"date": datetime.datetime.strptime(row, "%Y-%m-%d"), "event_type": "antenna_change"}[1]
    for row in station_steps
]
```

Adjust the column indices to your own step file. This block shows the pattern and was not tested.

---

## 7. Validation

Catalogs are validated automatically when a model is created (`TrajectoryModel(..., validate_catalogs=True)`). Fixable problems (string or `date` times, `latitude`/`longitude` names, duplicate steps) are corrected with a warning. Unfixable problems raise an error.

To check a catalog by hand:

```python
from trajmod.events import (
    validate_earthquake_catalog, validate_sse_catalog, validate_step_catalog,
)

is_valid, messages, fixed = validate_earthquake_catalog(earthquake_catalog, strict=False, fix=True)
if not is_valid:
    for msg in messages:
        print(" -", msg)
earthquake_catalog = fixed          # corrected copy

is_valid, messages, fixed = validate_sse_catalog(sse_catalog, strict=False, fix=True)
is_valid, messages, fixed = validate_step_catalog(step_catalog, fix=True)
```

All validators return `(is_valid, messages, fixed_catalog)`. With `strict=True` (the default for earthquakes and SSEs) problems are reported but not repaired. Without `fix=True`, `fixed_catalog` is `None`.

---

## 8. Quick start

```python
import datetime
from trajmod import TrajectoryModel, ModelConfig

earthquake_catalog = [
    {"time": datetime.datetime(2019, 7, 6, 3, 19, 53),
     "lat": 35.77, "lon": -117.599, "magnitude": 7.1},
]
sse_catalog = [
    {"start": datetime.datetime(2018, 5, 10), "end": datetime.datetime(2018, 7, 15),
     "lat": 36.5, "lon": 140.8, "magnitude": 6.3},
]
step_catalog = [
    {"date": datetime.datetime(2018, 3, 1), "event_type": "antenna_change"},
]

# t, y, sigma_y = ...   your GNSS time series (datetime array, mm, mm)

model = TrajectoryModel(
    t=t, y=y, sigma_y=sigma_y,
    station_lat=35.0, station_lon=140.0,
    eq_catalog=earthquake_catalog,
    sse_catalog=sse_catalog,
    step_catalog=step_catalog,
    config=ModelConfig(),
)
results = model.fit()
```

---

## 9. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Error mentioning a missing time field | No `time`, `date`, `datetime` or `eq_day` key | Add one, or run the validator to see which events are affected |
| Warning about auto-fixed string or `date` times | Wrong time type | Convert to `datetime.datetime` at the source |
| SSE error about end and start | `end` is not after `start` | Check the order |
| Earthquake not used | Outside the radius from `d_param`, or the magnitude/step gates | Check `d_param`, `d_param_eq`, `postseismic_mag_threshold` |
| Step not appearing | Wrong date or duplicate | Run `validate_step_catalog` and check the messages |
| Warning about moved rows | Antenna-type rows in `eq_catalog` | Move them to `step_catalog` |
| Identical columns | Two steps, or a step and an earthquake, on the same day | Merge or drop one |

---

## 10. Best practices

- Use `datetime.datetime` for all times and `float` for coordinates and magnitudes.
- Keep a single magnitude field plus `magnitude_type`.
- Keep earthquake, SSE and step catalogs separate. Steps never go in the earthquake catalog.
- Run the validators when you build a catalog by hand, and read the warnings rather than silencing them.
- Prefer catalog fetchers over hand-built earthquake lists when you can.

## 10. Best practices

### Do

- Use `datetime.datetime` for all times, and `float` for latitude, longitude and magnitude.
- Include `event_id` for traceability.
- Validate catalogs before use.
- Use UTC consistently, in the catalogs and in the time series.
- Use WGS84 latitude and longitude (state this if your catalog uses another system).
- Keep earthquakes, SSEs and station steps in separate catalogs. Antenna and receiver changes go in `step_catalog`.

### Prefer

- `CatalogFetcher` for earthquakes, since its output is already in the right format.
- One magnitude scale within a catalog (`magnitude_type` records which).
- Complete metadata (`depth`, `event_id`, `source`).
- Catalog times for SSEs that are your best estimate when you use `sse_optimize_timing`, since the search is centred on them.

### Consider

- Filtering events by time range before passing them to the model.
- Spatial filtering for large catalogs (`d_param`, `d_param_eq`, `d_param_sse`).
- Merging duplicate events (`merge_earthquakes_same_day`, `merge_close_sse`).
- Checking steps that fall on earthquake days, since they produce identical columns.

---

## 11. Additional resources

### In this repository

- Catalog fetching: `src/trajmod/events/catalogs/catalog_fetcher.py`
- Catalog validation: `src/trajmod/events/catalogs/validation.py`
- Worked examples: `src/trajmod/example_japan.py` (earthquakes, SSEs, steps, timing optimisation) and `src/trajmod/example_cascadia.py`

### External

- [USGS ComCat](https://earthquake.usgs.gov/data/comcat/): earthquake catalog API
- [ISC Bulletin](http://www.isc.ac.uk/iscbulletin/): International Seismological Centre
- [ObsPy documentation](https://docs.obspy.org/): seismology tools for Python

---

## Need help?

If something here does not cover your problem:

1. Run the catalog validation functions (section 7) and read the messages.
2. Check the [trajmod repository](https://github.com/gcostantino/trajmod).
3. Open an issue on [GitHub](https://github.com/gcostantino/trajmod/issues).
