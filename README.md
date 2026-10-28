<p align="center">
  <img src="https://raw.githubusercontent.com/gcostantino/trajmod/refs/heads/main/assets/logo.png" alt="trajmod logo" width="400"/>
</p>

---
**The simple non-linear GNSS time series trajectory model**

[![PyPI version](https://img.shields.io/pypi/v/trajmod)](https://pypi.org/project/trajmod/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/trajmod)](https://pypi.org/project/trajmod/)
[![Downloads](https://img.shields.io/pypi/dm/trajmod?style=round-square)](https://pepy.tech/project/trajmod)


A Python library for modeling GNSS position time series with support for:
- Secular trends, acceleration and seasonal components
- Slow Slip Events (SSE) with raised-cosine templates, with optional nonlinear optimisation of their timing
- Earthquake co-seismic offsets with log, exponential, or combined log+exp post-seismic terms
- Station-specific non-earthquake steps (antenna, receiver, monument changes) from a separate step catalogue

### Postseismic selection

An earthquake gets a postseismic term only if it passes these (customizable) checks:

1. Magnitude at least `postseismic_mag_threshold` (default 6.5; 0 disables the gate).
2. Co-seismic step at least `postseismic_min_step_amplitude` (default 5 mm).
3. The postseismic amplitude has the same sign as the step.
4. The information criterion (`postseismic_selection_criterion`) prefers the model with the term.

The shape is chosen per earthquake among the allowed candidates:

- `log`: afterslip-like, with tau from `tau_grid`.
- `exp`: viscoelastic-like, with tau from `tau_grid_exp`. Allowed from magnitude `postseismic_exp_min_mag` (default 7.5).
- `log+exp` pair: both taus searched jointly, with `tau_exp >= postseismic_exp_min_ratio * tau_log`. Needs `postseismic_allow_combined=True`, both kinds in `postseismic_models`, and magnitude at least `postseismic_pair_min_mag`.

By default, only `log` is used (`postseismic_models=['log']`). The accepted terms are stored in `model.post_specs` and shared by `predict()`.


### Nonlinear SSE timing

With `sse_optimize_timing=True` (or `model.fit(optimize_sse_timing=True)`), the start and end of each SSE are adjusted by whole days within a tolerance around the catalogue times:

- For each candidate (start, end), the linear coefficients are profiled out exactly: the chi-square is computed on data and template projected onto the complement of all other columns, so no refit is needed per candidate.
- Tolerance per event is `min(sse_timing_tol_max_days, max(sse_timing_tol_days, sse_timing_tol_frac * duration))`.
- `sse_timing_prior=True` adds a Gaussian prior pulling start and end toward the catalogue.
- A changed timing is accepted only if it improves the information criterion (`sse_timing_criterion`, 'bic' or 'aic').
- The search is coordinate descent over all SSEs (`sse_timing_max_rounds` passes). Each round rebuilds the design matrix, so postseismic selection is rerun with the current SSE times. It stops when neither SSE times nor postseismic decisions change.
- Each search starts from the catalogue times, so the tolerance is centred on the catalogue (or on the master's times when a `nonlinear_prior` is used, see below).

The outcome is in `results.metadata['sse_timing']`. Call `model.reset_sse_timing()` to go back to the catalogue times.

### Sharing nonlinear parameters across components

Postseismic decay times and SSE start and end times are physical properties of the source, so they are the same on every GNSS component, while amplitudes differ. The user can fit one component (the **master**, chosen from your prior knowledge of the signal), export its nonlinear parameters, and use them on the other components:

- **Master.** A full fit (postseismic selection, SSE timing) on the component where the signal is clearest.
- **`export_nonlinear()`.** Returns the postseismic kinds and taus (per earthquake) and the fitted SSE times as a plain, JSON-serialisable dict.
- **Followers.** Pass the dict as `nonlinear_prior=` to the model of another component.

Two modes, set with `nonlinear_prior_mode`:

| Mode | Postseismic | SSE times |
|---|---|---|
| `fixed` (default) | The master's kind and tau for each matched earthquake | The master's start and end |
| `window` | Same kinds; tau within `nonlinear_prior_tau_window` grid steps | Within `nonlinear_prior_sse_window_days` of the master's times |

Events are matched by date, so the follower can have a different event set. Events the master doesn't contain are searched normally, and each follower still decides per component whether a postseismic term is significant.

---

### Examples

East-west component of two stations, fitted with the default configuration and the nonlinear SSE timing.

<p align="center">
  <img src="https://raw.githubusercontent.com/gcostantino/trajmod/refs/heads/main/assets/fit_ALBH.png" alt="Trajectory fit of the E-W component at station ALBH, Cascadia" width="90%">
</p>

*Example of modeled GNSS position time series for the station ALBH (Cascadia), E-W component.*

<p align="center">
  <img src="https://raw.githubusercontent.com/gcostantino/trajmod/refs/heads/main/assets/fit_J550.png" alt="Trajectory fit of the E-W component at station J550, Japan" width="90%">
</p>

*Example of modeled GNSS position time series for the station J550 (Japan), E-W component.*


## Features

### Core Capabilities
- **Flexible design matrix construction** with polynomial trends, seasonal terms, and transient events
- **Automated postseismic selection** using magnitude thresholds, step amplitude filters, sign consistency checks, and AIC/BIC/F-test criteria
- **Event-type discrimination**: Distinguish between earthquakes (with postseismic) and equipment changes (step only)
- **Uncertainty quantification** with covariance matrices, confidence intervals, and VIF-based multicollinearity detection
- **Multiple fitting strategies**: OLS, LASSO, ElasticNet, iterative outlier removal

### Advanced Postseismic Filtering

trajmod implements a **4-tier filtering system** to prevent overfitting:

1. **Magnitude threshold**: Only earthquakes ≥ M6.5 (configurable) are considered
2. **Step amplitude threshold**: Only events with |step| ≥ 2mm (configurable) get postseismic
3. **Sign consistency**: Postseismic decay must have the same sign as co-seismic step
4. **Statistical significance**: ΔAIC < -2 (or BIC/F-test) required to include postseismic

This ensures physically meaningful postseismic signals are retained while suppressing noise fitting.

---

## Installation

From PyPI:

```bash
pip install trajmod
```

From source:

```bash
git clone [https://github.com/gcostantino/trajmod](https://github.com/gcostantino/trajmod)
cd trajmod
pip install -e .
```

Requirements: Python 3.9 or later, numpy, scipy, matplotlib, scikit-learn, pyproj, requests.


## Quick Start

```python
import numpy as np
from datetime import datetime, timedelta
from trajmod.model import TrajectoryModel
from trajmod.config import ModelConfig
from trajmod.events import CatalogFetcher

# Synthetic data
t0 = datetime(2020, 1, 1)
times = np.array([t0 + timedelta(days=i) for i in range(365)])
data = np.random.randn(365) * 2.0
errors = np.ones(365) * 2.0

# Earthquakes ...
eq_catalog = [{'date': datetime(2020, 3, 15), 'lat': 40.0, 'lon': -120.0, 'magnitude': 7.2}]

# ... or fetch them from USGS
fetcher = CatalogFetcher()
eq_catalog = fetcher.fetch_usgs(
    lat=40.0, lon=-120.0, radius_km=500,
    start_date="2020-01-01", end_date="2020-12-31",
    min_magnitude=5.0,
)

# Station steps (antenna, receiver changes): separate from earthquakes
step_catalog = [{'date': datetime(2020, 8, 1), 'event_type': 'antenna_change'}]

config = ModelConfig(
    d_param=1.0,
    include_seasonal=True,
    postseismic_models=['log', 'exp'],
    postseismic_allow_combined=True,
    postseismic_mag_threshold=6.5,
    postseismic_min_step_amplitude=2.0,
    postseismic_selection_criterion='bic',
    postseismic_selection_threshold=-2.0,
)

model = TrajectoryModel(
    t=times, y=data, sigma_y=errors,
    station_lat=40.0, station_lon=-120.0,
    eq_catalog=eq_catalog,
    step_catalog=step_catalog,
    config=config,
)

results = model.fit(compute_uncertainty=True)

print(results.summary())
print(results.template_names)
print(results.metadata['uncertainty'].standard_errors)

future_times = np.array([t0 + timedelta(days=i) for i in range(365, 730)])
predictions = model.predict(future_times)
```

---

## Catalogs

### Earthquakes

```python
from trajmod.events import CatalogFetcher

fetcher = CatalogFetcher()
eq_catalog = fetcher.fetch_usgs(
    lat=40.0, lon=-120.0, radius_km=500,
    start_date="2020-01-01", end_date="2023-12-31",
    min_magnitude=5.0,
)
```

Your own catalogue is a list of dicts with a datetime (not a string) in `'date'` or `'time'`, plus `lat`, `lon` and `magnitude`. The other fetchers are `fetch_isc`, `fetch_nied` and `fetch_jma`, and each has a `_box` variant for a rectangular region.

### Station steps

Antenna, receiver and monument changes go in `step_catalog`, usually one list per station:

```python
step_catalog = [
    {'date': datetime(2013, 5, 2), 'event_type': 'antenna_change'},
    {'date': datetime(2016, 8, 14), 'event_type': 'receiver_change'},
]
model = TrajectoryModel(..., eq_catalog=eq_catalog, step_catalog=step_catalog)
```

Each step gets one column named `STEP_{i}_{date}_{label}`. Steps are not filtered by distance, never get postseismic terms, and are not pruned unless requested with `select_events(selector, event_type='step')`. Two steps on the same day (or a step on the day of an earthquake) produce identical columns. Antenna-type rows found in `eq_catalog` are moved to the step catalogue with a warning.

Catalogs are validated on model creation. See the [Catalog Format Guidelines](CATALOG_FORMAT_GUIDELINES.md) for the full specification.

---

## Configuration

```python
ModelConfig(
    # Spatial filtering: R = 10^((0.43*M - 0.7)/d); None = no filter
    d_param=1.0,
    d_param_sse=None,        # SSE override (omit to inherit d_param)
    d_param_eq=0.8,          # earthquake override (omit to inherit d_param)

    # Baseline
    include_seasonal=True,
    acceleration_term=True,

    # Postseismic
    fit_postseismic_decay=True,
    postseismic_models=['log', 'exp'],
    postseismic_allow_combined=True,
    postseismic_mag_threshold=6.5,
    postseismic_exp_min_mag=7.5,
    postseismic_pair_min_mag=7.5,
    postseismic_exp_min_ratio=2.0,
    tau_grid=[7, 14, 30, 60, 90, 180, 1800],
    tau_grid_exp=[180, 365, 730, 1460, 2920],
    postseismic_min_step_amplitude=2.0,
    enforce_postseismic_sign_consistency=True,
    postseismic_selection_criterion='bic',
    postseismic_selection_threshold=-2.0,
    postseismic_backfit_passes=2,

    # SSE timing optimisation
    sse_optimize_timing=True,
    sse_timing_tol_days=5.0,
    sse_timing_tol_frac=0.25,
    sse_timing_tol_max_days=30.0,
    sse_min_duration_days=1,
    sse_timing_step_days=1,
    sse_timing_prior=True,
    sse_timing_criterion='bic',
    sse_timing_max_rounds=2,
    sse_timing_allow_overlap=False,

    # Event merging
    merge_earthquakes_same_day=False,
    merge_close_sse=False,
    max_overlap_merge_sse=5,
)
```


### Non-earthquake steps

Antenna, receiver and monument changes go in a separate `step_catalog`, usually one file per station. They get one step column each, are never filtered by distance, never get postseismic terms, and are not pruned unless you ask (`select_events(selector, event_type='step')`).

```
step_catalog = [
    {'date': datetime(2013, 5, 2), 'event_type': 'antenna_change'},
    {'date': datetime(2016, 8, 14), 'event_type': 'receiver_change'},
]

model = TrajectoryModel(..., eq_catalog=eq_catalog, step_catalog=step_catalog)
```

Columns are named `STEP_{i}_{date}_{label}`. Two steps on the same day, or a step on the day of an earthquake, give identical columns, and a warning is logged.
Antenna-type rows still found in `eq_catalog` are moved to the step catalogue automatically, with a warning. The step catalogue is validated by `validate_step_catalog`.


Supported event types:
- `'earthquake'` (default): Co-seismic step + postseismic (if passes filters)
- `'antenna_change'`: Step only, no postseismic
- `'equipment_change'`: Step only, no postseismic
- `'monument_change'`: Step only, no postseismic

---

## Example Log Output

```
INFO: Initialized model: 1095 time points, 0 SSEs, 3 EQs, 1 steps
INFO:   EQ 0 (M7.8, step=8.5mm): log tau=90d + exp tau=730d, ΔBIC=-15.3, Δχ²=-42.1 -> KEPT
INFO:   EQ 2 (M6.8, step=4.1mm): log tau=60d, ΔBIC=-8.2, Δχ²=-19.0 -> KEPT
...
```

---

## Advanced Usage

### Multi-component workflow

```python
from trajmod.model import TrajectoryModel
from trajmod.config import ModelConfig
from trajmod.model.nonlinear_prior import save_prior, load_prior

cfg = ModelConfig(sse_optimize_timing=True, postseismic_models=['log', 'exp'])

# 1. Fit the master component (here the vertical, chosen from prior knowledge)
master = TrajectoryModel(t=t, y=y_up, sigma_y=sigma_up,
                         station_lat=lat, station_lon=lon,
                         sse_catalog=sse_catalog, eq_catalog=eq_catalog, config=cfg)
master.fit()

# 2. Export its nonlinear parameters (optionally save them)
prior = master.export_nonlinear()
save_prior(prior, 'station_prior.json')        # load_prior('station_prior.json') to reuse

# 3. Fit the other components with the master's parameters
cfg_follow = ModelConfig(sse_optimize_timing=True, postseismic_models=['log', 'exp'],
                         nonlinear_prior_mode='fixed')
for name, (y, sigma) in {'E': (y_e, sigma_e), 'N': (y_n, sigma_n)}.items():
    follower = TrajectoryModel(t=t, y=y, sigma_y=sigma,
                               station_lat=lat, station_lon=lon,
                               sse_catalog=sse_catalog, eq_catalog=eq_catalog,
                               config=cfg_follow, nonlinear_prior=prior)
    follower.fit()
```

Notes:

- Use the same catalogues and config for the master and the followers, apart from the prior options.
- `fixed` mode estimates amplitudes only, so uncertainties on the followers do not include the uncertainty of the shared taus and times. They are conditional on the master's values.
- Compare the followers with independent fits if you suspect the components differ (for example a vertical component with a different time constant). Large differences mean the shared assumption is doubtful there.
- The prior is a plain dict. `export_nonlinear()` works on any fitted model, so a follower can serve as the master for another component.


### Fitting strategies

```python
from trajmod.strategies.fitting import LassoFitter, IterativeRefinementFitter

results = model.fit(method=LassoFitter(cv=5, positive=False))
results = model.fit(method=IterativeRefinementFitter(max_iterations=10, threshold=3.0))
```

### Event selection

```python
from trajmod.events.event_selection import KneeEventSelector

selector = KneeEventSelector(criterion='bic', method='knee')
selected = model.select_events(selector, event_type='sse', plot=True, save_plot='sse_selection.png')
```

`event_type` can be `'sse'`, `'eq'` or `'step'`. Other selectors: `LassoEventSelector`, `MarginalScreeningSelector`, `ThresholdEventSelector`, `AmplitudeThresholdSelector`, `CompositeEventSelector`.

### Visualization

```python
from trajmod.visualization import TrajectoryVisualizer

TrajectoryVisualizer.plot_fit(model)
TrajectoryVisualizer.plot_components(model)
```

Also available: `plot_fit_windowed`, `plot_coefficients`, `plot_event_selection`, `plot_correlation_matrix`, `plot_acf`, `plot_periodogram`, `plot_residuals_vs_fitted`.

---

## API Reference

### Core Classes

- **`TrajectoryModel`**: Main model class
  - `build_design_matrix()`: Construct design matrix with multi-level filtering (e.g., post-seismic)
  - `fit()`: Fit model with OLS or custom strategy
  - `predict()`: Predict at new time points
  - `select_events()`: Automatic event selection
  - `fit_pruning()`: Fit model with automatic event pruning according to information reduction
  - `reset_sse_timing()`: Optimize SSE templates with non-linear shrinking/expansion of templates
  - `export_nonlinear()`: Nonlinear parameters of a fitted model (postseismic kinds and taus, SSE times) as a JSON-serialisable dict
  - `TrajectoryModel(..., nonlinear_prior=dict)`: Use parameters exported from another component


- **`ModelConfig`**: Configuration dataclass
  - All model parameters with validation

- **`ModelResults`**: Results container
  - Coefficients, fitted values, residuals, uncertainties, template_names, rms, weighted rms, n_observations,
  n_parameters, metadata, get_coefficient(name), summary()

### Template Functions

- **`TemplateFunctions`**: Basis functions
  - `offset()`, `trend()`, `acceleration()`
  - `seasonal_sin()`, `seasonal_cos()`
  - `step()`: Heaviside step for earthquakes
  - `log_decay()`: Logarithmic postseismic decay
  - `exp_decay()`: Exponential (Maxwell-like) postseismic decay
  - `raised_cosine()`: SSE template

### Fitting Strategies

- **`OLSFitter`**: Ordinary least squares
- **`LassoFitter`**: LASSO with CV
- **`ElasticNetFitter`**: ElasticNet with CV
- **`IterativeRefinementFitter`**: Robust fitting with outlier removal


---

## Citation

If you use this code in your research, please cite:

```bibtex
@software{trajmod2026,
  author = {Costantino, Giuseppe},
  title = {trajmod: the simple non-linear GNSS trajectory model},
  year = {2026},
  url = {https://github.com/gcostantino/trajmod}
}
```

---

## License

MIT License - see LICENSE file for details.

---

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Run `black` formatter and `flake8` linter
4. Submit a pull request

---

## Acknowledgments

Parts of this codebase were developed with assistance from Claude (Anthropic AI).

This package implements a separable non-linear trajectory modeling technique, with particular focus on
automated model selection with information criteria and event filtering to prevent overfitting.

This software is inspired by Bevis & Brown, (2014), Blewitt et al. (2016) and Marill et al. (2021).

---

## Contact

Giuseppe Costantino - [giuseppe.costantino@ens.fr]

Project Link: [https://github.com/gcostantino/trajmod](https://github.com/gcostantino/trajmod)
