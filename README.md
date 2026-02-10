# trajmod

**Simple GNSS time series Trajectory Modeling**

A Python library for modeling GNSS position time series with support for:
- Secular trends, acceleration and seasonal components
- Slow Slip Events (SSE) with raised-cosine templates
- Earthquake co-seismic offsets and postseismic relaxation
- **Advanced multi-tier postseismic filtering** (magnitude, amplitude, sign consistency, statistical tests)
- Event-type support (earthquakes vs. antenna changes)

---

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

### From source:
```bash
pip install trajmod
```

### Requirements:
- Python ≥ 3.9
- numpy ≥ 1.20
- scipy ≥ 1.7
- matplotlib ≥ 3.3
- scikit-learn ≥ 1.0
- pyproj ≥ 3.0

---

## Quick Start

```python
import numpy as np
from datetime import datetime, timedelta
from trajmod.model import TrajectoryModel
from trajmod.config import ModelConfig

# Generate synthetic data
t0 = datetime(2020, 1, 1)
times = np.array([t0 + timedelta(days=i) for i in range(365)])
data = np.random.randn(365) * 2.0  # 2mm noise
errors = np.ones(365) * 2.0

# Define earthquake catalog
eq_catalog = [
    {
        'date': datetime(2020, 3, 15),
        'lat': 40.0,
        'lon': -120.0,
        'magnitude': 7.2
    },
    {
        'date': datetime(2020, 8, 1),
        'event_type': 'antenna_change'  # No postseismic for equipment changes
    }
]

# Configure model with multi-tier postseismic filtering
config = ModelConfig(
    d_param=1.0,                              # Spatial filtering
    include_seasonal=True,                     # Annual + semi-annual
    postseismic_mag_threshold=6.5,            # Tier 1: Magnitude filter
    postseismic_min_step_amplitude=2.0,       # Tier 2: Step amplitude filter
    enforce_postseismic_sign_consistency=True, # Tier 3: Sign check
    postseismic_selection_criterion='aic',    # Tier 4: Statistical test
    postseismic_selection_threshold=-2.0,     # ΔAIC < -2 required
    tau_grid=[7, 14, 30, 60, 90, 180, 1800]  # Optimize τ per event
)

# Create and fit model
model = TrajectoryModel(
    t=times,
    y=data,
    sigma_y=errors,
    station_lat=40.0,
    station_lon=-120.0,
    eq_catalog=eq_catalog,
    config=config
)

# Build design matrix (applies multi-tier filtering)
model.build_design_matrix()

# Fit model
results = model.fit(compute_uncertainty=True)

# Inspect results
print(f"RMS: {results.rms:.3f} mm")
print(f"Coefficients: {results.coeffs}")
print(f"Template names: {results.template_names}")

# Get uncertainty estimates
print(f"Standard errors: {results.uncertainty['standard_errors']}")

# Predict at new times
future_times = np.array([t0 + timedelta(days=i) for i in range(365, 730)])
predictions = model.predict(future_times)
```

---

## Configuration Options

### ModelConfig Parameters

```python
ModelConfig(
    # Spatial filtering
    d_param=1.0,                           # Empirical scaling in radius law (None = no filtering)

    # Baseline model
    include_seasonal=True,                  # Annual + semi-annual terms
    acceleration_term=True,                 # Quadratic trend

    # Postseismic filtering (4 tiers)
    fit_postseismic_decay=True,             # Enable postseismic templates
    postseismic_mag_threshold=6.5,          # Tier 1: Min magnitude for postseismic
    postseismic_min_step_amplitude=2.0,     # Tier 2: Min step amplitude (mm)
    enforce_postseismic_sign_consistency=True, # Tier 3: Post must match step sign
    postseismic_selection_criterion='aic',  # Tier 4: 'aic', 'bic', 'ftest', 'always'
    postseismic_selection_threshold=-2.0,   # ΔAIC/ΔBIC threshold (or p-value for F-test)
    fit_best_postseismic_tau=True,          # Optimize τ per event from tau_grid
    tau_grid=[7, 14, 30, 60, 90, 180, 1800], # Candidate τ values (days)

    # Event merging
    merge_earthquakes_same_day=False,       # Merge EQs on same day
    merge_close_sse=False,                  # Merge temporally close SSEs
    gap_merge_threshold=5                   # Days threshold for merging
)
```

### Event-Type Support

```python
eq_catalog = [
    # Regular earthquake → gets step + postseismic (if passes filters)
    {'date': datetime(2020, 3, 1), 'lat': 40.0, 'lon': -120.0, 'magnitude': 7.2},

    # Antenna change → gets step only, NO postseismic
    {'date': datetime(2020, 8, 15), 'event_type': 'antenna_change'},

    # Equipment change → gets step only
    {'date': datetime(2021, 2, 10), 'event_type': 'equipment_change'},
]
```

Supported event types:
- `'earthquake'` (default): Co-seismic step + postseismic (if passes filters)
- `'antenna_change'`: Step only, no postseismic
- `'equipment_change'`: Step only, no postseismic
- `'monument_change'`: Step only, no postseismic

---

## Example Log Output

```
INFO: Initialized model: 1095 time points, 0 SSEs, 3 EQs
INFO: Baseline design matrix: 1095 × 28
INFO: Baseline SSR: 1245.67
INFO:   EQ 1 (antenna_change): event type excludes postseismic → SKIPPED
INFO: Filtered to 2/3 candidate EQs (criterion: aic)
INFO:     EQ 0 (M7.2, step=8.5mm): τ=90d, ΔAIC=-15.3 → KEPT
INFO:     EQ 2 (M6.8, step=4.1mm): τ=60d, ΔAIC=-8.2 → KEPT
INFO: Final design matrix: 1095 × 30 (2/2 postseismic)
INFO: Fit complete: RMS = 1.854 mm
```

---

## Advanced Usage

### Custom fitting strategies

```python
from trajmod.strategies.fitting import LassoFitter, IterativeRefinementFitter

# LASSO with automatic regularization
lasso_fitter = LassoFitter(cv=5, positive=False)
results = model.fit(method=lasso_fitter)

# Iterative outlier removal
robust_fitter = IterativeRefinementFitter(max_iterations=10, threshold=3.0)
results = model.fit(method=robust_fitter)
```

### Event selection/filtering

```python
from trajmod.events.event_selection import KneeEventSelector

# Automatic SSE selection using knee detection
selector = KneeEventSelector(
    criterion='bic',
    method='knee',
    smooth_curve=True,
    min_amplitude=1.0  # Minimum 1mm amplitude
)

selected_indices = model.select_events(
    selector=selector,
    event_type='sse',
    plot=True,
    save_plot='sse_selection.png'
)
```

### Visualization

```python
from trajmod.visualization import plot_trajectory_decomposition

# Plot data with model components
plot_trajectory_decomposition(
    model,
    results,
    save_path='trajectory_fit.png',
    show_residuals=True,
    show_components=True
)
```

---

## API Reference

### Core Classes

- **`TrajectoryModel`**: Main model class
  - `build_design_matrix()`: Construct design matrix with multi-tier filtering
  - `fit()`: Fit model with OLS or custom strategy
  - `predict()`: Predict at new time points
  - `select_events()`: Automatic event selection

- **`ModelConfig`**: Configuration dataclass
  - All model parameters with validation

- **`ModelResults`**: Results container
  - Coefficients, fitted values, residuals, uncertainties

### Template Functions

- **`TemplateFunctions`**: Basis functions
  - `offset()`, `trend()`, `acceleration()`
  - `seasonal_sin()`, `seasonal_cos()`
  - `step()`: Heaviside step for earthquakes
  - `log_decay()`: Logarithmic postseismic decay
  - `raised_cosine()`: SSE template

### Fitting Strategies

- **`OLSFitter`**: Ordinary least squares
- **`LassoFitter`**: LASSO with CV
- **`ElasticNetFitter`**: ElasticNet with CV
- **`IterativeRefinementFitter`**: Robust fitting with outlier removal

---

## Testing

Run tests with:
```bash
pytest tests/
```

Run with coverage:
```bash
pytest --cov=trajmod tests/
```

---

## Citation

If you use this code in your research, please cite:

```bibtex
@software{trajmod2026,
  author = {Costantino, Giuseppe},
  title = {trajmod: Simple GNSS Trajectory Modeling},
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
3. Add tests for new features
4. Run `black` formatter and `flake8` linter
5. Submit a pull request

---

## Acknowledgments

This package implements trajectory modeling techniques commonly used in GNSS geodesy, with particular focus on:
- Multi-tier postseismic filtering to prevent overfitting
- Event-type discrimination for equipment changes
- Automated model selection with information criteria

---

## Contact

Giuseppe Costantino - [giuseppe.costantino@ens.fr]

Project Link: [https://github.com/gcostantino/trajmod](https://github.com/gcostantino/trajmod)
