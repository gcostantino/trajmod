"""Main TrajectoryModel class - orchestrates all components."""
import logging
from typing import List, Dict, Optional, Tuple

import numpy as np

from trajmod.config.modelconfig import ModelConfig
from trajmod.design.matrix import DesignMatrixBuilder
from trajmod.events.event_selection import EventSelector
from trajmod.events.events import EventCatalog
from trajmod.model.results import ModelResults
from trajmod.preprocessing.geodetic_utils import GeodesicCalculator
from trajmod.preprocessing.preprocessing import TimeSeriesPreprocessor
from trajmod.preprocessing.time_utils import DecimalYearConverter
from trajmod.strategies.fitting import FittingStrategy, OLSFitter
from trajmod.templates.templates import TemplateFunctions

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class TrajectoryModel:
    """GNSS trajectory model with SSE and earthquake signals."""

    def __init__(self,
                 t: np.ndarray,
                 y: np.ndarray,
                 sigma_y: np.ndarray,
                 station_lat: float,
                 station_lon: float,
                 sse_catalog: Optional[List[Dict]] = None,
                 eq_catalog: Optional[List[Dict]] = None,
                 config: Optional[ModelConfig] = None,
                 time_converter: Optional[DecimalYearConverter] = None):
        """Initialize trajectory model.

        Args:
            t: Observation times (datetime array)
            y: GNSS displacement values
            sigma_y: Uncertainties
            station_lat: Station latitude
            station_lon: Station longitude
            sse_catalog: List of SSE event dicts
            eq_catalog: List of earthquake event dicts
            config: Model configuration
            time_converter: Optional time converter
        """
        # Configuration
        self.config = config or ModelConfig()
        self.time_converter = time_converter or DecimalYearConverter.from_algorithm()

        # Validate inputs
        preprocessor = TimeSeriesPreprocessor(self.time_converter)
        preprocessor.validate_data(t, y, sigma_y, station_lat, station_lon)

        # Store station info
        self.station_lat = station_lat
        self.station_lon = station_lon
        self.geocalc = GeodesicCalculator()

        # Preprocess data (fill gaps)
        self.t, self.y, self.uncertainties, self.t_decyr = preprocessor.fill_gaps(t, y, sigma_y)
        self.t0 = self.t[0]
        self.t_days = (self.t - self.t0).astype('timedelta64[D]').astype(float)

        # Store catalogs
        self.sse_catalog = sse_catalog or []
        self.eq_catalog = eq_catalog or []

        # Select relevant events
        self.selected_sse = self._select_sse_events()
        self.selected_eq = self._select_eq_events()

        # Placeholders
        self.G: Optional[np.ndarray] = None
        self.template_names: Optional[List[str]] = None
        self.results: Optional[ModelResults] = None

        logger.info(f"Initialized model: {len(self.t)} time points, "
                    f"{len(self.selected_sse)} SSEs, {len(self.selected_eq)} EQs")

    '''def build_design_matrix(self) -> np.ndarray:
        """Build design matrix G."""
        templates = TemplateFunctions(self.t_days, self.t0)
        builder = DesignMatrixBuilder(templates, self.config)

        self.G, self.template_names = builder.build(self.selected_sse, self.selected_eq)
        return self.G'''

    '''def build_design_matrix(self) -> np.ndarray:
        templates = TemplateFunctions(self.t_days, self.t0)
        builder = DesignMatrixBuilder(templates, self.config)
        G, names = builder.build(self.selected_sse, self.selected_eq)

        # Postprocess postseismic τ if requested
        if self.config.fit_best_postseismic_tau:
            G, names = self._select_best_postseismic_per_eq(G, names)

        self.G = G
        self.template_names = names
        return self.G'''

    '''def _should_consider_postseismic(self, eq: Dict) -> bool:
        """Determine if an earthquake should be considered for postseismic decay.

        Logic:
        - If d_param is None: Consider ALL earthquakes (catalog is pre-filtered)
        - If d_param is set: Only consider EQs >= postseismic_mag_threshold
        """
        if self.config.d_param is None:
            return True
        else:
            return eq.get('magnitude', 0) >= self.config.postseismic_mag_threshold'''

    def _should_consider_postseismic(self, eq: Dict) -> bool:
        """Determine if an event should be considered for postseismic decay.

        Logic:
        1. Antenna changes and similar events NEVER get postseismic
        2. If d_param is None: Consider ALL earthquakes (catalog is pre-filtered)
        3. If d_param is set: Only consider EQs >= postseismic_mag_threshold

        Args:
            eq: Event dictionary with optional 'event_type' and 'magnitude'
        """
        # case 1: Check event type - antenna changes never get postseismic
        event_type = eq.get('event_type', 'earthquake')  # Default: earthquake

        if event_type in ['antenna_change', 'equipment_change', 'monument_change']:
            logger.debug(f"  Event {event_type} --> no postseismic")
            return False

        # case 2: Magnitude filtering (only for earthquakes)
        if event_type == 'earthquake':
            if self.config.d_param is None:
                # Catalog pre-filtered spatially
                return True
            else:
                return eq.get('magnitude', 0) >= self.config.postseismic_mag_threshold

        # Unknown event type - default to no postseismic (safe)
        logger.warning(f"  Unknown event_type '{event_type}' → no postseismic")
        return False

    def _get_eq_step_index(self, eq_idx: int, n_baseline_cols: int) -> int:
        """Get the column index for a specific EQ step in the design matrix.

        Args:
            eq_idx: Index of earthquake in selected_eq list
            n_baseline_cols: Total number of baseline columns

        Returns:
            Column index for this EQ's step
        """
        # Baseline structure: offset, trend, [accel], seasonal, SSE, EQ steps
        n_base = 2  # offset + trend
        if self.config.acceleration_term:
            n_base += 1

        if self.config.include_seasonal:
            n_base += 4  # annual_sin, annual_cos, semiannual_sin, semiannual_cos
            if self.config.use_envelope_basis:
                n_base += 2 * len(self.config.envelope_periods)

        n_sse = len(self.selected_sse)

        idx_step = n_base + n_sse + eq_idx
        return idx_step

    def _compute_information_criterion(
            self,
            ssr: float,
            n_obs: int,
            n_params: int,
            criterion: str = 'aic'
    ) -> float:
        """Compute AIC or BIC for a model."""
        if n_obs <= n_params:
            return np.inf

        sigma2 = ssr / n_obs

        if criterion == 'aic':
            return n_obs * np.log(max(1e-300, sigma2)) + 2 * n_params
        elif criterion == 'bic':
            return n_obs * np.log(max(1e-300, sigma2)) + n_params * np.log(n_obs)
        else:
            raise ValueError(f"Unknown criterion: {criterion}")

    def _compute_ftest(
            self,
            ssr_reduced: float,
            ssr_full: float,
            n_obs: int,
            p_reduced: int,
            p_full: int
    ) -> float:
        """Compute F-test p-value for nested models."""
        from scipy.stats import f as f_dist

        if ssr_full >= ssr_reduced or n_obs <= p_full:
            return 1.0

        df1 = p_full - p_reduced
        df2 = n_obs - p_full

        f_stat = ((ssr_reduced - ssr_full) / df1) / (ssr_full / df2)
        p_value = 1.0 - f_dist.cdf(f_stat, df1, df2)

        return p_value

    def _optimize_tau_for_eq(
            self,
            eq_idx: int,
            eq_date,
            G_baseline: np.ndarray,
            existing_post_columns: List[np.ndarray],
            method: Optional[FittingStrategy] = None
    ) -> Tuple[Optional[int], Optional[float], Optional[np.ndarray]]:
        """Find best τ for a single earthquake."""
        from ..templates.templates import TemplateFunctions
        from ..strategies.fitting import OLSFitter

        fitter = method or OLSFitter()
        weights = 1.0 / self.uncertainties
        weights[~np.isfinite(weights)] = 0.0

        templater = TemplateFunctions(self.t_days, self.t0)

        best_tau = None
        best_ssr = np.inf
        best_template = None

        for tau in self.config.tau_grid:
            template = templater.log_decay(eq_date, tau)

            if existing_post_columns:
                G_tmp = np.hstack([
                    G_baseline,
                    np.column_stack(existing_post_columns + [template])
                ])
            else:
                G_tmp = np.hstack([G_baseline, template[:, np.newaxis]])

            try:
                fit_result = fitter.fit(G_tmp, self.y, weights)
                ssr = float(np.nansum(fit_result['residuals'] ** 2))

                if ssr < best_ssr:
                    best_ssr = ssr
                    best_tau = tau
                    best_template = template

            except (np.linalg.LinAlgError, ValueError) as e:
                logger.debug(f"Failed to fit τ={tau} for EQ {eq_idx}: {e}")
                continue

        return best_tau, best_ssr, best_template

    def _test_postseismic_significance(
            self,
            eq_idx: int,
            eq_date,
            G_baseline: np.ndarray,
            existing_post_columns: List[np.ndarray],
            ssr_baseline: float,
            n_obs: int,
            coeff_step_ref: float,
            method: Optional[FittingStrategy] = None
    ) -> Tuple[bool, Optional[int], Optional[np.ndarray], Dict]:
        """Test if adding postseismic significantly improves model.

        Includes sign consistency check: postseismic must have same sign as step.
        """
        from ..strategies.fitting import OLSFitter

        # Find best τ
        best_tau, best_ssr, best_template = self._optimize_tau_for_eq(
            eq_idx, eq_date, G_baseline, existing_post_columns, method
        )

        if best_tau is None:
            logger.warning(f"EQ {eq_idx}: Could not find valid τ")
            return False, None, None, {}

        # Fit with postseismic to get coefficient
        if existing_post_columns:
            G_with = np.hstack([
                G_baseline,
                np.column_stack(existing_post_columns + [best_template])
            ])
        else:
            G_with = np.hstack([G_baseline, best_template[:, np.newaxis]])

        fitter = method or OLSFitter()
        weights = 1.0 / self.uncertainties
        weights[~np.isfinite(weights)] = 0.0

        try:
            fit_result = fitter.fit(G_with, self.y, weights)
            coeffs = fit_result['coeffs']
            coeff_post = coeffs[-1]  # Last column
        except Exception as e:
            logger.warning(f"EQ {eq_idx}: Failed to fit with postseismic: {e}")
            return False, None, None, {}

        # Sign consistency check
        if self.config.enforce_postseismic_sign_consistency:
            if np.sign(coeff_step_ref) != np.sign(coeff_post):
                logger.info(f"    EQ {eq_idx}: opposite sign "
                            f"(step={coeff_step_ref:.2f}mm, post={coeff_post:.2f}mm) → REJECTED")
                return False, None, None, {
                    'reason': 'opposite sign',
                    'coeff_step': coeff_step_ref,
                    'coeff_post': coeff_post
                }

        # Statistical test
        p_baseline = G_baseline.shape[1] + len(existing_post_columns)
        p_with = p_baseline + 1

        criterion = self.config.postseismic_selection_criterion
        threshold = self.config.postseismic_selection_threshold

        metrics = {
            'best_tau': best_tau,
            'ssr_without': ssr_baseline,
            'ssr_with': best_ssr,
            'coeff_post': coeff_post
        }

        if criterion == 'always':
            is_significant = True
            metrics['reason'] = 'always included'

        elif criterion == 'aic':
            aic_without = self._compute_information_criterion(
                ssr_baseline, n_obs, p_baseline, 'aic'
            )
            aic_with = self._compute_information_criterion(
                best_ssr, n_obs, p_with, 'aic'
            )
            delta_aic = aic_with - aic_without

            is_significant = delta_aic < threshold
            metrics['delta_aic'] = delta_aic
            metrics['reason'] = f"ΔAIC={delta_aic:.1f}"

        elif criterion == 'bic':
            bic_without = self._compute_information_criterion(
                ssr_baseline, n_obs, p_baseline, 'bic'
            )
            bic_with = self._compute_information_criterion(
                best_ssr, n_obs, p_with, 'bic'
            )
            delta_bic = bic_with - bic_without

            is_significant = delta_bic < threshold
            metrics['delta_bic'] = delta_bic
            metrics['reason'] = f"ΔBIC={delta_bic:.1f}"

        elif criterion == 'ftest':
            p_value = self._compute_ftest(
                ssr_baseline, best_ssr, n_obs, p_baseline, p_with
            )
            is_significant = p_value < threshold
            metrics['p_value'] = p_value
            metrics['reason'] = f"p={p_value:.4f}"

        else:
            raise ValueError(f"Unknown criterion: {criterion}")

        return is_significant, best_tau, best_template, metrics

    def build_design_matrix(self) -> np.ndarray:
        """Build design matrix G with multi-tier postseismic selection.

        Filtering tiers:
        1. Magnitude threshold (if d_param is set)
        2. Step amplitude threshold (always applied)
        3. Sign consistency (postseismic must match step sign)
        4. Statistical significance (ΔAIC/ΔBIC/F-test)
        """
        from ..templates.templates import TemplateFunctions
        from ..design.matrix import DesignMatrixBuilder
        from ..strategies.fitting import OLSFitter

        if not self.config.fit_postseismic_decay:
            templates = TemplateFunctions(self.t_days, self.t0)
            builder = DesignMatrixBuilder(templates, self.config)
            self.G, self.template_names = builder.build(self.selected_sse, self.selected_eq)
            logger.info(f"Postseismic disabled, final matrix: {self.G.shape}")
            return self.G

        templates = TemplateFunctions(self.t_days, self.t0)
        builder = DesignMatrixBuilder(templates, self.config)

        # Step 1: Build baseline + SSE + EQ steps (no postseismic)
        baseline_cols = []
        baseline_cols.extend(builder._build_base_model())

        if self.config.include_seasonal:
            baseline_cols.extend(builder._build_seasonal())

        baseline_cols.extend(builder._build_sse_templates(self.selected_sse))

        # Add EQ steps only
        for i, eq in enumerate(self.selected_eq):
            step = templates.step(eq['eq_day'])
            baseline_cols.append(step)
            eq_str = eq['eq_day'].strftime('%Y-%m-%d')
            builder.template_names.append(f"EQ_{i}_{eq_str}_step")

        G_baseline = np.column_stack(baseline_cols) if baseline_cols else np.empty((len(self.t_days), 0))
        names_baseline = builder.template_names.copy()

        logger.info(f"Baseline design matrix: {G_baseline.shape[0]} × {G_baseline.shape[1]}")

        # Step 2: Fit baseline to get SSR and step coefficients
        fitter = OLSFitter()
        weights = 1.0 / self.uncertainties
        weights[~np.isfinite(weights)] = 0.0

        valid_mask = ~np.isnan(self.y)
        n_obs = np.sum(valid_mask)

        try:
            result_baseline = fitter.fit(G_baseline, self.y, weights)
            ssr_baseline = float(np.nansum(result_baseline['residuals'] ** 2))
            coeffs_baseline = result_baseline['coeffs']
        except Exception as e:
            logger.error(f"Failed to fit baseline model: {e}")
            self.G = G_baseline
            self.template_names = names_baseline
            return self.G

        logger.info(f"Baseline SSR: {ssr_baseline:.2f}")

        # Step 3: Apply magnitude and step amplitude filters
        candidate_eqs = []
        n_baseline_cols = G_baseline.shape[1]

        for i, eq in enumerate(self.selected_eq):
            eq_mag = eq.get('magnitude', 0)
            event_type = eq.get('event_type', 'earthquake')

            # Tier 1: Magnitude filter
            if not self._should_consider_postseismic(eq):
                # ← REPLACE logger.debug line with these 4 lines:
                if event_type != 'earthquake':
                    logger.info(f"  EQ {i} ({event_type}): event type excludes postseismic → SKIPPED")
                else:
                    logger.debug(f"  EQ {i} (M{eq_mag:.1f}): below magnitude threshold → SKIPPED")
                continue

            # Tier 2: Step amplitude filter
            idx_step = self._get_eq_step_index(i, n_baseline_cols)
            coeff_step = coeffs_baseline[idx_step]

            if abs(coeff_step) < self.config.postseismic_min_step_amplitude:
                logger.info(f"  EQ {i} (M{eq_mag:.1f}): step={coeff_step:.2f}mm "
                            f"< {self.config.postseismic_min_step_amplitude:.1f}mm → SKIPPED")
                continue

            # Passed filters
            candidate_eqs.append((i, eq, coeff_step))

        if not candidate_eqs:
            logger.info("No earthquakes passed magnitude/amplitude filters")
            self.G = G_baseline
            self.template_names = names_baseline
            return self.G

        logger.info(f"Filtered to {len(candidate_eqs)}/{len(self.selected_eq)} candidate EQs "
                    f"(criterion: {self.config.postseismic_selection_criterion})")

        # Step 4: Test each candidate with statistical test + sign check
        post_columns = []
        post_names = []

        for eq_idx, eq, coeff_step in candidate_eqs:
            eq_date = eq['eq_day']
            eq_mag = eq.get('magnitude', 0)

            is_significant, best_tau, best_template, metrics = self._test_postseismic_significance(
                eq_idx=eq_idx,
                eq_date=eq_date,
                G_baseline=G_baseline,
                existing_post_columns=post_columns,
                ssr_baseline=ssr_baseline,
                n_obs=n_obs,
                coeff_step_ref=coeff_step
            )

            if is_significant and best_template is not None:
                post_columns.append(best_template)
                eq_str = eq_date.strftime('%Y-%m-%d')
                post_names.append(f"EQ_{eq_idx}_{eq_str}_log_tau{best_tau}")

                logger.info(f"    EQ {eq_idx} (M{eq_mag:.1f}, step={coeff_step:.1f}mm): "
                            f"τ={best_tau}d, {metrics['reason']} → KEPT")

                # Update baseline SSR for next test
                ssr_baseline = metrics['ssr_with']
            else:
                reason = metrics.get('reason', 'failed')
                logger.info(f"    EQ {eq_idx} (M{eq_mag:.1f}, step={coeff_step:.1f}mm): {reason} → SKIPPED")

        # Step 5: Assemble final design matrix
        if post_columns:
            G_final = np.hstack([G_baseline, np.column_stack(post_columns)])
            names_final = names_baseline + post_names
        else:
            G_final = G_baseline
            names_final = names_baseline
            logger.info("No postseismic events passed all filters")

        self.G = G_final
        self.template_names = names_final

        logger.info(f"Final design matrix: {self.G.shape[0]} × {self.G.shape[1]} "
                    f"({len(post_columns)}/{len(candidate_eqs)} postseismic)")

        return self.G

    def _select_sse_events(self) -> List[Dict]:
        selected = []
        for ev in self.sse_catalog:
            evc = dict(ev)
            evc['start_day'] = ev['start']
            evc['end_day'] = ev['end']

            if self.config.d_param is None:
                # Include all events
                selected.append(evc)
            else:
                # Apply distance filtering
                dist = self.geocalc.distance_km(
                    self.station_lat, self.station_lon,
                    ev['lat'], ev['lon']
                )
                rad = self.geocalc.radius_from_magnitude(
                    ev['magnitude'], self.config.d_param
                )
                if dist <= rad:
                    selected.append(evc)

        return selected

    def _select_eq_events(self) -> List[Dict]:
        """Select earthquake events within radius of influence."""
        selected = []
        for ev in self.eq_catalog:
            evc = dict(ev)
            evc['eq_day'] = ev.get('date') or ev.get('time')

            # skip events without date/time
            if evc['eq_day'] is None:
                logger.warning(f"Skipping earthquake without date/time: {ev}")
                continue

            if self.config.d_param is not None:
                dist = self.geocalc.distance_km(
                    self.station_lat, self.station_lon, ev['lat'], ev['lon']
                )
                rad = self.geocalc.radius_from_magnitude(ev['magnitude'], self.config.d_param)
                if dist <= rad:
                    selected.append(evc)
            else:
                selected.append(evc)

        if self.config.merge_earthquakes_same_day:
            selected = EventCatalog.merge_earthquakes_by_date(selected)

        return selected

    def _select_best_postseismic_per_eq(
            self,
            G: np.ndarray,
            template_names: List[str],
            method: Optional[FittingStrategy] = None
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Postprocess design matrix to keep only the best-τ postseismic
        template per earthquake (based on residual norm).

        Uses the same fitting strategy and weighting as the main fit() method.

        Args:
            G: full design matrix (baseline + SSE + EQ step + all postseismic)
            template_names: list of column names, same length as G.shape[1]
            method: Fitting strategy (default: OLS)

        Returns:
            (G_reduced, names_reduced)
        """
        # Initialize fitter and weights (same as in fit())
        fitter = method or OLSFitter()

        weights = 1.0 / self.uncertainties
        weights[~np.isfinite(weights)] = 0.0

        y = self.y
        G_full = G
        names_full = template_names

        # Identify baseline columns
        baseline_idx = []
        sse_idx = []
        eq_step_idx = []
        post_idx_by_eq = {}

        for j, name in enumerate(names_full):
            if name.startswith("SSE_"):
                sse_idx.append(j)
            elif "_step" in name and name.startswith("EQ_"):
                eq_id = name.split("_")[1]  # 'EQ_i_YYYY-MM-DD_step' → i
                eq_step_idx.append(j)
                post_idx_by_eq.setdefault(eq_id, [])
            elif "_log_tau" in name and name.startswith("EQ_"):
                parts = name.split("_")  # EQ, i, YYYY-MM-DD, 'log', 'tauX'
                eq_id = parts[1]
                post_idx_by_eq.setdefault(eq_id, []).append(j)
            else:
                baseline_idx.append(j)

        # If no postseismic columns, nothing to do
        if not any(post_idx_by_eq.values()):
            logger.info("No postseismic columns found, skipping τ selection")
            return G_full, names_full

        # For each EQ, pick best τ (if any)
        keep_post_indices = []

        for eq_id, idx_list in post_idx_by_eq.items():
            if not idx_list:
                continue

            best_idx = None
            best_ssr = np.inf

            # Columns that are always present: baseline + all SSE + all EQ steps
            # (all postseismic candidates are tested one at a time)
            for j_post in idx_list:
                cols_tmp = (
                        baseline_idx
                        + sse_idx
                        + eq_step_idx
                        + [j_post]
                )
                G_tmp = G_full[:, cols_tmp]

                try:
                    # Use the fitting strategy (same as main fit)
                    fit_result = fitter.fit(G_tmp, y, weights)
                    residuals = fit_result["residuals"]

                    # Compute SSR
                    ssr = float(np.nansum(residuals ** 2))

                    if ssr < best_ssr:
                        best_ssr = ssr
                        best_idx = j_post

                except (np.linalg.LinAlgError, ValueError) as e:
                    logger.warning(f"Failed to fit postseismic τ for EQ {eq_id}, "
                                   f"column {j_post}: {e}")
                    continue

            if best_idx is not None:
                keep_post_indices.append(best_idx)
                # Extract τ value from name for logging
                tau_name = names_full[best_idx]
                logger.info(f"EQ {eq_id}: selected {tau_name}")

        # Build final column index list:
        # baseline + SSE + EQ step + chosen postseismic (one per EQ)
        keep_indices = sorted(
            set(baseline_idx) | set(sse_idx) | set(eq_step_idx) | set(keep_post_indices)
        )

        G_reduced = G_full[:, keep_indices]
        names_reduced = [names_full[j] for j in keep_indices]

        n_original_post = sum(len(v) for v in post_idx_by_eq.values())
        n_selected_post = len(keep_post_indices)
        logger.info(f"Postseismic τ selection: {n_original_post} candidates → "
                    f"{n_selected_post} selected (1 per EQ)")

        return G_reduced, names_reduced

    '''def _select_postseismic_events(
            self,
            G_baseline: np.ndarray,
            post_templates: List[Dict],
            selector: 'EventSelector'
    ) -> List[int]:
        """Apply event selector to decide which EQs need postseismic decay.

        This reuses the existing EventSelector interface, treating postseismic
        candidates as events (same as SSE selection).

        Args:
            G_baseline: Design matrix with baseline + SSE + all EQ steps (no postseismic)
            post_templates: List of reference postseismic templates
            selector: EventSelector instance (e.g., AmplitudeThresholdSelector)

        Returns:
            List of selected indices into post_templates
        """
        if len(post_templates) == 0:
            logger.info("No postseismic candidates to select from")
            return []

        valid_mask = ~np.isnan(self.y)

        # Apply selector (same API as SSE/EQ selection!)
        selected_indices = selector.select(
            y=self.y,
            G_baseline=G_baseline,
            event_templates=post_templates,
            valid_mask=valid_mask
        )

        logger.info(f"Postseismic selection: {len(selected_indices)}/{len(post_templates)} "
                    f"earthquakes need postseismic decay")

        return selected_indices

    def _optimize_postseismic_tau(
            self,
            G_base: np.ndarray,
            selected_indices: List[int],
            post_metadata: List[Dict],
            method: Optional[FittingStrategy] = None
    ) -> Tuple[List[np.ndarray], List[str]]:
        """Find best τ for each selected postseismic event.

        Args:
            G_base: Design matrix with baseline + SSE + all EQ steps
            selected_indices: Indices of EQs that need postseismic
            post_metadata: Metadata for all postseismic candidates
            method: Fitting strategy (default: OLS)

        Returns:
            (post_columns, post_names): Lists of templates and names for selected postseismics
        """
        from ..templates.templates import TemplateFunctions
        from ..strategies.fitting import OLSFitter

        if len(selected_indices) == 0:
            return [], []

        # Initialize fitter and weights (same as main fit)
        fitter = method or OLSFitter()
        weights = 1.0 / self.uncertainties
        weights[~np.isfinite(weights)] = 0.0

        templater = TemplateFunctions(self.t_days, self.t0)

        post_columns = []
        post_names = []

        logger.info(f"Optimizing τ for {len(selected_indices)} postseismic events...")

        for idx in selected_indices:
            meta = post_metadata[idx]
            eq_date = meta['eq_date']
            eq_idx = meta['eq_index']

            # Test all τ values
            best_tau = None
            best_ssr = np.inf
            best_template = None

            for tau in self.config.tau_grid:
                template = templater.log_decay(eq_date, tau)

                # Build candidate G with all existing columns + this postseismic
                G_tmp = np.hstack([
                    G_base,
                    np.column_stack(post_columns + [template]) if post_columns else template[:, np.newaxis]
                ])

                try:
                    fit_result = fitter.fit(G_tmp, self.y, weights)
                    ssr = float(np.nansum(fit_result['residuals'] ** 2))

                    if ssr < best_ssr:
                        best_ssr = ssr
                        best_tau = tau
                        best_template = template

                except (np.linalg.LinAlgError, ValueError) as e:
                    logger.debug(f"Failed to fit τ={tau} for EQ {eq_idx}: {e}")
                    continue

            if best_template is not None:
                post_columns.append(best_template)
                eq_str = eq_date.strftime('%Y-%m-%d')
                post_names.append(f"EQ_{eq_idx}_{eq_str}_log_tau{best_tau}")
                logger.info(f"  EQ {eq_idx}: τ = {best_tau} days")
            else:
                logger.warning(f"Could not find valid τ for EQ {eq_idx}, skipping")

        logger.info(f"Optimized τ for {len(post_columns)}/{len(selected_indices)} events")

        return post_columns, post_names'''

    def select_events(self, selector: EventSelector,
                      event_type: str = 'sse', plot: bool = False,
                      save_plot: Optional[str] = None, update_catalog: bool = True) -> List[int]:
        """Select events using a given strategy.

        Args:
            selector: EventSelector instance
            event_type: 'sse' or 'eq'

        Returns:
            List of selected event indices
        """
        from ..templates.templates import TemplateFunctions
        from ..design.matrix import DesignMatrixBuilder

        templates = TemplateFunctions(self.t_days, self.t0)
        builder = DesignMatrixBuilder(templates, self.config)

        # Build baseline design matrix
        baseline_cols = []
        baseline_cols.extend(builder._build_base_model())

        if self.config.include_seasonal:
            baseline_cols.extend(builder._build_seasonal())

        G_baseline = np.column_stack(baseline_cols) if baseline_cols else np.empty((len(self.t_days), 0))

        # Prepare event templates
        event_catalog = self.selected_sse if event_type == 'sse' else self.selected_eq
        event_templates_list = []

        for i, ev in enumerate(event_catalog):
            if event_type == 'sse':
                template = templates.raised_cosine(ev['start_day'], ev['end_day'])
                name = f"SSE_{i}"
            else:
                template = templates.step(ev['eq_day'])
                name = f"EQ_{i}"

            event_templates_list.append({'template': template, 'name': name})

        # Select events
        valid_mask = ~np.isnan(self.y)
        selected_indices = selector.select(self.y, G_baseline, event_templates_list, valid_mask)

        logger.info(f"Selected {len(selected_indices)}/{len(event_catalog)} {event_type} events")

        # optionally update catalog
        if update_catalog:
            if event_type == 'sse':
                original_count = len(self.selected_sse)
                self.selected_sse = [self.selected_sse[i] for i in selected_indices]
                logger.info(f"Updated SSE catalog: {original_count} → {len(self.selected_sse)} events")
            else:
                original_count = len(self.selected_eq)
                self.selected_eq = [self.selected_eq[i] for i in selected_indices]
                logger.info(f"Updated EQ catalog: {original_count} → {len(self.selected_eq)} events")

            # Force design matrix rebuild
            self.G = None

        if plot and hasattr(selector, 'plot_selection'):
            selector.plot_selection(save_path=save_plot)

        return selected_indices

    def fit_pruning(self, event_type: str = 'sse',
                    criterion: str = 'bic', method: str = 'knee',
                    weights: Optional[np.ndarray] = None, plot: bool = False,
                    save_plot: Optional[str] = None) -> ModelResults:
        """Fit model with automatic event pruning/selection.

        Args:
            event_type: 'sse' or 'eq' - which events to prune
            criterion: 'aic', 'bic', or 'ssr'
            method: 'knee' or 'min'
            weights: Optional observation weights

        Returns:
            ModelResults with selected events
        """
        from ..events.event_selection import KneeEventSelector

        # Select events
        selector = KneeEventSelector(criterion=criterion, method=method)
        selected_indices = self.select_events(selector, event_type=event_type)

        # Update selected events
        if event_type == 'sse':
            original_sse = self.selected_sse.copy()
            self.selected_sse = [original_sse[i] for i in selected_indices]
            logger.info(f"Pruned SSE catalog: {len(original_sse)} → {len(self.selected_sse)}")
        else:
            original_eq = self.selected_eq.copy()
            self.selected_eq = [original_eq[i] for i in selected_indices]
            logger.info(f"Pruned EQ catalog: {len(original_eq)} → {len(self.selected_eq)}")

        # Fit with pruned catalog
        self.G = None  # Force rebuild
        results = self.fit(weights=weights)

        # Store selection info
        results.metadata['event_selection'] = {
            'criterion': criterion,
            'method': method,
            'n_selected': len(selected_indices),
            'selected_indices': selected_indices
        }

        return results

    def fit(self, method: Optional[FittingStrategy] = None,
            weights: Optional[np.ndarray] = None,
            compute_uncertainty: bool = True) -> ModelResults:
        """Fit the trajectory model.

        Args:
            method: Fitting strategy (default: OLS)
            weights: Optional observation weights
            compute_uncertainty: Whether to compute uncertainty estimates

        Returns:
            ModelResults object
        """
        if self.G is None:
            self.build_design_matrix()

        fitter = method or OLSFitter()

        if weights is None:
            weights = 1.0 / self.uncertainties
            weights[~np.isfinite(weights)] = 0.0

        fit_result = fitter.fit(self.G, self.y, weights)

        # Compute uncertainty if requested
        uncertainty_results = None
        if compute_uncertainty:
            from .uncertainty_quantification import UncertaintyQuantifier
            valid_mask = ~np.isnan(self.y)
            uncertainty_results = UncertaintyQuantifier.quantify(
                G=self.G,
                residuals=fit_result['residuals'],
                valid_mask=valid_mask,
                coeffs=fit_result['coeffs'],
                use_hc=False,
                compute_vif=True
            )

        metadata = {k: v for k, v in fit_result.items()
                    if k not in ['coeffs', 'fitted', 'residuals']}
        metadata['method'] = fitter.__class__.__name__
        metadata['uncertainty'] = uncertainty_results

        self.results = ModelResults.from_fit(
            coeffs=fit_result['coeffs'],
            fitted=fit_result['fitted'],
            residuals=fit_result['residuals'],
            template_names=self.template_names,
            weights=weights,
            **metadata
        )

        logger.info(f"Fit complete: RMS = {self.results.rms:.3f} mm")

        return self.results

    def predict(self, t_pred: np.ndarray) -> np.ndarray:
        """Predict at new time points.

        Args:
            t_pred: Prediction times (datetime array)

        Returns:
            Predicted values

        Raises:
            ValueError: If model not fitted or times invalid
        """
        if self.results is None:
            raise ValueError("Model must be fit before prediction")

        # Validate prediction times
        t_pred = np.asarray(t_pred, dtype='datetime64')
        if len(t_pred) == 0:
            raise ValueError("t_pred cannot be empty")

        # Warn if extrapolating far beyond data range
        t_min, t_max = self.t.min(), self.t.max()
        if t_pred.min() < t_min or t_pred.max() > t_max:
            import warnings
            warnings.warn(
                f"Prediction times [{t_pred.min()}, {t_pred.max()}] "
                f"outside training range [{t_min}, {t_max}]. "
                f"Extrapolation may be unreliable.",
                UserWarning
            )

        # Build design matrix for prediction times
        t_days_pred = (t_pred - self.t0).astype('timedelta64[D]').astype(float)
        templates_pred = TemplateFunctions(t_days_pred, self.t0)
        builder = DesignMatrixBuilder(templates_pred, self.config)
        G_pred, _ = builder.build(self.selected_sse, self.selected_eq)

        return G_pred @ self.results.coeffs
