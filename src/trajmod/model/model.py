"""Main TrajectoryModel class - orchestrates all components."""
import datetime
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from trajmod.config.modelconfig import ModelConfig
from trajmod.design.matrix import DesignMatrixBuilder
from trajmod.design.sse_timing import optimize_sse_timing
from trajmod.events.event_selection import EventSelector, KneeEventSelector
from trajmod.events.events import EventCatalog
from trajmod.model.results import ModelResults
from trajmod.preprocessing.geodetic_utils import GeodesicCalculator
from trajmod.preprocessing.preprocessing import TimeSeriesPreprocessor
from trajmod.preprocessing.time_utils import DecimalYearConverter
from trajmod.strategies.fitting import FittingStrategy, OLSFitter
from trajmod.templates.templates import TemplateFunctions

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

_NON_EQ_TYPES = ('antenna_change', 'equipment_change', 'monument_change')


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
                 time_converter: Optional[DecimalYearConverter] = None,
                 validate_catalogs: bool = True,
                 step_catalog: Optional[List[Dict]] = None):
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
            validate_catalogs: Optional flag to validate sse_catalog and eq_catalog
            step_catalog: Station-specific non-earthquake steps (antenna, equipment, ...);
                never filtered by distance, never given postseismic terms, never pruned unless requested
        """
        self.config = config or ModelConfig()
        self.time_converter = time_converter or DecimalYearConverter.from_algorithm()

        preprocessor = TimeSeriesPreprocessor(self.time_converter)
        preprocessor.validate_data(t, y, sigma_y, station_lat, station_lon)

        self.station_lat = station_lat
        self.station_lon = station_lon
        self.geocalc = GeodesicCalculator()

        self.t, self.y, self.uncertainties, self.t_decyr = preprocessor.fill_gaps(t, y, sigma_y)
        self.t0 = self.t[0]
        self.t_days = (self.t - self.t0).astype('timedelta64[D]').astype(float)

        self.sse_catalog = sse_catalog or []
        self.eq_catalog = eq_catalog or []
        self.step_catalog = step_catalog or []

        if validate_catalogs:
            self._validate_catalogs()

        self.selected_sse = self._select_sse_events()
        self.selected_eq = self._select_eq_events()
        legacy = [e for e in self.eq_catalog
                  if e.get('event_type', 'earthquake') in _NON_EQ_TYPES]
        if legacy:
            logger.warning(f"{len(legacy)} non-earthquake event(s) found in eq_catalog; "
                           f"move them to step_catalog")
        self.selected_steps = self._select_step_events()

        # Accepted postseismic terms: [{'eq_idx', 'eq_day', 'tau', 'coeff_step', 'coeff_post'}]
        self.post_specs: List[Dict] = []

        # SSE timing optimisation state
        self._sse_catalog_selected: Optional[List[Dict]] = None  # SSEs as selected from the catalog
        self._sse_timing_done: bool = False
        self.sse_timing_report: Optional[Dict] = None

        self.G: Optional[np.ndarray] = None
        self.template_names: Optional[List[str]] = None
        self.results: Optional[ModelResults] = None

        logger.info(f"Initialized model: {len(self.t)} time points, "
                    f"{len(self.selected_sse)} SSEs, {len(self.selected_eq)} EQs, {len(self.selected_steps)} steps")

    # ------------------------------------------------------------------ #
    # Catalog validation
    # ------------------------------------------------------------------ #
    def _validate_catalogs(self) -> None:
        """Validate input catalogs and attempt to fix common issues.

        Raises:
            ValueError: If catalogs are invalid and cannot be auto-fixed
        """
        from trajmod.events import (
            validate_earthquake_catalog,
            validate_sse_catalog,
        )

        if self.eq_catalog:
            logger.debug(f"Validating earthquake catalog ({len(self.eq_catalog)} events)...")
            is_valid, messages, fixed = validate_earthquake_catalog(
                self.eq_catalog, strict=False, fix=True)

            if not is_valid:
                if fixed:
                    logger.warning(f"Auto-fixed {len(messages)} issue(s) in earthquake catalog:")
                    for msg in messages[:5]:
                        logger.warning(f"  - {msg}")
                    if len(messages) > 5:
                        logger.warning(f"  ... and {len(messages) - 5} more")
                    self.eq_catalog = fixed
                    logger.info(f"Earthquake catalog validated ({len(fixed)} events)")
                else:
                    error_summary = "\n".join(f"  - {msg}" for msg in messages[:10])
                    raise ValueError(
                        f"Invalid earthquake catalog ({len(messages)} errors):\n"
                        f"{error_summary}\n"
                        f"Please fix these issues or consult the catalog format guidelines.")
            else:
                if messages:
                    logger.info(f"Earthquake catalog valid with {len(messages)} warning(s)")
                    for msg in messages[:3]:
                        logger.debug(f"  {msg}")
                else:
                    logger.debug(f"Earthquake catalog validated ({len(self.eq_catalog)} events)")

        if self.sse_catalog:
            logger.debug(f"Validating SSE catalog ({len(self.sse_catalog)} events)...")
            is_valid, messages, fixed = validate_sse_catalog(
                self.sse_catalog, strict=False, fix=True)

            if not is_valid:
                if fixed:
                    logger.warning(f"Auto-fixed {len(messages)} issue(s) in SSE catalog:")
                    for msg in messages[:5]:
                        logger.warning(f"  - {msg}")
                    if len(messages) > 5:
                        logger.warning(f"  ... and {len(messages) - 5} more")
                    self.sse_catalog = fixed
                    logger.info(f"SSE catalog validated ({len(fixed)} events)")
                else:
                    error_summary = "\n".join(f"  - {msg}" for msg in messages[:10])
                    raise ValueError(
                        f"Invalid SSE catalog ({len(messages)} errors):\n"
                        f"{error_summary}\n"
                        f"Please fix these issues or consult the catalog format guidelines.")
            else:
                if messages:
                    logger.info(f"SSE catalog valid with {len(messages)} warning(s)")
                    for msg in messages[:3]:
                        logger.debug(f"  {msg}")
                else:
                    logger.debug(f"SSE catalog validated ({len(self.sse_catalog)} events)")

    # ------------------------------------------------------------------ #
    # Weighted statistics helpers
    # ------------------------------------------------------------------ #
    def _default_weights(self) -> np.ndarray:
        """Weights w = 1/sigma, zero where not finite."""
        with np.errstate(divide='ignore', invalid='ignore'):
            w = 1.0 / np.asarray(self.uncertainties, dtype=float)
        w = np.array(w, dtype=float)
        w[~np.isfinite(w)] = 0.0
        return w

    def _prepare_weights(self, weights: Optional[np.ndarray]) -> np.ndarray:
        """Return a clean copy of the weights (never mutates the caller's array)."""
        if weights is None:
            return self._default_weights()
        w = np.array(weights, dtype=float)
        w[~np.isfinite(w)] = 0.0
        return w

    def _valid_obs_mask(self, weights: np.ndarray) -> np.ndarray:
        """Observations that contribute to the fit: finite y and positive weight."""
        return np.isfinite(self.y) & np.isfinite(weights) & (weights > 0)

    @staticmethod
    def _chi2(residuals: np.ndarray, weights: np.ndarray) -> float:
        """Weighted sum of squares: sum (w_i * r_i)^2 over contributing observations.

        OLSFitter returns unweighted residuals, so the weights are applied here.
        """
        mask = np.isfinite(residuals) & np.isfinite(weights) & (weights > 0)
        wr = weights[mask] * residuals[mask]
        return float(np.sum(wr ** 2))

    def _fit_chi2(self,
                  G: np.ndarray,
                  weights: np.ndarray,
                  fitter: FittingStrategy) -> Optional[Tuple[float, np.ndarray]]:
        """Fit and return (chi2, coeffs); None if the fit fails."""
        try:
            res = fitter.fit(G, self.y, weights)
        except (np.linalg.LinAlgError, ValueError) as e:
            logger.debug(f"Fit failed: {e}")
            return None
        return self._chi2(res['residuals'], weights), res['coeffs']

    def _n_params(self, n_cols: int, n_post: int) -> int:
        """Effective parameter count used in the information criteria.

        Each postseismic term counts as `postseismic_tau_penalty_params` parameters
        (1 = amplitude only, 2 = amplitude + tau). n_cols is the total number of
        columns, including the n_post postseismic ones.
        """
        pen = int(getattr(self.config, 'postseismic_tau_penalty_params', 1))
        return n_cols - n_post + n_post * pen

    def _compute_information_criterion(self,
                                       chi2: float,
                                       n_obs: int,
                                       n_params: int,
                                       criterion: str = 'aic') -> float:
        """AIC/BIC from a weighted chi-square: n ln(chi2/n) + penalty."""
        if n_obs <= n_params:
            return np.inf

        s2 = chi2 / n_obs
        if criterion == 'aic':
            return n_obs * np.log(max(1e-300, s2)) + 2 * n_params
        elif criterion == 'bic':
            return n_obs * np.log(max(1e-300, s2)) + n_params * np.log(n_obs)
        else:
            raise ValueError(f"Unknown criterion: {criterion}")

    def _compute_ftest(self,
                       chi2_reduced: float,
                       chi2_full: float,
                       n_obs: int,
                       p_reduced: int,
                       p_full: int) -> float:
        """F-test p-value for nested models (weighted chi-square)."""
        from scipy.stats import f as f_dist

        df1 = p_full - p_reduced
        df2 = n_obs - p_full
        if chi2_full >= chi2_reduced or df1 <= 0 or df2 <= 0:
            return 1.0

        f_stat = ((chi2_reduced - chi2_full) / df1) / (chi2_full / df2)
        return float(1.0 - f_dist.cdf(f_stat, df1, df2))

    # ------------------------------------------------------------------ #
    # Postseismic selection
    # ------------------------------------------------------------------ #
    def _should_consider_postseismic(self, eq: Dict) -> bool:
        """Determine if earthquake should be considered for postseismic decay.

        Filtering strategy:
        1. Event type: non-earthquakes never get postseismic.
        2. Magnitude threshold (postseismic_mag_threshold, otherwise always applied).
        3. Step amplitude filter (applied in build_design_matrix).
        4. Statistical test (AIC/BIC/F-test, applied in build_design_matrix).
        """
        event_type = eq.get('event_type', 'earthquake')

        if event_type in _NON_EQ_TYPES:
            logger.debug(f"  Event {event_type} --> no postseismic")
            return False

        if event_type == 'earthquake':
            return eq.get('magnitude', 0) >= self.config.postseismic_mag_threshold

        logger.warning(f"  Unknown event_type '{event_type}' -> no postseismic")
        return False

    # ------------------------------------------------------------------ #
    # Postseismic candidates: log, exp, or log+exp for one earthquake
    # ------------------------------------------------------------------ #
    def _postseismic_kinds(self, mag: float = 0.0) -> List[str]:
        """Single-term kinds allowed for an earthquake of magnitude `mag`."""
        exp_min = float(getattr(self.config, 'postseismic_exp_min_mag', 0.0))
        return [k for k in getattr(self.config, 'postseismic_models', ['log'])
                if k != 'exp' or mag >= exp_min]

    def _combined_allowed(self, mag: float = 0.0) -> bool:
        models = getattr(self.config, 'postseismic_models', ['log'])
        pair_min = float(getattr(self.config, 'postseismic_pair_min_mag', 0.0))
        return bool(getattr(self.config, 'postseismic_allow_combined', False)
                    and 'log' in models and 'exp' in models and mag >= pair_min)

    def _tau_grid_for(self, kind: str) -> List[float]:
        return list(self.config.tau_grid_exp if kind == 'exp' else self.config.tau_grid)

    def _single_candidates(self, kinds: List[str]) -> List[List[Tuple[str, float]]]:
        return [[(k, tau)] for k in kinds for tau in self._tau_grid_for(k)]

    def _pair_candidates(self) -> List[List[Tuple[str, float]]]:
        ratio = float(getattr(self.config, 'postseismic_exp_min_ratio', 2.0))
        return [[('log', tl), ('exp', te)]
                for tl in self._tau_grid_for('log')
                for te in self._tau_grid_for('exp')
                if te >= ratio * tl]

    def _search_terms(self,
                      eq_day,
                      G_baseline: np.ndarray,
                      existing_cols: List[np.ndarray],
                      weights: np.ndarray,
                      builder: DesignMatrixBuilder,
                      fitter: FittingStrategy,
                      candidates: List[List[Tuple[str, float]]],
                      ref_sign: Optional[float] = None) -> Optional[Dict]:
        """Best candidate (lowest weighted chi2); all amplitudes of the new terms
        must have sign ref_sign when it is given. Returns None if none is valid."""
        base = (np.hstack([G_baseline, np.column_stack(existing_cols)])
                if existing_cols else G_baseline)
        best = None
        for terms in candidates:
            cols = [builder.make_postseismic_column(eq_day, tau, kind) for kind, tau in terms]
            out = self._fit_chi2(np.hstack([base, np.column_stack(cols)]), weights, fitter)
            if out is None:
                continue
            chi2, coeffs = out
            cp = [float(c) for c in coeffs[-len(terms):]]
            if ref_sign is not None and ref_sign != 0 \
                    and any(np.sign(c) != ref_sign for c in cp):
                continue
            if best is None or chi2 < best['chi2']:
                best = {'terms': [{'kind': k, 'tau': t} for k, t in terms],
                        'chi2': chi2, 'coeffs': cp}
        return best

    @staticmethod
    def _describe_terms(terms: List[Dict]) -> str:
        return ' + '.join(f"{t['kind']} tau={t['tau']}d" for t in terms)

    def _test_postseismic_significance(self,
                                       eq_idx: int,
                                       eq_day,
                                       G_baseline: np.ndarray,
                                       post_specs: List[Dict],
                                       chi2_baseline: float,
                                       n_obs: int,
                                       coeff_step_ref: float,
                                       weights: np.ndarray,
                                       builder: DesignMatrixBuilder,
                                       fitter: FittingStrategy,
                                       eq_mag: float = 0.0
                                       ) -> Tuple[bool, Optional[Dict], Dict]:
        """Test the postseismic model(s) for one earthquake.

        Options: best single term (log or exp), and, if allowed, the best joint
        log+exp pair over (tau_log, tau_exp). Every amplitude must have the sign
        of the step. The pair must beat the best single option by the criterion,
        and the chosen model must beat 'no postseismic term'.
        Returns (is_significant, chosen, metrics); chosen = {'terms', 'coeffs', 'chi2'}.
        """
        existing_cols = [builder.make_postseismic_column(s['eq_day'], s['tau'], s.get('kind', 'log'))
                         for s in post_specs]
        enforce = self.config.enforce_postseismic_sign_consistency
        step_sign = float(np.sign(coeff_step_ref))

        singles, pairs, rejected = [], [], None

        single = self._search_terms(eq_day, G_baseline, existing_cols, weights, builder, fitter,
                                    self._single_candidates(self._postseismic_kinds(eq_mag)))
        if single is not None:
            if (not enforce) or all(np.sign(c) == step_sign for c in single['coeffs']):
                singles.append(single)
            else:
                rejected = single

        if self._combined_allowed(eq_mag):
            pair = self._search_terms(eq_day, G_baseline, existing_cols, weights, builder, fitter,
                                      self._pair_candidates(),
                                      ref_sign=step_sign if enforce else None)
            if pair is not None:
                pairs.append(pair)

        if not singles and not pairs:
            if rejected is not None:
                logger.info(f"  EQ {eq_idx}: opposite sign "
                            f"(step={coeff_step_ref:.2f}mm, post={rejected['coeffs'][-1]:.2f}mm) "
                            f"-> REJECTED")
                return False, None, {'reason': 'opposite sign', 'coeff_step': coeff_step_ref,
                                     'coeff_post': rejected['coeffs']}
            logger.warning(f"EQ {eq_idx}: could not find a valid postseismic model")
            return False, None, {}

        n_post = len(post_specs)
        n_cols_without = G_baseline.shape[1] + n_post
        p_without = self._n_params(n_cols_without, n_post)
        for o in singles + pairs:
            k = len(o['terms'])
            o['p'] = self._n_params(n_cols_without + k, n_post + k)

        criterion = self.config.postseismic_selection_criterion
        threshold = self.config.postseismic_selection_threshold
        chosen = singles[0] if singles else pairs[0]
        metrics: Dict = {}

        if criterion == 'always':
            is_significant = True
            reason = 'always included'

        elif criterion in ('aic', 'bic'):
            for o in singles + pairs:
                o['ic'] = self._compute_information_criterion(o['chi2'], n_obs, o['p'], criterion)
            if singles and pairs and pairs[0]['ic'] - singles[0]['ic'] < threshold:
                chosen = pairs[0]
            ic_without = self._compute_information_criterion(chi2_baseline, n_obs, p_without, criterion)
            delta_ic = chosen['ic'] - ic_without
            is_significant = delta_ic < threshold
            metrics[f'delta_{criterion}'] = delta_ic
            reason = f"Δ{criterion.upper()}={delta_ic:.1f}, Δχ²={chosen['chi2'] - chi2_baseline:.1f}"

        elif criterion == 'ftest':
            if singles and pairs:
                p_pair = self._compute_ftest(singles[0]['chi2'], pairs[0]['chi2'], n_obs,
                                             singles[0]['p'], pairs[0]['p'])
                if p_pair < threshold:
                    chosen = pairs[0]
            p_value = self._compute_ftest(chi2_baseline, chosen['chi2'], n_obs,
                                          p_without, chosen['p'])
            is_significant = p_value < threshold
            metrics['p_value'] = p_value
            reason = f"p={p_value:.4f}, Δχ²={chosen['chi2'] - chi2_baseline:.1f}"

        else:
            raise ValueError(f"Unknown criterion: {criterion}")

        metrics.update({
            'desc': self._describe_terms(chosen['terms']),
            'chi2_without': chi2_baseline,
            'chi2_with': chosen['chi2'],
            'delta_chi2': chosen['chi2'] - chi2_baseline,
            'coeff_post': chosen['coeffs'],
            'reason': reason,
        })
        return is_significant, chosen, metrics

    def _backfit_tau(self,
                     G_baseline: np.ndarray,
                     post_specs: List[Dict],
                     weights: np.ndarray,
                     builder: DesignMatrixBuilder,
                     fitter: FittingStrategy,
                     n_passes: int) -> None:
        """Re-optimise the tau(s) of each earthquake with the other earthquakes fixed.

        A single term keeps its kind and searches its own grid; a log+exp pair is
        re-optimised jointly. Signs are enforced as in the selection.
        """
        eq_ids = list(dict.fromkeys(s['eq_idx'] for s in post_specs))
        if len(eq_ids) < 2 or n_passes <= 0:
            return

        for p in range(n_passes):
            changed = False
            for eq_id in eq_ids:
                group = [j for j, s in enumerate(post_specs) if s['eq_idx'] == eq_id]
                others = [builder.make_postseismic_column(s['eq_day'], s['tau'], s.get('kind', 'log'))
                          for j, s in enumerate(post_specs) if j not in group]
                first = post_specs[group[0]]
                ref_sign = (float(np.sign(first['coeff_step']))
                            if self.config.enforce_postseismic_sign_consistency else None)

                if len(group) == 1:
                    cands = self._single_candidates([first.get('kind', 'log')])
                else:
                    cands = self._pair_candidates()

                best = self._search_terms(first['eq_day'], G_baseline, others, weights,
                                          builder, fitter, cands, ref_sign)
                if best is None:
                    continue

                old = [(post_specs[j].get('kind', 'log'), post_specs[j]['tau']) for j in group]
                new = [(t['kind'], t['tau']) for t in best['terms']]
                if new != old:
                    logger.info(f"  Backfit pass {p + 1}: EQ {eq_id} {old} -> {new}")
                    changed = True
                for j, t, c in zip(group, best['terms'], best['coeffs']):
                    post_specs[j]['kind'] = t['kind']
                    post_specs[j]['tau'] = t['tau']
                    post_specs[j]['coeff_post'] = c

            if not changed:
                break

    # ------------------------------------------------------------------ #
    # Design matrix
    # ------------------------------------------------------------------ #
    def build_design_matrix(self) -> np.ndarray:
        """Build design matrix G with multi-tier postseismic selection.

        Filtering tiers:
        1. Magnitude threshold (postseismic_mag_threshold, otherwise is always applied)
        2. Step amplitude threshold (always applied)
        3. Sign consistency (postseismic must match step sign)
        4. Statistical significance (ΔAIC/ΔBIC/F-test, on weighted chi-square)
        Then optional backfitting of tau values.

        The accepted terms are stored in self.post_specs, and the same specs are
        used by predict(), so fit and prediction share one assembly path.
        """
        templates = TemplateFunctions(self.t_days, self.t0)
        builder = DesignMatrixBuilder(templates, self.config)
        self.post_specs = []

        if not self.config.fit_postseismic_decay:
            self.G, self.template_names = builder.build(self.selected_sse, self.selected_eq,
                                                        selected_steps=self.selected_steps)
            logger.info(f"Postseismic disabled, final matrix: {self.G.shape}")
            return self.G

        # Step 1: baseline + SSE + EQ steps (no postseismic)
        G_baseline, names_baseline = builder.build_baseline(self.selected_sse, self.selected_eq, self.selected_steps)
        logger.info(f"Baseline design matrix: {G_baseline.shape[0]} x {G_baseline.shape[1]}")

        # Step 2: baseline fit (weighted chi-square and step coefficients)
        fitter = OLSFitter()
        weights = self._default_weights()
        n_obs = int(np.sum(self._valid_obs_mask(weights)))

        out = self._fit_chi2(G_baseline, weights, fitter)
        if out is None:
            logger.error("Failed to fit baseline model; postseismic selection skipped")
            self.G, self.template_names = G_baseline, names_baseline
            return self.G
        chi2_baseline, coeffs_baseline = out
        logger.info(f"Baseline chi2: {chi2_baseline:.2f} (n={n_obs})")

        # Step 3: magnitude and step-amplitude filters
        candidates = []
        for i, eq in enumerate(self.selected_eq):
            eq_mag = eq.get('magnitude', 0)
            event_type = eq.get('event_type', 'earthquake')

            if not self._should_consider_postseismic(eq):
                if event_type != 'earthquake':
                    logger.info(f"  EQ {i} ({event_type}): event type excludes postseismic -> SKIPPED")
                else:
                    logger.debug(f"  EQ {i} (M{eq_mag:.1f}): below magnitude threshold -> SKIPPED")
                continue

            idx_step = builder.eq_step_index(names_baseline, i)
            if idx_step is None:
                logger.warning(f"  EQ {i}: step column not found -> SKIPPED")
                continue
            coeff_step = float(coeffs_baseline[idx_step])

            if abs(coeff_step) < self.config.postseismic_min_step_amplitude:
                logger.info(f"  EQ {i} (M{eq_mag:.1f}): step={coeff_step:.2f}mm "
                            f"< {self.config.postseismic_min_step_amplitude:.1f}mm -> SKIPPED")
                continue

            candidates.append((i, eq, coeff_step))

        if not candidates:
            logger.info("No earthquakes passed magnitude/amplitude filters")
            self.G, self.template_names = G_baseline, names_baseline
            return self.G

        logger.info(f"Filtered to {len(candidates)}/{len(self.selected_eq)} candidate EQs "
                    f"(criterion: {self.config.postseismic_selection_criterion})")

        # Step 4: greedy forward selection
        post_specs: List[Dict] = []
        chi2_current = chi2_baseline

        for eq_idx, eq, coeff_step in candidates:
            eq_day = eq['eq_day']
            eq_mag = eq.get('magnitude', 0)

            is_sig, chosen, metrics = self._test_postseismic_significance(
                eq_idx=eq_idx, eq_day=eq_day, G_baseline=G_baseline,
                post_specs=post_specs, chi2_baseline=chi2_current, n_obs=n_obs,
                coeff_step_ref=coeff_step, weights=weights,
                builder=builder, fitter=fitter,
                eq_mag=float(eq_mag or 0.0))

            if is_sig and chosen is not None:
                for term, coeff in zip(chosen['terms'], chosen['coeffs']):
                    post_specs.append({
                        'eq_idx': eq_idx,
                        'eq_day': eq_day,
                        'kind': term['kind'],
                        'tau': term['tau'],
                        'coeff_step': coeff_step,
                        'coeff_post': coeff,
                    })
                logger.info(f"  EQ {eq_idx} (M{eq_mag:.1f}, step={coeff_step:.1f}mm): "
                            f"{metrics['desc']}, {metrics['reason']} -> KEPT")
                chi2_current = metrics['chi2_with']
            else:
                reason = metrics.get('reason', 'failed')
                logger.info(f"  EQ {eq_idx} (M{eq_mag:.1f}, step={coeff_step:.1f}mm): "
                            f"{reason} -> SKIPPED")

        # Step 5: backfitting of tau values
        n_passes = int(getattr(self.config, 'postseismic_backfit_passes', 2))
        self._backfit_tau(G_baseline, post_specs, weights, builder, fitter, n_passes)

        # Step 6: final assembly through the shared path
        self.post_specs = post_specs
        self.G, self.template_names = builder.build(self.selected_sse, self.selected_eq, post_specs=post_specs,
                                                    selected_steps=self.selected_steps)

        logger.info(f"Final design matrix: {self.G.shape[0]} x {self.G.shape[1]} "
                    f"({len(post_specs)}/{len(candidates)} postseismic)")
        return self.G

    # ------------------------------------------------------------------ #
    # SSE timing optimisation
    # ------------------------------------------------------------------ #
    @staticmethod
    def _post_signature(post_specs: List[Dict]) -> Tuple:
        """Hashable summary of the postseismic decisions (which EQs, which tau)."""
        return tuple(sorted((s['eq_idx'], s.get('kind', 'log'), s['tau']) for s in post_specs))

    @staticmethod
    def _sse_times(events: List[Dict]) -> Tuple:
        return tuple((e['start_day'], e['end_day']) for e in events)

    def reset_sse_timing(self) -> None:
        """Go back to the catalog SSE times (and rebuild on the next fit)."""
        if self._sse_catalog_selected is not None:
            self.selected_sse = [dict(e) for e in self._sse_catalog_selected]
        self._sse_catalog_selected = None
        self._sse_timing_done = False
        self.sse_timing_report = None
        self.G = None
        self.post_specs = []

    def _optimize_sse_timing(self) -> None:
        """Alternate SSE timing optimisation and postseismic selection.

        Each round:
          1. build the design matrix with the current SSE times
             (this also runs the postseismic selection),
          2. optimise all SSE start/end times against everything that is NOT an SSE
             (baseline, EQ steps, accepted postseismic terms), always starting from
             the catalog times, so the tolerance is centred on the catalog,
          3. stop if the SSE times and the postseismic decisions no longer change.
        The design matrix is always rebuilt at the end with the final SSE times,
        which re-runs the postseismic selection and its tau backfitting.
        """
        if not self.selected_sse:
            self.build_design_matrix()
            self._sse_timing_done = True
            return

        if self._sse_catalog_selected is None:
            self._sse_catalog_selected = [dict(e) for e in self.selected_sse]
        catalog_sse = self._sse_catalog_selected

        max_rounds = int(getattr(self.config, 'sse_timing_max_rounds', 2))
        weights = self._default_weights()

        templates = TemplateFunctions(self.t_days, self.t0)
        builder = DesignMatrixBuilder(templates, self.config)

        self.selected_sse = [dict(e) for e in catalog_sse]
        history = []
        converged = False
        built_current = False
        result = None
        first_post = None

        for rnd in range(max_rounds):
            self.build_design_matrix()
            built_current = True
            sig = self._post_signature(self.post_specs)
            if first_post is None:
                first_post = [dict(s) for s in self.post_specs]

            G_other, _ = builder.build([], self.selected_eq, post_specs=self.post_specs,
                                       selected_steps=self.selected_steps)
            result = optimize_sse_timing(
                t_days=self.t_days, t0=self.t0, y=self.y, weights=weights,
                G_other=G_other, sse_events=catalog_sse, config=self.config)

            new_times = self._sse_times(result.events)
            changed_sse = new_times != self._sse_times(self.selected_sse)
            history.append({
                'round': rnd + 1,
                'n_moved': result.n_moved,
                'chi2_catalog': result.chi2_catalog,
                'chi2_final': result.chi2_final,
                'delta_ic': result.delta_ic,
                'post_signature': sig,
            })

            if not changed_sse:
                converged = True
                break

            self.selected_sse = [dict(e) for e in result.events]
            built_current = False

            # the postseismic decisions for the new SSE times
            self.build_design_matrix()
            built_current = True
            if self._post_signature(self.post_specs) == sig:
                converged = True
                break

        if not built_current:
            self.build_design_matrix()

        if not converged:
            logger.warning(
                f"SSE timing / postseismic selection did not converge in {max_rounds} "
                f"round(s); SSE and postseismic terms may not be separable at this station")

        self._sse_timing_done = True
        self.sse_timing_report = {
            'converged': converged,
            'n_rounds': len(history),
            'rounds': history,
            'events': result.report if result is not None else [],
            'chi2_catalog': result.chi2_catalog if result is not None else None,
            'chi2_final': result.chi2_final if result is not None else None,
            'delta_ic': result.delta_ic if result is not None else None,
            'criterion': self.config.sse_timing_criterion,
            'post_specs_initial': first_post,
            'post_specs_final': [dict(s) for s in self.post_specs],
        }

    # ------------------------------------------------------------------ #
    # Event selection from catalogs
    # ------------------------------------------------------------------ #
    def _select_sse_events(self) -> List[Dict]:
        selected = []
        for ev in self.sse_catalog:
            evc = dict(ev)
            evc['start_day'] = ev['start']
            evc['end_day'] = ev['end']

            # Keep SSE if it overlaps with the time series at all (conservative)
            if evc['end_day'] < self.t.min() or evc['start_day'] > self.t.max():
                logger.debug(f"Skipping SSE outside time range: "
                             f"{evc['start_day']} to {evc['end_day']}")
                continue

            if self.config.d_param_sse is None:
                selected.append(evc)
            else:
                dist = self.geocalc.distance_km(
                    self.station_lat, self.station_lon, ev['lat'], ev['lon'])
                rad = self.geocalc.radius_from_magnitude(ev['magnitude'], self.config.d_param_sse)
                if dist <= rad:
                    selected.append(evc)

        if self.config.merge_close_sse:
            selected = EventCatalog.merge_close_events(
                selected, gap_threshold=self.config.max_overlap_merge_sse)

        return selected

    def _select_eq_events(self) -> List[Dict]:
        """Select earthquake events within radius of influence."""
        selected = []
        for ev in self.eq_catalog:
            evc = dict(ev)
            evc['eq_day'] = ev.get('date') or ev.get('time')

            if evc['eq_day'] is None:
                logger.warning(f"Skipping earthquake without date/time: {ev}")
                continue

            if evc['eq_day'] < self.t.min() or evc['eq_day'] > self.t.max():
                logger.debug(f"Skipping earthquake outside time range: {evc['eq_day']}")
                continue

            if self.config.d_param_eq is not None:
                dist = self.geocalc.distance_km(
                    self.station_lat, self.station_lon, ev['lat'], ev['lon'])
                rad = self.geocalc.radius_from_magnitude(ev['magnitude'], self.config.d_param_eq)
                if dist <= rad:
                    selected.append(evc)
            else:
                selected.append(evc)

        if self.config.merge_earthquakes_same_day:
            selected = EventCatalog.merge_earthquakes_by_date(selected)

        return selected

    def _select_step_events(self) -> List[Dict]:
        """Station-specific non-earthquake steps: time-range filter only.

        No distance filter, no magnitude, no postseismic. Exact duplicates (same day and
        same label) are dropped; different labels on the same day each keep their column.
        """
        selected, seen = [], set()
        t_min, t_max = self.t.min(), self.t.max()
        for ev in self.step_catalog:
            day = ev.get('date') or ev.get('time') or ev.get('step_day')
            if day is None:
                logger.warning(f"Skipping step without date: {ev}")
                continue
            if isinstance(day, np.datetime64):
                day = day.astype('datetime64[s]').astype(datetime.datetime)

            if day <= t_min or day > t_max:
                logger.debug(f"Skipping step outside time range: {day}")
                continue

            label = ev.get('event_type', 'step')
            key = (day.date(), label)
            if key in seen:
                logger.warning(f"Duplicate step '{label}' on {day.date()} dropped")
                continue
            seen.add(key)

            evc = dict(ev)
            evc['step_day'] = day
            evc['event_type'] = label
            selected.append(evc)

        selected.sort(key=lambda e: e['step_day'])

        days = {}
        for e in selected:
            days.setdefault(e['step_day'].date(), []).append(e['event_type'])
        eq_days = {(eq.get('eq_day') or eq.get('date')).date()
                   for eq in self.selected_eq if (eq.get('eq_day') or eq.get('date')) is not None}
        for d, labels in days.items():
            if len(labels) > 1 or d in eq_days:
                logger.warning(f"Step(s) {labels} on {d} share a day with another step or an "
                               f"earthquake: identical columns, amplitudes not separable")
        return selected

    # ------------------------------------------------------------------ #
    # Event pruning
    # ------------------------------------------------------------------ #
    def select_events(self, selector: EventSelector,
                      event_type: str = 'sse', plot: bool = False,
                      save_plot: Optional[str] = None,
                      update_catalog: bool = True) -> List[int]:
        """Select events using a given strategy.

        Args:
            selector: EventSelector instance
            event_type: 'sse', 'eq' or 'step'

        Returns:
            List of selected event indices
        """
        templates = TemplateFunctions(self.t_days, self.t0)
        builder = DesignMatrixBuilder(templates, self.config)

        baseline_cols = []
        baseline_cols.extend(builder._build_base_model())
        if self.config.include_seasonal:
            baseline_cols.extend(builder._build_seasonal())

        G_baseline = (np.column_stack(baseline_cols) if baseline_cols
                      else np.empty((len(self.t_days), 0)))

        if event_type not in ('sse', 'eq', 'step'):
            raise ValueError("event_type must be 'sse', 'eq' or 'step'")
        if event_type == 'sse':
            event_catalog = self.selected_sse
        elif event_type == 'step':
            event_catalog = self.selected_steps
        else:
            event_catalog = self.selected_eq

        event_templates_list = []

        for i, ev in enumerate(event_catalog):
            if event_type == 'sse':
                template = templates.raised_cosine(ev['start_day'], ev['end_day'])
                name = f"SSE_{i}"
            elif event_type == 'step':
                template = templates.step(ev['step_day'])
                name = f"STEP_{i}"
            else:
                template = templates.step(ev['eq_day'])
                name = f"EQ_{i}"
            event_templates_list.append({'template': template, 'name': name})

        valid_mask = ~np.isnan(self.y)
        selected_indices = selector.select(self.y, G_baseline, event_templates_list, valid_mask)

        logger.info(f"Selected {len(selected_indices)}/{len(event_catalog)} {event_type} events")

        if update_catalog:
            if event_type == 'sse':
                original_count = len(self.selected_sse)
                self.selected_sse = [self.selected_sse[i] for i in selected_indices]
                logger.info(f"Updated SSE catalog: {original_count} -> {len(self.selected_sse)} events")
            elif event_type == 'step':
                original_count = len(self.selected_steps)
                self.selected_steps = [self.selected_steps[i] for i in selected_indices]
                logger.info(f"Updated step catalog: {original_count} -> {len(self.selected_steps)} events")
            else:
                original_count = len(self.selected_eq)
                self.selected_eq = [self.selected_eq[i] for i in selected_indices]
                logger.info(f"Updated EQ catalog: {original_count} -> {len(self.selected_eq)} events")

            # Event indices changed: force a full rebuild (incl. postseismic selection)
            self.G = None
            self.post_specs = []
            self._sse_catalog_selected = None
            self._sse_timing_done = False
            self.sse_timing_report = None

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
        selector = KneeEventSelector(criterion=criterion, method=method)
        selected_indices = self.select_events(
            selector, event_type=event_type, plot=plot,
            save_plot=save_plot, update_catalog=True)  # updates selected_sse / selected_eq once
        self.G = None

        results = self.fit(weights=weights)

        results.metadata['event_selection'] = {
            'criterion': criterion,
            'method': method,
            'n_selected': len(selected_indices),
            'selected_indices': selected_indices,
        }
        return results

    # ------------------------------------------------------------------ #
    # Fit / predict
    # ------------------------------------------------------------------ #
    def fit(self, method: Optional[FittingStrategy] = None,
            weights: Optional[np.ndarray] = None,
            compute_uncertainty: bool = True,
            optimize_sse_timing: Optional[bool] = None) -> ModelResults:

        """Fit the trajectory model.

        Args:
            method: Fitting strategy (default: OLS)
            weights: Optional observation weights (default 1/sigma)
            compute_uncertainty: Whether to compute uncertainty estimates
            optimize_sse_timing: Optimise SSE start/end times (None = use config.sse_optimize_timing)

        Returns:
            ModelResults object
        """
        do_timing = (self.config.sse_optimize_timing if optimize_sse_timing is None
                     else optimize_sse_timing)

        if do_timing and not self._sse_timing_done:
            self._optimize_sse_timing()
        elif self.G is None:
            self.build_design_matrix()

        fitter = method or OLSFitter()
        weights = self._prepare_weights(weights)

        fit_result = fitter.fit(self.G, self.y, weights)

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
                compute_vif=True,
                weights=weights,
            )

        metadata = {k: v for k, v in fit_result.items()
                    if k not in ['coeffs', 'fitted', 'residuals']}
        metadata['method'] = fitter.__class__.__name__
        metadata['uncertainty'] = uncertainty_results
        metadata['post_specs'] = [dict(s) for s in self.post_specs]
        metadata['sse_timing'] = self.sse_timing_report

        self.results = ModelResults.from_fit(
            coeffs=fit_result['coeffs'],
            fitted=fit_result['fitted'],
            residuals=fit_result['residuals'],
            template_names=self.template_names,
            weights=weights,
            **metadata,
        )

        logger.info(f"Fit complete: RMS = {self.results.rms:.3f} mm")
        return self.results

    def predict(self, t_pred: np.ndarray) -> np.ndarray:
        """Predict at new time points.

        Uses the same builder path and the same postseismic terms (self.post_specs)
        as the fit, so the columns always match the fitted coefficients.

        Args:
            t_pred: Prediction times (datetime array)

        Returns:
            Predicted values

        Raises:
            ValueError: If model not fitted or times invalid
        """
        if self.results is None:
            raise ValueError("Model must be fit before prediction")

        t_pred = np.asarray(t_pred, dtype='datetime64')
        if len(t_pred) == 0:
            raise ValueError("t_pred cannot be empty")

        t_min, t_max = self.t.min(), self.t.max()
        if t_pred.min() < t_min or t_pred.max() > t_max:
            import warnings
            warnings.warn(
                f"Prediction times [{t_pred.min()}, {t_pred.max()}] "
                f"outside training range [{t_min}, {t_max}]. "
                f"Extrapolation may be unreliable.",
                UserWarning,
            )

        t_days_pred = (t_pred - self.t0).astype('timedelta64[D]').astype(float)
        templates_pred = TemplateFunctions(t_days_pred, self.t0)
        builder = DesignMatrixBuilder(templates_pred, self.config)
        G_pred, names_pred = builder.build(
            self.selected_sse, self.selected_eq, post_specs=self.post_specs, selected_steps=self.selected_steps)

        if G_pred.shape[1] != len(self.results.coeffs):
            raise ValueError(
                f"Prediction design matrix has {G_pred.shape[1]} columns but the fit has "
                f"{len(self.results.coeffs)} coefficients; re-fit the model before predicting.")

        return G_pred @ self.results.coeffs
