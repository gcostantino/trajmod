"""Design matrix construction for trajectory models."""
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from trajmod.config.modelconfig import ModelConfig
from trajmod.templates.templates import TemplateFunctions

logger = logging.getLogger(__name__)

# Periods already covered by the annual / semiannual terms of _build_seasonal
_BASE_SEASONAL_PERIODS = (365.25, 365.25 / 2)


class DesignMatrixBuilder:
    """Builder for design matrix G.

    Column order (stable, relied upon by name-based lookups):
        offset, trend, [acceleration], [seasonal], SSE_*, EQ_*_step, STEP_*,
        EQ_*_log_tau*, EQ_*_exp_tau*

    Column names:
        SSE_{i}_{start}_to_{end}             SSE raised-cosine template
        EQ_{i}_{date}_step                   earthquake step
        STEP_{i}_{date}_{label}              non-earthquake step (antenna, equipment, ...)
        EQ_{i}_{date}_log_tau{tau}           log postseismic term
        EQ_{i}_{date}_exp_tau{tau}           exponential postseismic term

    Postseismic columns follow the order of post_specs, one column per term. An
    earthquake with a log+exp pair has two consecutive specs with the same eq_idx,
    so its log and exp columns are not necessarily adjacent to other earthquakes'
    columns of the same kind.
    """

    def __init__(self, templates: TemplateFunctions, config: ModelConfig):
        self.templates = templates
        self.config = config
        self.template_names: List[str] = []

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def build(self,
              selected_sse: List[Dict],
              selected_eq: List[Dict],
              post_specs: Optional[List[Dict]] = None,
              selected_steps: Optional[List[Dict]] = None) -> Tuple[np.ndarray, List[str]]:
        """Build the complete design matrix.

        Args:
            selected_sse: SSE event dicts (with 'start_day', 'end_day').
            selected_eq: Earthquake event dicts (with 'eq_day').
            post_specs: Optional accepted postseismic terms, one dict per term:
                {'eq_idx': int, 'eq_day': datetime, 'tau': float, 'kind': 'log' | 'exp'}.
                'kind' defaults to 'log' when absent. 'eq_idx' is the index of the
                earthquake in selected_eq. A log+exp pair is two terms with the same
                'eq_idx'. If None or empty, no postseismic columns are added.

        Returns:
            (G, template_names)
        """
        self.template_names = []
        columns = self.build_baseline_columns(selected_sse, selected_eq, selected_steps)

        if post_specs:
            columns.extend(self._build_postseismic_templates(post_specs))

        G = self._stack(columns)
        logger.debug(f"Design matrix: {G.shape[0]} x {G.shape[1]}")
        return G, self.template_names

    def build_baseline_columns(self,
                               selected_sse: List[Dict],
                               selected_eq: List[Dict],
                               selected_steps: Optional[List[Dict]] = None) -> List[np.ndarray]:
        """Baseline + SSE + EQ steps (no postseismic).

        Resets template_names and fills it with the names of the returned columns.
        """
        self.template_names = []
        columns: List[np.ndarray] = []
        columns.extend(self._build_base_model())
        if self.config.include_seasonal:
            columns.extend(self._build_seasonal())
        columns.extend(self._build_sse_templates(selected_sse))
        columns.extend(self._build_earthquake_templates(selected_eq))
        columns.extend(self._build_step_templates(selected_steps))
        return columns

    def build_baseline(self,
                       selected_sse: List[Dict],
                       selected_eq: List[Dict],
                       selected_steps: Optional[List[Dict]] = None) -> Tuple[np.ndarray, List[str]]:
        """Same as build() without postseismic terms, returned as (G, names)."""
        columns = self.build_baseline_columns(selected_sse, selected_eq, selected_steps)
        return self._stack(columns), list(self.template_names)

    def make_postseismic_column(self, eq_day, tau: float, kind: str = 'log') -> np.ndarray:
        """Single postseismic column ('log' afterslip-like, 'exp' viscoelastic)."""
        if kind == 'log':
            return self.templates.log_decay(eq_day, tau)
        if kind == 'exp':
            return self.templates.exp_decay(eq_day, tau)
        raise ValueError(f"Unknown postseismic kind '{kind}'")

    @staticmethod
    def postseismic_name(eq_idx: int, eq_day, tau: float, kind: str = 'log') -> str:
        """Canonical name of a postseismic column."""
        return f"EQ_{eq_idx}_{eq_day.strftime('%Y-%m-%d')}_{kind}_tau{tau}"

    @staticmethod
    def eq_step_index(names: List[str], eq_idx: int) -> Optional[int]:
        """Column index of the step of earthquake `eq_idx`, looked up by name.

        Replaces any hard-coded column arithmetic. Returns None if not found.
        """
        prefix = f"EQ_{eq_idx}_"
        for j, name in enumerate(names):
            if name.startswith(prefix) and name.endswith("_step"):
                return j
        return None

    # ------------------------------------------------------------------ #
    # Internal builders
    # ------------------------------------------------------------------ #
    def _stack(self, columns: List[np.ndarray]) -> np.ndarray:
        if columns:
            return np.column_stack(columns)
        return np.empty((len(self.templates.t_days), 0))

    def _build_base_model(self) -> List[np.ndarray]:
        columns = []
        columns.append(self.templates.offset())
        self.template_names.append('offset')
        columns.append(self.templates.trend())
        self.template_names.append('trend')

        if self.config.acceleration_term:
            columns.append(self.templates.acceleration())
            self.template_names.append('acceleration')

        return columns

    def _build_seasonal(self) -> List[np.ndarray]:
        columns = []
        columns.append(self.templates.seasonal_sin(365.25))
        self.template_names.append('annual_sin')
        columns.append(self.templates.seasonal_cos(365.25))
        self.template_names.append('annual_cos')
        columns.append(self.templates.seasonal_sin(365.25 / 2))
        self.template_names.append('semiannual_sin')
        columns.append(self.templates.seasonal_cos(365.25 / 2))
        self.template_names.append('semiannual_cos')

        if self.config.use_envelope_basis:
            for T in self.config.envelope_periods:
                # skip periods already present above (would make G rank deficient)
                if any(np.isclose(T, p, rtol=1e-6) for p in _BASE_SEASONAL_PERIODS):
                    logger.debug(f"Envelope period {T:.2f} d duplicates a base seasonal term, skipped")
                    continue
                columns.append(self.templates.seasonal_sin(T))
                self.template_names.append(f'envelope_sin_{int(T)}d')
                columns.append(self.templates.seasonal_cos(T))
                self.template_names.append(f'envelope_cos_{int(T)}d')

        return columns

    def _build_sse_templates(self, selected_sse: List[Dict]) -> List[np.ndarray]:
        columns = []
        for i, event in enumerate(selected_sse):
            rc = self.templates.raised_cosine(event['start_day'], event['end_day'])
            columns.append(rc)
            start_str = event['start_day'].strftime('%Y-%m-%d')
            end_str = event['end_day'].strftime('%Y-%m-%d')
            self.template_names.append(f'SSE_{i}_{start_str}_to_{end_str}')
        return columns

    def _build_earthquake_templates(self, selected_eq: List[Dict]) -> List[np.ndarray]:
        columns = []
        for i, eq in enumerate(selected_eq):
            eq_date = eq['eq_day']
            columns.append(self.templates.step(eq_date))
            eq_str = eq_date.strftime('%Y-%m-%d')
            self.template_names.append(f'EQ_{i}_{eq_str}_step')
        return columns

    @staticmethod
    def step_name(i: int, ev: Dict) -> str:
        """Canonical name of a non-earthquake step column (no underscores inside the label)."""
        label = str(ev.get('event_type', 'step')).replace('_', '-').replace(' ', '-')
        return f"STEP_{i}_{ev['step_day'].strftime('%Y-%m-%d')}_{label}"

    def _build_step_templates(self, selected_steps: Optional[List[Dict]]) -> List[np.ndarray]:
        columns = []
        for i, ev in enumerate(selected_steps or []):
            columns.append(self.templates.step(ev['step_day']))
            self.template_names.append(self.step_name(i, ev))
        return columns

    def _build_postseismic_templates(self, post_specs: List[Dict]) -> List[np.ndarray]:
        columns = []
        for spec in post_specs:
            kind = spec.get('kind', 'log')
            columns.append(self.make_postseismic_column(spec['eq_day'], spec['tau'], kind))
            self.template_names.append(
                self.postseismic_name(spec['eq_idx'], spec['eq_day'], spec['tau'], kind)
            )
        return columns
