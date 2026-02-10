"""Design matrix construction for trajectory models."""
import numpy as np
from typing import List, Dict, Tuple
import logging

from src.trajmod.config.modelconfig import ModelConfig
from src.trajmod.templates.templates import TemplateFunctions

logger = logging.getLogger(__name__)


class DesignMatrixBuilder:
    """Builder for design matrix G."""

    def __init__(self, templates: TemplateFunctions, config: ModelConfig):
        self.templates = templates
        self.config = config
        self.template_names: List[str] = []

    def build(self, selected_sse: List[Dict], selected_eq: List[Dict]) -> Tuple[np.ndarray, List[str]]:
        """Build complete design matrix."""
        self.template_names = []
        columns = []

        columns.extend(self._build_base_model())

        if self.config.include_seasonal:
            columns.extend(self._build_seasonal())

        columns.extend(self._build_sse_templates(selected_sse))
        columns.extend(self._build_earthquake_templates(selected_eq))

        G = np.column_stack(columns) if columns else np.empty((len(self.templates.t_days), 0))

        logger.info(f"Design matrix: {G.shape[0]} x {G.shape[1]}")
        return G, self.template_names

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
                columns.append(self.templates.seasonal_sin(T))
                self.template_names.append(f'envelope_sin_{int(T / 365.25)}yr')
                columns.append(self.templates.seasonal_cos(T))
                self.template_names.append(f'envelope_cos_{int(T / 365.25)}yr')

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
            eq_mag = eq.get('magnitude', 0)

            step = self.templates.step(eq_date)
            columns.append(step)
            eq_str = eq_date.strftime('%Y-%m-%d')
            self.template_names.append(f'EQ_{i}_{eq_str}_step')

            if eq_mag >= self.config.postseismic_mag_threshold:
                for tau in self.config.tau_grid:
                    log_dec = self.templates.log_decay(eq_date, tau)
                    columns.append(log_dec)
                    self.template_names.append(f'EQ_{i}_{eq_str}_log_tau{tau}')

        return columns
