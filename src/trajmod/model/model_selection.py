"""Model selection utilities for trajectory models."""
import logging
from typing import Tuple

import numpy as np
from scipy.special import erfc

logger = logging.getLogger(__name__)


class ModelSelector:
    """Utilities for model selection and diagnostics."""

    @staticmethod
    def compute_aic_bic(y: np.ndarray, G: np.ndarray,
                        valid_mask: np.ndarray) -> Tuple[float, float, float]:
        """Compute SSR, AIC, and BIC for a given model.

        Args:
            y: Observations (N,)
            G: Design matrix (N x P)
            valid_mask: Boolean mask of valid (non-NaN) observations

        Returns:
            Tuple of (SSR, AIC, BIC)
        """
        if G.shape[1] == 0:
            # Empty model - just use mean
            residuals = y.copy()
            residuals[valid_mask] = y[valid_mask] - 0.0
            ssr = np.nansum(residuals[valid_mask] ** 2)
            n = np.sum(valid_mask)
            return ssr, np.inf, np.inf

        # Fit model
        Gsub = G[valid_mask, :]
        ysub = y[valid_mask]

        try:
            coeffs, _, _, _ = np.linalg.lstsq(Gsub, ysub, rcond=None)
            fitted = G @ coeffs
            residuals = y - fitted
            ssr = np.nansum(residuals[valid_mask] ** 2)
        except np.linalg.LinAlgError:
            return np.inf, np.inf, np.inf

        n = np.sum(valid_mask)  # Number of observations
        p = G.shape[1]  # Number of parameters

        # AIC = n*ln(SSR/n) + 2*p
        if ssr > 0 and n > p:
            aic = n * np.log(ssr / n) + 2 * p
            # BIC = n*ln(SSR/n) + p*ln(n)
            bic = n * np.log(ssr / n) + p * np.log(n)
        else:
            aic = np.inf
            bic = np.inf

        return ssr, aic, bic

    @staticmethod
    def find_knee_by_curvature(x: np.ndarray, y: np.ndarray) -> int:
        """Find knee/elbow point using curvature method.

        The knee point is where the curve has maximum curvature.

        Args:
            x: X-coordinates (e.g., number of events)
            y: Y-coordinates (e.g., SSR values)

        Returns:
            Index of knee point
        """
        if len(x) < 3:
            return 0

        # Normalize to [0, 1] range
        x_norm = (x - x.min()) / (x.max() - x.min() + 1e-10)
        y_norm = (y - y.min()) / (y.max() - y.min() + 1e-10)

        # Compute distances from each point to the line connecting endpoints
        p1 = np.array([x_norm[0], y_norm[0]])
        p2 = np.array([x_norm[-1], y_norm[-1]])

        distances = np.zeros(len(x))
        for i in range(len(x)):
            p = np.array([x_norm[i], y_norm[i]])
            # Distance from point to line
            distances[i] = np.abs(np.cross(p2 - p1, p1 - p)) / np.linalg.norm(p2 - p1)

        # Knee is at maximum distance
        knee_idx = int(np.argmax(distances))

        # Avoid selecting endpoints
        if knee_idx == 0:
            knee_idx = 1
        elif knee_idx == len(x) - 1:
            knee_idx = len(x) - 2

        return knee_idx

    @staticmethod
    def compute_pvalue(t_statistic: float) -> float:
        """Compute two-tailed p-value from t-statistic.

        Args:
            t_statistic: T-statistic value

        Returns:
            Two-tailed p-value
        """
        return erfc(np.abs(t_statistic) / np.sqrt(2.0))

    @staticmethod
    def benjamini_hochberg(pvalues: np.ndarray, alpha: float = 0.05) -> np.ndarray:
        """Benjamini-Hochberg FDR control.

        Args:
            pvalues: Array of p-values
            alpha: FDR level

        Returns:
            Boolean mask of selected features
        """
        m = len(pvalues)
        if m == 0:
            return np.array([], dtype=bool)

        # Sort p-values and track original indices
        sorted_indices = np.argsort(pvalues)
        sorted_pvals = pvalues[sorted_indices]

        # Find largest k such that p(k) <= k/m * alpha
        thresholds = (np.arange(1, m + 1) / m) * alpha
        passing = sorted_pvals <= thresholds

        if not np.any(passing):
            return np.zeros(m, dtype=bool)

        # Find largest passing index
        k_max = np.where(passing)[0][-1]

        # Create mask
        selected = np.zeros(m, dtype=bool)
        selected[sorted_indices[:k_max + 1]] = True

        return selected

    @staticmethod
    def mad_sigma(data: np.ndarray) -> float:
        """Robust standard deviation estimate using MAD.

        Args:
            data: Data array

        Returns:
            Robust sigma estimate
        """
        data_valid = data[np.isfinite(data)]
        if len(data_valid) == 0:
            return 0.0
        med = np.median(data_valid)
        mad = np.median(np.abs(data_valid - med))
        return 1.4826 * mad


class NestedModelSelector:
    """Select optimal model from nested sequence using AIC/BIC/SSR."""

    def __init__(self, criterion: str = 'bic'):
        """Initialize selector.

        Args:
            criterion: Selection criterion ('aic', 'bic', or 'ssr')
        """
        if criterion not in ['aic', 'bic', 'ssr']:
            raise ValueError(f"Unknown criterion: {criterion}")
        self.criterion = criterion
        self.selector = ModelSelector()

    def select(self, y: np.ndarray, baseline_G: np.ndarray,
               event_templates: list, valid_mask: np.ndarray,
               method: str = 'knee') -> dict:
        """Select optimal number of events.

        Args:
            y: Observations
            baseline_G: Baseline design matrix (offset, trend, seasonal)
            event_templates: List of event template columns
            valid_mask: Valid observation mask
            method: 'knee' or 'min' (minimize criterion)

        Returns:
            Dictionary with selected model information
        """
        n_events = len(event_templates)

        if n_events == 0:
            return {
                'n_selected': 0,
                'selected_indices': [],
                'ssr': self.selector.compute_aic_bic(y, baseline_G, valid_mask)[0],
                'aic': self.selector.compute_aic_bic(y, baseline_G, valid_mask)[1],
                'bic': self.selector.compute_aic_bic(y, baseline_G, valid_mask)[2]
            }

        # Fit full model to get event amplitudes
        event_matrix = np.column_stack(event_templates)
        G_full = np.hstack([baseline_G, event_matrix])

        Gsub = G_full[valid_mask, :]
        ysub = y[valid_mask]
        coeffs_full, _, _, _ = np.linalg.lstsq(Gsub, ysub, rcond=None)

        # Get event coefficients and sort by amplitude
        n_baseline = baseline_G.shape[1]
        event_coeffs = coeffs_full[n_baseline:]
        sorted_indices = np.argsort(-np.abs(event_coeffs))  # Descending

        # Evaluate nested models
        ssr_path = np.zeros(n_events + 1)
        aic_path = np.zeros(n_events + 1)
        bic_path = np.zeros(n_events + 1)

        # k=0: baseline only
        ssr_path[0], aic_path[0], bic_path[0] = self.selector.compute_aic_bic(
            y, baseline_G, valid_mask
        )

        # k=1 to n_events
        for k in range(1, n_events + 1):
            selected_events = sorted_indices[:k]
            G_k = np.hstack([baseline_G, event_matrix[:, selected_events]])
            ssr_path[k], aic_path[k], bic_path[k] = self.selector.compute_aic_bic(
                y, G_k, valid_mask
            )

        # Select optimal k
        if method == 'min':
            if self.criterion == 'ssr':
                k_opt = int(np.argmin(ssr_path))
            elif self.criterion == 'aic':
                k_opt = int(np.argmin(aic_path))
            else:  # bic
                k_opt = int(np.argmin(bic_path))
        else:  # knee
            if self.criterion == 'ssr':
                k_opt = self.selector.find_knee_by_curvature(
                    np.arange(len(ssr_path)), ssr_path
                )
            elif self.criterion == 'aic':
                k_opt = self.selector.find_knee_by_curvature(
                    np.arange(len(aic_path)), aic_path
                )
            else:  # bic
                k_opt = self.selector.find_knee_by_curvature(
                    np.arange(len(bic_path)), bic_path
                )

        selected_indices = sorted_indices[:k_opt].tolist() if k_opt > 0 else []

        return {
            'n_selected': k_opt,
            'selected_indices': selected_indices,
            'sorted_indices': sorted_indices.tolist(),
            'ssr_path': ssr_path,
            'aic_path': aic_path,
            'bic_path': bic_path,
            'ssr': ssr_path[k_opt],
            'aic': aic_path[k_opt],
            'bic': bic_path[k_opt],
            'criterion': self.criterion,
            'method': method
        }
