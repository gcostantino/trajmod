"""Event selection strategies for large catalogs."""
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Tuple

import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class EventSelector(ABC):
    """Abstract base class for event selection strategies."""

    @abstractmethod
    def select(self, y: np.ndarray, G_baseline: np.ndarray,
               event_templates: List[Dict], valid_mask: np.ndarray) -> List[int]:
        """Select events from catalog.

        Args:
            y: Observations
            G_baseline: Baseline design matrix
            event_templates: List of dicts with 'template' and 'name' keys
            valid_mask: Valid observation mask

        Returns:
            List of selected event indices
        """
        pass


class LassoEventSelector(EventSelector):
    """LASSO-based event selection.

    Removes baseline signal, then applies LASSO to event templates.
    """

    def __init__(self, cv: int = 5, alpha_multiplier: float = 1.0):
        """Initialize LASSO selector.

        Args:
            cv: Number of CV folds
            alpha_multiplier: Multiplier for LASSO alpha (larger = more sparse)
        """
        self.cv = cv
        self.alpha_multiplier = alpha_multiplier

    def select(self, y: np.ndarray, G_baseline: np.ndarray,
               event_templates: List[Dict], valid_mask: np.ndarray) -> List[int]:
        """Select events using LASSO."""

        if len(event_templates) == 0:
            return []

        # Step 1: Remove baseline signal
        Gsub_baseline = G_baseline[valid_mask, :]
        ysub = y[valid_mask]

        coeffs_baseline, _, _, _ = np.linalg.lstsq(Gsub_baseline, ysub, rcond=None)
        baseline_fit = G_baseline @ coeffs_baseline
        residuals = y - baseline_fit

        # Step 2: Apply LASSO to event templates on residuals
        event_matrix = np.column_stack([ev['template'] for ev in event_templates])
        Gsub_events = event_matrix[valid_mask, :]
        residuals_sub = residuals[valid_mask]

        # Standardize for stability
        scaler = StandardScaler()
        Gsub_scaled = scaler.fit_transform(Gsub_events)

        # Fit LASSO
        lasso = LassoCV(cv=self.cv, n_jobs=-1, positive=False)
        lasso.fit(Gsub_scaled, residuals_sub)

        # Unscale coefficients
        coeffs_events = lasso.coef_ / scaler.scale_

        # Apply alpha multiplier for threshold
        threshold = self.alpha_multiplier * 1e-8
        selected_mask = np.abs(coeffs_events) > threshold
        selected_indices = np.where(selected_mask)[0].tolist()

        logger.info(f"LASSO selected {len(selected_indices)}/{len(event_templates)} events")

        return selected_indices


'''class KneeEventSelector(EventSelector):
    """Knee/elbow-based event selection using AIC/BIC/SSR."""

    def __init__(self, criterion: str = 'bic', method: str = 'knee'):
        """Initialize knee selector.

        Args:
            criterion: 'aic', 'bic', or 'ssr'
            method: 'knee' for elbow detection or 'min' for minimum
        """
        from ..model.model_selection import NestedModelSelector
        self.selector = NestedModelSelector(criterion=criterion)
        self.method = method
        # Store diagnostics for plotting
        self.last_selection_result = None

    def select(self, y: np.ndarray, G_baseline: np.ndarray,
               event_templates: List[Dict], valid_mask: np.ndarray) -> List[int]:
        """Select events using knee/elbow detection."""

        if len(event_templates) == 0:
            return []

        # Convert event templates to list of arrays
        templates = [ev['template'] for ev in event_templates]

        result = self.selector.select(
            y=y,
            baseline_G=G_baseline,
            event_templates=templates,
            valid_mask=valid_mask,
            method=self.method
        )

        self.last_selection_result = result

        selected_indices = result['selected_indices']

        logger.info(f"Knee selector ({self.selector.criterion}/{self.method}) "
                    f"selected {len(selected_indices)}/{len(event_templates)} events")

        return selected_indices

    def plot_selection(self, save_path: Optional[str] = None,
                       figsize: Tuple[int, int] = (12, 6)):
        """Plot the selection diagnostics (SSR/AIC/BIC paths).

        Args:
            save_path: Path to save figure
            figsize: Figure size

        Returns:
            Figure and axes objects
        """
        if self.last_selection_result is None:
            raise ValueError("Must call select() before plotting")

        import matplotlib.pyplot as plt

        result = self.last_selection_result
        fig, axes = plt.subplots(1, 2, figsize=figsize)

        k_values = np.arange(len(result['ssr_path']))
        selected_k = result['n_selected']

        # SSR plot
        ssr_normalized = result['ssr_path'] / result['ssr_path'][0]
        axes[0].plot(k_values, ssr_normalized, '-o', linewidth=2, markersize=6)
        axes[0].axvline(selected_k, color='red', linestyle='--',
                        label=f'Selected k={selected_k}', linewidth=2)
        axes[0].set_xlabel('Number of Events', fontsize=12)
        axes[0].set_ylabel('Normalized SSR', fontsize=12)
        axes[0].set_title('Sum of Squared Residuals', fontsize=13, fontweight='bold')
        axes[0].legend(fontsize=11)
        axes[0].grid(True, alpha=0.3)

        # AIC/BIC plot
        axes[1].plot(k_values, result['aic_path'], '-o', label='AIC',
                     linewidth=2, markersize=6)
        axes[1].plot(k_values, result['bic_path'], '-s', label='BIC',
                     linewidth=2, markersize=6)
        axes[1].axvline(selected_k, color='red', linestyle='--',
                        label=f'Selected k={selected_k}', linewidth=2)
        axes[1].set_xlabel('Number of Events', fontsize=12)
        axes[1].set_ylabel('Information Criterion', fontsize=12)
        axes[1].set_title(f'Model Selection ({self.selector.criterion.upper()} - {self.method})',
                          fontsize=13, fontweight='bold')
        axes[1].legend(fontsize=11)
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved selection plot to {save_path}")

        return fig, axes'''


class KneeEventSelector(EventSelector):
    """Knee/elbow-based event selection using AIC/BIC/SSR.

    Enhanced with curve smoothing and minimum amplitude filtering.
    """

    def __init__(self, criterion: str = 'bic', method: str = 'knee',
                 smooth_curve: bool = True, smooth_window: int = 3,
                 min_amplitude: Optional[float] = None):
        """Initialize knee selector.

        Args:
            criterion: 'aic', 'bic', or 'ssr'
            method: 'knee' for elbow detection or 'min' for minimum
            smooth_curve: Whether to smooth SSR/AIC/BIC curves before knee detection
            smooth_window: Window size for smoothing (odd number, default=3)
            min_amplitude: Minimum absolute amplitude (mm) to keep an event.
                          Events selected by knee but with |amplitude| < min_amplitude
                          will be filtered out.
        """
        from src.trajmod.model.model_selection import NestedModelSelector
        self.selector = NestedModelSelector(criterion=criterion)
        self.method = method
        self.smooth_curve = smooth_curve
        self.smooth_window = smooth_window if smooth_window % 2 == 1 else smooth_window + 1
        self.min_amplitude = min_amplitude
        self.last_selection_result = None

    @staticmethod
    def _smooth_curve(data: np.ndarray, window: int = 3) -> np.ndarray:
        """Apply moving average smoothing to curve.

        Args:
            data: Data to smooth
            window: Window size (should be odd)

        Returns:
            Smoothed data
        """
        if len(data) < window:
            return data

        # Simple moving average
        smoothed = np.copy(data)
        half_window = window // 2

        for i in range(len(data)):
            start = max(0, i - half_window)
            end = min(len(data), i + half_window + 1)
            smoothed[i] = np.mean(data[start:end])

        return smoothed

    def select(self, y: np.ndarray, G_baseline: np.ndarray,
               event_templates: List[Dict], valid_mask: np.ndarray) -> List[int]:
        """Select events using knee/elbow detection."""

        if len(event_templates) == 0:
            return []

        # Convert event templates to list of arrays
        templates = [ev['template'] for ev in event_templates]

        result = self.selector.select(
            y=y,
            baseline_G=G_baseline,
            event_templates=templates,
            valid_mask=valid_mask,
            method=self.method
        )

        # ENHANCEMENT 1: Smooth curves if requested
        if self.smooth_curve:
            result['ssr_path_original'] = result['ssr_path'].copy()
            result['aic_path_original'] = result['aic_path'].copy()
            result['bic_path_original'] = result['bic_path'].copy()

            result['ssr_path'] = self._smooth_curve(result['ssr_path'], self.smooth_window)
            result['aic_path'] = self._smooth_curve(result['aic_path'], self.smooth_window)
            result['bic_path'] = self._smooth_curve(result['bic_path'], self.smooth_window)

            # Re-run knee detection on smoothed curves
            from src.trajmod.model.model_selection import ModelSelector
            if self.method == 'min':
                if self.selector.criterion == 'ssr':
                    k_opt = int(np.argmin(result['ssr_path']))
                elif self.selector.criterion == 'aic':
                    k_opt = int(np.argmin(result['aic_path']))
                else:  # bic
                    k_opt = int(np.argmin(result['bic_path']))
            else:  # knee
                if self.selector.criterion == 'ssr':
                    k_opt = ModelSelector.find_knee_by_curvature(
                        np.arange(len(result['ssr_path'])), result['ssr_path']
                    )
                elif self.selector.criterion == 'aic':
                    k_opt = ModelSelector.find_knee_by_curvature(
                        np.arange(len(result['aic_path'])), result['aic_path']
                    )
                else:  # bic
                    k_opt = ModelSelector.find_knee_by_curvature(
                        np.arange(len(result['bic_path'])), result['bic_path']
                    )

            # Update result with smoothed selection
            result['n_selected'] = k_opt
            result['selected_indices'] = result['sorted_indices'][:k_opt] if k_opt > 0 else []

        selected_indices = result['selected_indices']

        # ENHANCEMENT 2: Filter by minimum amplitude if specified
        if self.min_amplitude is not None and len(selected_indices) > 0:
            # Fit model with selected events to get their amplitudes
            selected_templates = [templates[i] for i in selected_indices]
            if selected_templates:
                event_matrix = np.column_stack(selected_templates)
                G_full = np.hstack([G_baseline, event_matrix])

                Gsub = G_full[valid_mask, :]
                ysub = y[valid_mask]

                try:
                    coeffs_full, _, _, _ = np.linalg.lstsq(Gsub, ysub, rcond=None)
                    n_baseline = G_baseline.shape[1]
                    event_coeffs = coeffs_full[n_baseline:]

                    # Filter by amplitude
                    amplitude_mask = np.abs(event_coeffs) >= self.min_amplitude
                    filtered_indices = [selected_indices[i] for i, keep in enumerate(amplitude_mask) if keep]

                    n_filtered = len(selected_indices) - len(filtered_indices)
                    if n_filtered > 0:
                        logger.info(f"Filtered out {n_filtered} events with |amplitude| < {self.min_amplitude} mm")

                    selected_indices = filtered_indices
                    result['selected_indices'] = selected_indices
                    result['n_selected'] = len(selected_indices)
                    result['amplitude_filtered'] = True
                    result['min_amplitude'] = self.min_amplitude

                except np.linalg.LinAlgError:
                    logger.warning("Failed to compute amplitudes for filtering")

        # Store for plotting
        self.last_selection_result = result

        logger.info(f"Knee selector ({self.selector.criterion}/{self.method}, "
                    f"smooth={self.smooth_curve}, min_amp={self.min_amplitude}) "
                    f"selected {len(selected_indices)}/{len(event_templates)} events")

        return selected_indices

    def plot_selection(self, save_path: Optional[str] = None,
                       figsize: Tuple[int, int] = (12, 6),
                       show_original: bool = True):
        """Plot the selection diagnostics (SSR/AIC/BIC paths).

        Args:
            save_path: Path to save figure
            figsize: Figure size
            show_original: If curves were smoothed, also show original curves

        Returns:
            Figure and axes objects
        """
        if self.last_selection_result is None:
            raise ValueError("Must call select() before plotting")

        import matplotlib.pyplot as plt

        result = self.last_selection_result
        fig, axes = plt.subplots(1, 2, figsize=figsize)

        k_values = np.arange(len(result['ssr_path']))
        selected_k = result['n_selected']

        # SSR plot
        axes[0].plot(k_values, result['ssr_path'] / result['ssr_path'][0],
                     '-o', linewidth=2, markersize=6, label='SSR (smoothed)' if self.smooth_curve else 'SSR')

        # Show original if smoothed
        if self.smooth_curve and show_original and 'ssr_path_original' in result:
            axes[0].plot(k_values, result['ssr_path_original'] / result['ssr_path_original'][0],
                         '--', alpha=0.5, linewidth=1, color='gray', label='SSR (original)')

        axes[0].axvline(selected_k, color='red', linestyle='--',
                        label=f'Selected k={selected_k}', linewidth=2)
        axes[0].set_xlabel('Number of Events', fontsize=12)
        axes[0].set_ylabel('Normalized SSR', fontsize=12)
        axes[0].set_title('Sum of Squared Residuals', fontsize=13, fontweight='bold')
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)

        # AIC/BIC plot
        axes[1].plot(k_values, result['aic_path'], '-o', label='AIC (smoothed)' if self.smooth_curve else 'AIC',
                     linewidth=2, markersize=6)
        axes[1].plot(k_values, result['bic_path'], '-s', label='BIC (smoothed)' if self.smooth_curve else 'BIC',
                     linewidth=2, markersize=6)

        # Show original if smoothed
        if self.smooth_curve and show_original and 'aic_path_original' in result:
            axes[1].plot(k_values, result['aic_path_original'], '--', alpha=0.5,
                         linewidth=1, color='C0', label='AIC (original)')
            axes[1].plot(k_values, result['bic_path_original'], '--', alpha=0.5,
                         linewidth=1, color='C1', label='BIC (original)')

        axes[1].axvline(selected_k, color='red', linestyle='--',
                        label=f'Selected k={selected_k}', linewidth=2)
        axes[1].set_xlabel('Number of Events', fontsize=12)
        axes[1].set_ylabel('Information Criterion', fontsize=12)

        title = f'Model Selection ({self.selector.criterion.upper()} - {self.method})'
        if self.smooth_curve:
            title += f' [smoothed, w={self.smooth_window}]'
        if self.min_amplitude is not None:
            title += f' [min_amp={self.min_amplitude}]'
        axes[1].set_title(title, fontsize=13, fontweight='bold')
        axes[1].legend(fontsize=9)
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved selection plot to {save_path}")

        return fig, axes


class MarginalScreeningSelector(EventSelector):
    """Marginal screening with p-value thresholding."""

    def __init__(self, alpha: float = 0.05, use_fdr: bool = True):
        """Initialize marginal screening selector.

        Args:
            alpha: Significance level
            use_fdr: Whether to use FDR correction (Benjamini-Hochberg)
        """
        self.alpha = alpha
        self.use_fdr = use_fdr

    def select(self, y: np.ndarray, G_baseline: np.ndarray,
               event_templates: List[Dict], valid_mask: np.ndarray) -> List[int]:
        """Select events using marginal screening."""

        if len(event_templates) == 0:
            return []

        from ..model.model_selection import ModelSelector
        from scipy.special import erfc

        # Remove baseline
        Gsub_baseline = G_baseline[valid_mask, :]
        ysub = y[valid_mask]

        coeffs_baseline, _, _, _ = np.linalg.lstsq(Gsub_baseline, ysub, rcond=None)
        baseline_fit = G_baseline @ coeffs_baseline
        residuals = y - baseline_fit
        residuals_valid = residuals[valid_mask]

        # Estimate noise level
        sigma = ModelSelector.mad_sigma(residuals_valid)

        if sigma == 0:
            logger.warning("Zero residual variance - cannot perform screening")
            return list(range(len(event_templates)))

        # Compute marginal statistics for each template
        n_events = len(event_templates)
        t_values = np.zeros(n_events)
        p_values = np.ones(n_events)

        for i, ev in enumerate(event_templates):
            g = ev['template'][valid_mask]
            norm_sq = np.dot(g, g)

            if norm_sq == 0:
                continue

            # Marginal OLS: amplitude = g^T r / g^T g
            amplitude = np.dot(g, residuals_valid) / norm_sq

            # Standard error: sigma / sqrt(g^T g)
            se = sigma / np.sqrt(norm_sq)

            # T-statistic
            t_values[i] = amplitude / se if se > 0 else 0

            # Two-tailed p-value
            p_values[i] = erfc(np.abs(t_values[i]) / np.sqrt(2.0))

        # Select based on p-values
        if self.use_fdr:
            # Benjamini-Hochberg FDR control
            selected_mask = ModelSelector.benjamini_hochberg(p_values, self.alpha)
        else:
            # Simple thresholding
            selected_mask = p_values < self.alpha

        selected_indices = np.where(selected_mask)[0].tolist()

        logger.info(f"Marginal screening selected {len(selected_indices)}/{n_events} events "
                    f"(alpha={self.alpha}, FDR={self.use_fdr})")

        return selected_indices


class ThresholdEventSelector(EventSelector):
    """Select events by amplitude threshold."""

    def __init__(self, threshold: float = 1.0):
        """Initialize threshold selector.

        Args:
            threshold: Minimum absolute amplitude (mm)
        """
        self.threshold = threshold

    def select(self, y: np.ndarray, G_baseline: np.ndarray,
               event_templates: List[Dict], valid_mask: np.ndarray) -> List[int]:
        """Select events with amplitude above threshold."""

        if len(event_templates) == 0:
            return []

        # Fit full model
        event_matrix = np.column_stack([ev['template'] for ev in event_templates])
        G_full = np.hstack([G_baseline, event_matrix])

        Gsub = G_full[valid_mask, :]
        ysub = y[valid_mask]

        coeffs_full, _, _, _ = np.linalg.lstsq(Gsub, ysub, rcond=None)

        # Extract event coefficients
        n_baseline = G_baseline.shape[1]
        event_coeffs = coeffs_full[n_baseline:]

        # Select by threshold
        selected_mask = np.abs(event_coeffs) >= self.threshold
        selected_indices = np.where(selected_mask)[0].tolist()

        logger.info(f"Threshold selector selected {len(selected_indices)}/{len(event_templates)} events "
                    f"(threshold={self.threshold} mm)")

        return selected_indices


class CompositeEventSelector(EventSelector):
    """Composite selector combining multiple strategies."""

    def __init__(self, selectors: List[EventSelector], mode: str = 'intersection'):
        """Initialize composite selector.

        Args:
            selectors: List of EventSelector instances
            mode: 'intersection' (AND) or 'union' (OR)
        """
        self.selectors = selectors
        self.mode = mode

    def select(self, y: np.ndarray, G_baseline: np.ndarray,
               event_templates: List[Dict], valid_mask: np.ndarray) -> List[int]:
        """Select events using composite strategy."""

        if len(event_templates) == 0:
            return []

        # Apply all selectors
        selections = []
        for selector in self.selectors:
            selected = selector.select(y, G_baseline, event_templates, valid_mask)
            selections.append(set(selected))

        # Combine
        if self.mode == 'intersection':
            combined = set.intersection(*selections) if selections else set()
        else:  # union
            combined = set.union(*selections) if selections else set()

        selected_indices = sorted(list(combined))

        logger.info(f"Composite selector ({self.mode}) selected {len(selected_indices)} events")

        return selected_indices
