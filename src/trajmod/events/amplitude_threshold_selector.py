"""Amplitude-based event selection."""
import logging
from typing import List, Dict, Optional

import numpy as np

from .event_selection import EventSelector

logger = logging.getLogger(__name__)


class AmplitudeThresholdSelector(EventSelector):
    """Select events based on fitted amplitude threshold.

    Fits all events and keeps only those with amplitude above threshold.
    """

    def __init__(self, threshold: float = 2.0, mode: str = 'absolute'):
        """Initialize amplitude threshold selector.

        Args:
            threshold: Threshold value
            mode: Threshold mode:
                - 'absolute': Keep events with |amplitude| > threshold (mm)
                - 'relative': Keep events with |amplitude| > threshold × RMS(data)
                - 'snr': Keep events with |amplitude|/σ > threshold
        """
        if mode not in ['absolute', 'relative', 'snr']:
            raise ValueError(f"mode must be 'absolute', 'relative', or 'snr', got {mode}")

        self.threshold = threshold
        self.mode = mode
        self.last_amplitudes = None
        self.last_threshold_value = None

    def select(self, y: np.ndarray, G_baseline: np.ndarray,
               event_templates: List[Dict], valid_mask: np.ndarray) -> List[int]:
        """Select events with amplitude above threshold."""

        if len(event_templates) == 0:
            return []

        # Fit full model to get event amplitudes
        event_matrix = np.column_stack([ev['template'] for ev in event_templates])
        G_full = np.hstack([G_baseline, event_matrix])

        Gsub = G_full[valid_mask, :]
        ysub = y[valid_mask]

        try:
            coeffs_full, _, _, _ = np.linalg.lstsq(Gsub, ysub, rcond=None)
        except np.linalg.LinAlgError:
            logger.warning("Singular matrix in amplitude selection, returning empty")
            return []

        # Extract event coefficients
        n_baseline = G_baseline.shape[1]
        event_coeffs = coeffs_full[n_baseline:]

        # Store for debugging
        self.last_amplitudes = event_coeffs

        # Compute threshold based on mode
        if self.mode == 'absolute':
            threshold_value = self.threshold
        elif self.mode == 'relative':
            # Threshold relative to data RMS
            data_rms = np.sqrt(np.mean(y[valid_mask] ** 2))
            threshold_value = self.threshold * data_rms
        else:  # snr
            # Estimate noise level using MAD
            from trajmod.model.model_selection import ModelSelector
            # Compute residuals from full fit
            fitted = G_full @ coeffs_full
            residuals = y - fitted
            sigma = ModelSelector.mad_sigma(residuals[valid_mask])
            # SNR = |amplitude| / sigma > threshold
            # So amplitude threshold = threshold * sigma
            threshold_value = self.threshold * sigma

        self.last_threshold_value = threshold_value

        # Select events above threshold
        selected_mask = np.abs(event_coeffs) >= threshold_value
        selected_indices = np.where(selected_mask)[0].tolist()

        logger.info(f"Amplitude selector ({self.mode}, threshold={self.threshold}) "
                    f"selected {len(selected_indices)}/{len(event_templates)} events "
                    f"(effective threshold={threshold_value:.2f} mm)")

        return selected_indices

    def plot_amplitudes(self, event_names: Optional[List[str]] = None,
                        figsize=(10, 6), save_path: Optional[str] = None):
        """Plot event amplitudes with threshold line.

        Args:
            event_names: Optional list of event names for x-axis labels
            figsize: Figure size
            save_path: Path to save plot
        """
        if self.last_amplitudes is None:
            raise ValueError("Must call select() before plotting")

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=figsize)

        n_events = len(self.last_amplitudes)
        x = np.arange(n_events)
        amplitudes = self.last_amplitudes

        # Sort by amplitude for better visualization
        sorted_idx = np.argsort(-np.abs(amplitudes))
        amplitudes_sorted = amplitudes[sorted_idx]

        # Plot bars
        colors = ['steelblue' if abs(a) >= self.last_threshold_value else 'lightgray'
                  for a in amplitudes_sorted]
        bars = ax.bar(x, amplitudes_sorted, color=colors, alpha=0.7, edgecolor='black')

        # Add threshold lines
        ax.axhline(self.last_threshold_value, color='red', linestyle='--',
                   linewidth=2, label=f'Threshold (+{self.last_threshold_value:.2f} mm)')
        ax.axhline(-self.last_threshold_value, color='red', linestyle='--',
                   linewidth=2, label=f'Threshold (−{self.last_threshold_value:.2f} mm)')
        ax.axhline(0, color='black', linewidth=0.8)

        ax.set_xlabel('Event Index (sorted by amplitude)', fontsize=12)
        ax.set_ylabel('Amplitude (mm)', fontsize=12)
        ax.set_title(f'Event Amplitudes - {self.mode.capitalize()} Threshold',
                     fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')

        # Add text showing selection
        n_selected = np.sum(np.abs(amplitudes) >= self.last_threshold_value)
        ax.text(0.98, 0.98, f'{n_selected}/{n_events} events selected',
                transform=ax.transAxes, ha='right', va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                fontsize=11)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved amplitude plot to {save_path}")

        return fig, ax
