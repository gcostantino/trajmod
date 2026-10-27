"""Visualization utilities for trajectory models."""
import logging
from typing import Optional, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


class TrajectoryVisualizer:
    """Comprehensive visualization for trajectory models."""

    '''@staticmethod
    def plot_fit(model, figsize: Tuple[int, int] = (14, 10),
                 save_path: Optional[str] = None, show_uncertainty: bool = True,
                 show_residuals: bool = True):
        """Plot comprehensive model fit diagnostics.

        Args:
            model: Fitted TrajectoryModel instance
            figsize: Figure size
            save_path: Path to save figure
            show_uncertainty: Whether to show uncertainty bands
        """
        if model.results is None:
            raise ValueError("Model must be fit before plotting")

        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(4, 2, hspace=0.3, wspace=0.3)

        # Main plot: Data + Fit
        ax1 = fig.add_subplot(gs[0:2, :])
        valid = ~np.isnan(model.y)
        ax1.plot(model.t[valid], model.y[valid], 'k.', alpha=0.3,
                 label='Data', markersize=2)
        ax1.plot(model.t, model.results.fitted, 'r-',
                 label='Fit', linewidth=1.5)

        if show_uncertainty and hasattr(model.results.metadata, 'uncertainty'):
            unc = model.results.metadata['uncertainty']
            if unc is not None and hasattr(unc, 'standard_errors'):
                # Plot prediction bands (approximate)
                pass  # Would need prediction variance

        ax1.set_ylabel('Displacement (mm)', fontsize=12)
        ax1.set_title(f'Trajectory Fit - RMS: {model.results.rms:.2f} mm, '
                      f'WRMS: {model.results.wrms:.2f} mm', fontsize=14, fontweight='bold')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)

        # Residuals time series
        if show_residuals:
            ax2 = fig.add_subplot(gs[2, :])
            ax2.plot(model.t, model.results.residuals, 'k.', alpha=0.5, markersize=2)
            ax2.axhline(0, color='r', linestyle='--', linewidth=1)
            ax2.axhline(3 * model.results.rms, color='orange', linestyle=':',
                        linewidth=1, alpha=0.7, label='±3σ')
            ax2.axhline(-3 * model.results.rms, color='orange', linestyle=':',
                        linewidth=1, alpha=0.7)
            # add detected events
            for eq in model.selected_eq:
                ax2.axvline(eq['eq_day'], color='red', alpha=0.3, linestyle='--')
            for sse in model.selected_sse:
                ax2.axvspan(sse['start'], sse['end'], alpha=0.2, color='blue')

            ax2.set_xlabel('Time', fontsize=12)
            ax2.set_ylabel('Residuals (mm)', fontsize=12)
            ax2.legend()
            ax2.grid(True, alpha=0.3)

        # Residual histogram
        ax3 = fig.add_subplot(gs[3, 0])
        residuals_valid = model.results.residuals[~np.isnan(model.results.residuals)]
        ax3.hist(residuals_valid, bins=50, density=True, alpha=0.7,
                 color='steelblue', edgecolor='black')

        # Overlay normal distribution
        mu, sigma = 0, model.results.rms
        x = np.linspace(residuals_valid.min(), residuals_valid.max(), 100)
        ax3.plot(x, 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mu) / sigma) ** 2),
                 'r-', linewidth=2, label='N(0, σ²)')
        ax3.set_xlabel('Residuals (mm)', fontsize=10)
        ax3.set_ylabel('Density', fontsize=10)
        ax3.set_title('Residual Distribution', fontsize=11)
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Q-Q plot
        ax4 = fig.add_subplot(gs[3, 1])
        from scipy.stats import probplot
        probplot(residuals_valid, dist="norm", plot=ax4)
        ax4.set_title('Q-Q Plot', fontsize=11)
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved fit plot to {save_path}")

        return fig'''

    '''@staticmethod
    def plot_fit(model, figsize: Tuple[int, int] = (14, 12),
                 save_path: Optional[str] = None, show_uncertainty: bool = True,
                 show_residuals: bool = True, label_position:Optional[str]='parallel'):
        """Plot comprehensive model fit diagnostics with event markers."""
        if model.results is None:
            raise ValueError("Model must be fit before plotting")

        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(5, 2, hspace=0.35, wspace=0.3)

        # ========================================================================
        # Row 0-1: Main plot (Data + Fit) WITH EVENT MARKERS
        # ========================================================================
        ax1 = fig.add_subplot(gs[0:2, :])
        valid = ~np.isnan(model.y)
        ax1.plot(model.t[valid], model.y[valid], 'k.', alpha=0.3,
                 label='Data', markersize=2)
        ax1.plot(model.t, model.results.fitted, 'r-',
                 label='Fit', linewidth=1.5)

        # Add event markers
        TrajectoryVisualizer._add_event_markers_to_axis(ax1, model, show_labels=True, label_position=label_position)

        ax1.set_ylabel('Displacement (mm)', fontsize=12)
        ax1.set_title(f'Trajectory Fit - RMS: {model.results.rms:.2f} mm, '
                      f'WRMS: {model.results.wrms:.2f} mm',
                      fontsize=14, fontweight='bold')
        ax1.legend(loc='best', fontsize=9)
        ax1.grid(True, alpha=0.3)

        # ========================================================================
        # Row 2: Residuals time series WITH EVENT MARKERS
        # ========================================================================
        ax2 = fig.add_subplot(gs[2, :])
        ax2.plot(model.t, model.results.residuals, 'k.', alpha=0.5, markersize=2)
        ax2.axhline(0, color='r', linestyle='--', linewidth=1)
        ax2.axhline(3 * model.results.rms, color='orange', linestyle=':',
                    linewidth=1, alpha=0.7, label='±3σ')
        ax2.axhline(-3 * model.results.rms, color='orange', linestyle=':',
                    linewidth=1, alpha=0.7)

        # Add event markers (no labels in residuals to avoid clutter)
        TrajectoryVisualizer._add_event_markers_to_axis(ax2, model, show_labels=False, label_position=None)

        ax2.set_ylabel('Residuals (mm)', fontsize=11)
        ax2.set_title('Residuals vs Time', fontsize=11, fontweight='bold')
        ax2.legend(loc='upper right', fontsize=9)
        ax2.grid(True, alpha=0.3)

        # ========================================================================
        # Row 3: Autocorrelation Function (ACF)
        # ========================================================================
        ax3 = fig.add_subplot(gs[3, :])
        residuals_clean = model.results.residuals[~np.isnan(model.results.residuals)]

        max_lag = min(50, len(residuals_clean) // 4)
        lags = range(max_lag + 1)
        acf = []

        for lag in lags:
            if lag == 0:
                acf.append(1.0)
            else:
                r1 = residuals_clean[:-lag]
                r2 = residuals_clean[lag:]
                if len(r1) > 0:
                    acf.append(np.corrcoef(r1, r2)[0, 1])
                else:
                    acf.append(0.0)

        ax3.stem(lags, acf, basefmt=' ', linefmt='steelblue', markerfmt='o')
        ax3.axhline(0, color='k', linestyle='-', linewidth=0.5)

        # 95% confidence interval
        conf = 1.96 / np.sqrt(len(residuals_clean))
        ax3.axhline(conf, color='r', linestyle='--', linewidth=1, alpha=0.7)
        ax3.axhline(-conf, color='r', linestyle='--', linewidth=1, alpha=0.7)
        ax3.fill_between(lags, -conf, conf, alpha=0.2, color='red')

        ax3.set_xlabel('Lag (days)', fontsize=11)
        ax3.set_ylabel('Autocorrelation', fontsize=11)
        ax3.set_title('Residual Autocorrelation (should be within red bands)',
                      fontsize=11, fontweight='bold')
        ax3.grid(alpha=0.3)
        ax3.set_ylim([-1, 1])

        # ========================================================================
        # Row 4: Left - Histogram, Right - Q-Q Plot
        # ========================================================================
        ax4 = fig.add_subplot(gs[4, 0])
        ax4.hist(residuals_clean, bins=50, density=True, alpha=0.7,
                 color='steelblue', edgecolor='black')

        # Overlay normal distribution
        mu, sigma = 0, model.results.rms
        x = np.linspace(residuals_clean.min(), residuals_clean.max(), 100)
        ax4.plot(x, 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mu) / sigma) ** 2),
                 'r-', linewidth=2, label='N(0, σ²)')
        ax4.set_xlabel('Residuals (mm)', fontsize=10)
        ax4.set_ylabel('Density', fontsize=10)
        ax4.set_title('Residual Distribution', fontsize=11, fontweight='bold')
        ax4.legend(fontsize=9)
        ax4.grid(True, alpha=0.3)

        # Q-Q plot
        ax5 = fig.add_subplot(gs[4, 1])
        from scipy.stats import probplot
        probplot(residuals_clean, dist="norm", plot=ax5)
        ax5.set_title('Q-Q Plot (Normality Check)', fontsize=11, fontweight='bold')
        ax5.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved fit plot to {save_path}")

        return fig'''

    @staticmethod
    def plot_fit(model, figsize: Tuple[int, int] = (14, 12),
                 save_path: Optional[str] = None, show_uncertainty: bool = True,
                 show_residuals: bool = True, label_position: Optional[str] = 'parallel'):
        """Plot comprehensive model fit diagnostics with event markers."""
        if model.results is None:
            raise ValueError("Model must be fit before plotting")

        # Use ALL data (no filtering)
        fig = TrajectoryVisualizer._plot_fit_core(
            model=model,
            t_mask=None,  # None = use all data
            window_title=None,
            figsize=figsize,
            label_position=label_position
        )

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved fit plot to {save_path}")

        return fig

    @staticmethod
    def plot_fit_windowed(model, window_size_years: float = 2.0,
                          overlap_years: float = 0.0,
                          figsize: Tuple[int, int] = (14, 12),
                          save_dir: Optional[str] = None,
                          label_position: Optional[str] = 'parallel'):
        """Plot comprehensive diagnostics for sequential time windows.

        Creates the EXACT same plot as plot_fit(), but for sliding windows.

        Args:
            model: Fitted TrajectoryModel instance
            window_size_years: Window size in years (default: 2.0)
            overlap_years: Overlap between windows in years (default: 0.0)
            figsize: Figure size for each window plot
            save_dir: Directory to save window plots (required)
            label_position: 'top' or 'parallel' for event labels

        Returns:
            List of figure objects (one per window)
        """
        if model.results is None:
            raise ValueError("Model must be fit before plotting")

        if save_dir is None:
            raise ValueError("save_dir is required for windowed plots")

        import os
        from datetime import timedelta

        os.makedirs(save_dir, exist_ok=True)

        # Convert years to timedelta
        window_size = timedelta(days=int(window_size_years * 365.25))
        overlap = timedelta(days=int(overlap_years * 365.25))
        step_size = window_size - overlap

        # Calculate windows
        t_start = model.t[0]
        t_end = model.t[-1]

        windows = []
        current_start = t_start

        while current_start < t_end:
            current_end = current_start + window_size
            if current_end > t_end:
                current_end = t_end

            windows.append((current_start, current_end))

            current_start += step_size

            # Stop if window is too small
            if (t_end - current_start).days < 180:  # Less than 6 months
                break

        logger.info(f"Creating {len(windows)} window plots "
                    f"({window_size_years}-year windows)")

        figures = []

        # Plot each window
        for i, (win_start, win_end) in enumerate(windows):
            logger.info(f"Plotting window {i + 1}/{len(windows)}: "
                        f"{win_start.strftime('%Y-%m-%d')} to {win_end.strftime('%Y-%m-%d')}")

            # Create mask for this window
            t_mask = (model.t >= win_start) & (model.t <= win_end)

            if not np.any(t_mask):
                logger.warning(f"  Skipping window {i + 1}: no data")
                continue

            # Create window title
            window_title = (f'Window {i + 1}/{len(windows)}: '
                            f'{win_start.strftime("%Y-%m-%d")} to {win_end.strftime("%Y-%m-%d")}')

            # Use SAME core plotting function
            fig = TrajectoryVisualizer._plot_fit_core(
                model=model,
                t_mask=t_mask,
                window_title=window_title,
                figsize=figsize,
                label_position=label_position
            )

            # Save
            filename = (f"window_{i + 1:02d}_"
                        f"{win_start.strftime('%Y%m%d')}_"
                        f"{win_end.strftime('%Y%m%d')}.png")
            save_path = os.path.join(save_dir, filename)

            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"  Saved: {save_path}")

            figures.append(fig)
            plt.close(fig)  # Close to save memory

        logger.info(f"Completed {len(windows)} windowed plots in {save_dir}/")

        return figures

    @staticmethod
    def _plot_fit_core(model, t_mask=None, window_title=None,
                       figsize=(14, 12), label_position='parallel'):
        """Core plotting logic used by both plot_fit() and plot_fit_windowed().

        This is the SINGLE source of truth for the plot layout and styling.

        Args:
            model: TrajectoryModel instance
            t_mask: Boolean mask for time filtering (None = use all data)
            window_title: Custom title (None = use default)
            figsize: Figure size
            label_position: Event label position

        Returns:
            Matplotlib figure
        """
        from scipy.stats import probplot

        # ========================================================================
        # Apply time mask (or use all data)
        # ========================================================================
        if t_mask is None:
            # Use all data
            t_mask = np.ones(len(model.t), dtype=bool)
            t_window_start = None
            t_window_end = None
        else:
            # Get window bounds for xlim
            t_window_start = model.t[t_mask][0]
            t_window_end = model.t[t_mask][-1]

        # Filter data
        t_plot = model.t[t_mask]
        y_plot = model.y[t_mask]
        fitted_plot = model.results.fitted[t_mask]
        residuals_plot = model.results.residuals[t_mask]

        uncertainties_plot = model.uncertainties[t_mask]  # ← Use stored uncertainties
        weights_plot = 1.0 / uncertainties_plot ** 2
        weights_plot[~np.isfinite(weights_plot)] = 0.0  # Handle inf/nan

        # Compute statistics for this window
        residuals_clean = residuals_plot[~np.isnan(residuals_plot)]
        rms_window = np.sqrt(np.nanmean(residuals_plot ** 2))

        # WRMS
        valid_weights = (weights_plot > 0) & np.isfinite(residuals_plot)
        if np.any(valid_weights):
            wrms_window = np.sqrt(
                np.sum(weights_plot[valid_weights] * residuals_plot[valid_weights] ** 2) /
                np.sum(weights_plot[valid_weights])
            )
        else:
            wrms_window = rms_window

        # Filter events to window
        if t_window_start is not None:
            eq_plot = [eq for eq in model.selected_eq
                       if t_window_start <= eq['eq_day'] <= t_window_end]
            sse_plot = [sse for sse in model.selected_sse
                        if not (sse['end'] < t_window_start or sse['start'] > t_window_end)]
        else:
            eq_plot = model.selected_eq
            sse_plot = model.selected_sse

        # ========================================================================
        # Create figure (SINGLE SOURCE OF TRUTH for layout)
        # ========================================================================
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(5, 2, hspace=0.35, wspace=0.3)

        # ========================================================================
        # Row 0-1: Main plot (Data + Fit) WITH EVENT MARKERS
        # ========================================================================
        ax1 = fig.add_subplot(gs[0:2, :])
        valid = ~np.isnan(y_plot)
        ax1.plot(t_plot[valid], y_plot[valid], 'k.', alpha=0.3,
                 label='Data', markersize=2)
        ax1.plot(t_plot, fitted_plot, 'r-',
                 label='Fit', linewidth=1.5)

        # Add event markers
        TrajectoryVisualizer._add_event_markers_to_axis(
            ax1, model, eq_plot, sse_plot,
            show_labels=True, label_position=label_position
        )

        ax1.set_ylabel('Displacement (mm)', fontsize=12)

        # Title: custom or default
        if window_title:
            title = f'{window_title}\nRMS: {rms_window:.2f} mm, WRMS: {wrms_window:.2f} mm'
        else:
            title = f'Trajectory Fit - RMS: {rms_window:.2f} mm, WRMS: {wrms_window:.2f} mm'
        ax1.set_title(title, fontsize=13, fontweight='bold')

        ax1.legend(loc='best', fontsize=9)
        ax1.grid(True, alpha=0.3)

        # Set xlim if windowed
        if t_window_start is not None:
            ax1.set_xlim(t_window_start, t_window_end)

        # ========================================================================
        # Row 2: Residuals time series WITH EVENT MARKERS
        # ========================================================================
        ax2 = fig.add_subplot(gs[2, :], sharex=ax1)
        ax2.plot(t_plot, residuals_plot, 'k.', alpha=0.5, markersize=2)
        ax2.axhline(0, color='r', linestyle='--', linewidth=1)
        ax2.axhline(3 * rms_window, color='orange', linestyle=':',
                    linewidth=1, alpha=0.7, label='±3σ')
        ax2.axhline(-3 * rms_window, color='orange', linestyle=':',
                    linewidth=1, alpha=0.7)

        # Add event markers (no labels)
        TrajectoryVisualizer._add_event_markers_to_axis(
            ax2, model, eq_plot, sse_plot,
            show_labels=False, label_position=None
        )

        ax2.set_ylabel('Residuals (mm)', fontsize=11)
        ax2.set_title('Residuals vs Time', fontsize=11, fontweight='bold')
        ax2.legend(loc='upper right', fontsize=9)
        ax2.grid(True, alpha=0.3)

        # ========================================================================
        # Row 3: Autocorrelation Function (ACF)
        # ========================================================================
        ax3 = fig.add_subplot(gs[3, :])

        if len(residuals_clean) > 10:
            max_lag = min(50, len(residuals_clean) // 4)
            lags = range(max_lag + 1)
            acf = []

            for lag in lags:
                if lag == 0:
                    acf.append(1.0)
                else:
                    r1 = residuals_clean[:-lag]
                    r2 = residuals_clean[lag:]
                    if len(r1) > 0:
                        acf.append(np.corrcoef(r1, r2)[0, 1])
                    else:
                        acf.append(0.0)

            ax3.stem(lags, acf, basefmt=' ', linefmt='steelblue', markerfmt='o')
            ax3.axhline(0, color='k', linestyle='-', linewidth=0.5)

            # 95% confidence interval
            conf = 1.96 / np.sqrt(len(residuals_clean))
            ax3.axhline(conf, color='r', linestyle='--', linewidth=1, alpha=0.7)
            ax3.axhline(-conf, color='r', linestyle='--', linewidth=1, alpha=0.7)
            ax3.fill_between(lags, -conf, conf, alpha=0.2, color='red')

            ax3.set_ylim([-1, 1])
        else:
            ax3.text(0.5, 0.5, 'Insufficient data for ACF',
                     ha='center', va='center', transform=ax3.transAxes, fontsize=11)

        ax3.set_xlabel('Lag (days)', fontsize=11)
        ax3.set_ylabel('Autocorrelation', fontsize=11)
        ax3.set_title('Residual Autocorrelation (should be within red bands)',
                      fontsize=11, fontweight='bold')
        ax3.grid(alpha=0.3)

        # ========================================================================
        # Row 4: Left - Histogram, Right - Q-Q Plot
        # ========================================================================
        ax4 = fig.add_subplot(gs[4, 0])

        if len(residuals_clean) > 10:
            n_bins = min(50, max(10, len(residuals_clean) // 5))
            ax4.hist(residuals_clean, bins=n_bins, density=True, alpha=0.7,
                     color='steelblue', edgecolor='black')

            # Overlay normal distribution
            mu, sigma = 0, rms_window
            x = np.linspace(residuals_clean.min(), residuals_clean.max(), 100)
            ax4.plot(x, 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mu) / sigma) ** 2),
                     'r-', linewidth=2, label='N(0, σ²)')
            ax4.legend(fontsize=9)
        else:
            ax4.text(0.5, 0.5, 'Insufficient data',
                     ha='center', va='center', transform=ax4.transAxes, fontsize=11)

        ax4.set_xlabel('Residuals (mm)', fontsize=10)
        ax4.set_ylabel('Density', fontsize=10)
        ax4.set_title('Residual Distribution', fontsize=11, fontweight='bold')
        ax4.grid(True, alpha=0.3)

        # Q-Q plot
        ax5 = fig.add_subplot(gs[4, 1])

        if len(residuals_clean) > 10:
            probplot(residuals_clean, dist="norm", plot=ax5)
        else:
            ax5.text(0.5, 0.5, 'Insufficient data',
                     ha='center', va='center', transform=ax5.transAxes, fontsize=11)

        ax5.set_title('Q-Q Plot (Normality Check)', fontsize=11, fontweight='bold')
        ax5.grid(True, alpha=0.3)

        plt.tight_layout()

        return fig

    @staticmethod
    def _add_event_markers_to_axis(ax, model, eq_list, sse_list,
                                   show_labels=True, label_position='top'):
        """Add event markers with intelligent styling and labeling.

        SINGLE SOURCE OF TRUTH for event styling.

        Args:
            ax: Matplotlib axis
            model: TrajectoryModel instance
            eq_list: List of earthquakes to plot (filtered or full)
            sse_list: List of SSEs to plot (filtered or full)
            show_labels: Whether to show magnitude labels
            label_position: 'top' (horizontal on top) or 'parallel' (along line)
        """
        ylim = ax.get_ylim()
        y_range = ylim[1] - ylim[0]

        # ========================================================================
        # 1. Identify which earthquakes have postseismic decay
        # ========================================================================
        eq_with_postseismic = set()

        for name in model.results.template_names:
            # Check for postseismic templates: "EQ_0_2011-03-11_log_tau180"
            if '_log_tau' in name.lower() and name.lower().startswith('eq_'):
                # Extract EQ index
                parts = name.split('_')
                if len(parts) >= 2:
                    try:
                        eq_idx = int(parts[1])
                        eq_with_postseismic.add(eq_idx)
                    except ValueError:
                        pass

        # ========================================================================
        # 2. Plot Earthquakes
        # ========================================================================
        for i, eq in enumerate(model.selected_eq):
            # Check if this EQ is in the filtered list
            if eq not in eq_list:
                continue

            eq_time = eq['eq_day']
            mag = eq.get('magnitude', 0)
            event_type = eq.get('event_type', 'earthquake')

            # Determine event type and styling
            if event_type in ['antenna_change', 'antennachange', 'equipment_change',
                              'equipmentchange', 'monument_change', 'monumentchange']:
                # Antenna/Equipment change
                color = 'gray'
                linestyle = ':'
                linewidth = 1.5
                alpha = 0.6
                label_text = 'Antenna'
                label_color = 'gray'

            elif i in eq_with_postseismic:
                # Earthquake WITH postseismic decay
                color = 'darkred'
                linestyle = '--'
                linewidth = 2.0
                alpha = 0.7
                label_text = f'M{mag:.1f}+PS' if mag > 0 else 'EQ+PS'
                label_color = 'darkred'

            else:
                # Regular earthquake (step only)
                color = 'red'
                linestyle = '--'
                linewidth = 1.5
                alpha = 0.5
                label_text = f'M{mag:.1f}' if mag > 0 else 'EQ'
                label_color = 'red'

            # Plot vertical line
            ax.axvline(eq_time, color=color, linestyle=linestyle,
                       linewidth=linewidth, alpha=alpha, zorder=2)

            # Add label
            if show_labels:
                if label_position == 'top':
                    # Horizontal label at top
                    ax.text(eq_time, ylim[1] - 0.02 * y_range,
                            label_text,
                            rotation=0, ha='center', va='top',
                            fontsize=7, color=label_color,
                            fontweight='bold',
                            bbox=dict(boxstyle='round,pad=0.3',
                                      facecolor='white',
                                      edgecolor=label_color,
                                      alpha=0.8,
                                      linewidth=1))

                elif label_position == 'parallel':
                    # Label parallel to line (rotated 90°)
                    ax.text(eq_time, ylim[0] + 0.1 * y_range,
                            label_text,
                            rotation=90, ha='right', va='bottom',
                            fontsize=10, color=label_color,
                            fontweight='bold')

        # ========================================================================
        # 3. Plot SSEs (keep as bands)
        # ========================================================================
        for sse in sse_list:
            start, end = sse['start'], sse['end']
            mag = sse.get('magnitude', 0)

            # Plot band
            ax.axvspan(start, end, alpha=0.15, color='blue', zorder=0)

            # Add label if requested
            if show_labels:
                mid_time = start + (end - start) / 2
                label_text = f'SSE M{mag:.1f}' if mag > 0 else 'SSE'
                if label_position == 'top':
                    ax.text(mid_time, ylim[1] - 0.08 * y_range,
                            label_text,
                            rotation=0, ha='center', va='top',
                            fontsize=10, color='darkblue',
                            fontweight='bold',
                            bbox=dict(boxstyle='round,pad=0.3',
                                      facecolor='white',
                                      edgecolor='blue',
                                      alpha=0.8,
                                      linewidth=1))

                elif label_position == 'parallel':
                    ax.text(mid_time, ylim[0] + 0.1 * y_range,
                            label_text,
                            rotation=90, ha='right', va='bottom',
                            fontsize=10, color='darkblue',
                            fontweight='bold')

    # ============================================================================
    # Helper Function for Event Markers
    # ============================================================================
    '''@staticmethod
    def _add_event_markers_to_axis(ax, model, show_labels=True, label_position='top'):
        """Add event markers with intelligent styling and labeling.

        Args:
            ax: Matplotlib axis
            model: TrajectoryModel instance
            show_labels: Whether to show magnitude labels
            label_position: 'top' (horizontal on top) or 'parallel' (along line)
        """
        ylim = ax.get_ylim()
        y_range = ylim[1] - ylim[0]

        # ========================================================================
        # 1. Identify which earthquakes have postseismic decay
        # ========================================================================
        eq_with_postseismic = set()

        for name in model.results.template_names:
            # Check for postseismic templates: "EQ_0_2011-03-11_log_tau180"
            if '_log_tau' in name.lower() and name.lower().startswith('eq_'):
                # Extract EQ index
                parts = name.split('_')
                if len(parts) >= 2:
                    try:
                        eq_idx = int(parts[1])
                        eq_with_postseismic.add(eq_idx)
                    except ValueError:
                        pass

        # ========================================================================
        # 2. Plot Earthquakes
        # ========================================================================
        for i, eq in enumerate(model.selected_eq):
            eq_time = eq['eq_day']
            mag = eq.get('magnitude', 0)
            event_type = eq.get('event_type', 'earthquake')

            # Determine event type and styling
            if event_type in ['antenna_change', 'antennachange', 'equipment_change',
                              'equipmentchange', 'monument_change', 'monumentchange']:
                # Antenna/Equipment change
                color = 'gray'
                linestyle = ':'
                linewidth = 1.5
                alpha = 0.6
                label_text = 'Antenna'
                label_color = 'gray'

            elif i in eq_with_postseismic:
                # Earthquake WITH postseismic decay
                color = 'darkred'
                linestyle = '--'
                linewidth = 2.0
                alpha = 0.7
                label_text = f'M{mag:.1f}+PS' if mag > 0 else 'EQ+PS'
                label_color = 'darkred'

            else:
                # Regular earthquake (step only)
                color = 'red'
                linestyle = '--'
                linewidth = 1.5
                alpha = 0.5
                label_text = f'M{mag:.1f}' if mag > 0 else 'EQ'
                label_color = 'red'

            # Plot vertical line
            ax.axvline(eq_time, color=color, linestyle=linestyle,
                       linewidth=linewidth, alpha=alpha, zorder=2)

            # Add label
            if show_labels:
                if label_position == 'top':
                    # Horizontal label at top
                    ax.text(eq_time, ylim[1] - 0.02 * y_range,
                            label_text,
                            rotation=0, ha='center', va='top',
                            fontsize=7, color=label_color,
                            fontweight='bold',
                            bbox=dict(boxstyle='round,pad=0.3',
                                      facecolor='white',
                                      edgecolor=label_color,
                                      alpha=0.8,
                                      linewidth=1))

                elif label_position == 'parallel':
                    # Label parallel to line (rotated 90°)
                    ax.text(eq_time, ylim[0] + 0.1 * y_range,
                            label_text,
                            rotation=90, ha='right', va='bottom',
                            fontsize=10, color=label_color,
                            fontweight='bold',
                            #bbox=dict(boxstyle='round,pad=0.3',
                            #          facecolor='white',
                            #          edgecolor=label_color,
                            #          alpha=0.8,
                            #          linewidth=1)
                    )

        # ========================================================================
        # 3. Plot SSEs (keep as bands)
        # ========================================================================
        for sse in model.selected_sse:
            start, end = sse['start'], sse['end']
            mag = sse.get('magnitude', 0)

            # Plot band
            ax.axvspan(start, end, alpha=0.15, color='blue', zorder=0)

            # Add label if requested
            if show_labels:
                mid_time = start + (end - start) / 2
                label_text = f'SSE M{mag:.1f}' if mag > 0 else 'SSE'

                ax.text(mid_time, ylim[1] - 0.08 * y_range,
                        label_text,
                        rotation=0, ha='center', va='top',
                        fontsize=7, color='darkblue',
                        fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3',
                                  facecolor='white',
                                  edgecolor='blue',
                                  alpha=0.8,
                                  linewidth=1))'''

    @staticmethod
    def plot_coefficients(results, top_n: int = 20, figsize: Tuple[int, int] = (10, 8),
                          save_path: Optional[str] = None, show_uncertainty: bool = True):
        """Plot top N coefficients by magnitude.

        Args:
            results: ModelResults object
            top_n: Number of top coefficients to show
            figsize: Figure size
            save_path: Path to save figure
            show_uncertainty: Whether to show error bars
        """
        sorted_idx = np.argsort(np.abs(results.coeffs))[::-1][:top_n]

        fig, ax = plt.subplots(figsize=figsize)
        y_pos = np.arange(len(sorted_idx))

        coeffs_top = results.coeffs[sorted_idx]
        names_top = [results.template_names[i] for i in sorted_idx]

        # Plot bars
        colors = ['steelblue' if c > 0 else 'coral' for c in coeffs_top]
        bars = ax.barh(y_pos, coeffs_top, color=colors, alpha=0.7, edgecolor='black')

        # Add error bars if uncertainty available
        if show_uncertainty and hasattr(results.metadata, 'uncertainty'):
            unc = results.metadata.get('uncertainty')
            if unc is not None and hasattr(unc, 'standard_errors'):
                se_top = unc.standard_errors[sorted_idx]
                ax.errorbar(coeffs_top, y_pos, xerr=1.96 * se_top,
                            fmt='none', ecolor='black', alpha=0.5, capsize=3)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(names_top, fontsize=9)
        ax.set_xlabel('Coefficient Value (mm)', fontsize=12)
        ax.set_title(f'Top {top_n} Model Coefficients', fontsize=14, fontweight='bold')
        ax.axvline(0, color='black', linewidth=0.8)
        ax.grid(True, axis='x', alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig, ax

    @staticmethod
    def plot_components(model, components: List[str] = None,
                        figsize: Tuple[int, int] = (14, 10),
                        save_path: Optional[str] = None):
        """Plot individual model components."""
        if model.results is None:
            raise ValueError("Model must be fit before plotting")

        if components is None:
            components = ['baseline', 'seasonal', 'sse', 'eq']

        n_components = len(components)
        fig, axes = plt.subplots(n_components + 1, 1, figsize=figsize, sharex=True)

        if n_components == 0:
            axes = [axes]

        # Plot total fit
        axes[0].plot(model.t, model.y, 'k.', alpha=0.2, markersize=1, label='Data')
        axes[0].plot(model.t, model.results.fitted, 'r-', linewidth=1.5, label='Total Fit')
        axes[0].set_ylabel('Displacement (mm)', fontsize=11)
        axes[0].set_title('Total Signal', fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Plot individual components
        for i, comp_type in enumerate(components, 1):
            indices = []

            for j, name in enumerate(model.results.template_names):
                name_lower = name.lower()

                if comp_type == 'baseline':
                    # Match: offset, trend, acceleration
                    if any(kw in name_lower for kw in ['offset', 'trend', 'accel']):
                        indices.append(j)

                elif comp_type == 'seasonal':
                    # Match: anything with sin/cos or envelope
                    if any(kw in name_lower for kw in ['sin', 'cos', 'envelope']):
                        indices.append(j)

                elif comp_type == 'sse':
                    # Match: SSE_* (case-insensitive)
                    if name_lower.startswith('sse_'):
                        indices.append(j)

                elif comp_type == 'eq':
                    # Match: EQ_* (case-insensitive)
                    if name_lower.startswith('eq_'):
                        indices.append(j)

            if not indices:
                axes[i].text(0.5, 0.5, f'No {comp_type} components',
                             ha='center', va='center', transform=axes[i].transAxes,
                             fontsize=11, color='gray')
                axes[i].set_ylabel(comp_type.capitalize(), fontsize=11)
                axes[i].grid(True, alpha=0.3)
                continue

            # Compute component signal
            G_comp = model.G[:, indices]
            coeffs_comp = model.results.coeffs[indices]
            signal_comp = G_comp @ coeffs_comp

            # Choose color by component type
            color_map = {
                'baseline': 'steelblue',
                'seasonal': 'green',
                'sse': 'orange',
                'eq': 'purple'
            }

            axes[i].plot(model.t, signal_comp, linewidth=1.5,
                         color=color_map.get(comp_type, 'black'))
            axes[i].axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
            axes[i].set_ylabel(f'{comp_type.capitalize()} (mm)', fontsize=11)
            axes[i].set_title(f'{comp_type.capitalize()} Component ({len(indices)} terms)',
                              fontsize=11, fontweight='bold')
            axes[i].grid(True, alpha=0.3)

        axes[-1].set_xlabel('Time', fontsize=12)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved component plot to {save_path}")

        return fig, axes

    @staticmethod
    def plot_event_selection(ssr_path: np.ndarray, aic_path: np.ndarray,
                             bic_path: np.ndarray, selected_k: int,
                             figsize: Tuple[int, int] = (12, 6),
                             save_path: Optional[str] = None):
        """Plot model selection diagnostics.

        Args:
            ssr_path: SSR values for k=0, 1, 2, ..., n_events
            aic_path: AIC values
            bic_path: BIC values
            selected_k: Selected number of events
            figsize: Figure size
            save_path: Path to save figure
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)

        k_values = np.arange(len(ssr_path))

        # SSR plot
        axes[0].plot(k_values, ssr_path / ssr_path[0], '-o', label='SSR (normalized)')
        axes[0].axvline(selected_k, color='red', linestyle='--',
                        label=f'Selected k={selected_k}', linewidth=2)
        axes[0].set_xlabel('Number of Events', fontsize=12)
        axes[0].set_ylabel('Normalized SSR', fontsize=12)
        axes[0].set_title('Sum of Squared Residuals', fontsize=13, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # AIC/BIC plot
        axes[1].plot(k_values, aic_path, '-o', label='AIC')
        axes[1].plot(k_values, bic_path, '-s', label='BIC')
        axes[1].axvline(selected_k, color='red', linestyle='--',
                        label=f'Selected k={selected_k}', linewidth=2)
        axes[1].set_xlabel('Number of Events', fontsize=12)
        axes[1].set_ylabel('Information Criterion', fontsize=12)
        axes[1].set_title('Model Selection Criteria', fontsize=13, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig, axes

    @staticmethod
    def plot_correlation_matrix(results, threshold: float = 0.7,
                                figsize: Tuple[int, int] = (12, 10),
                                save_path: Optional[str] = None):
        """Plot correlation matrix of coefficients.

        Args:
            results: ModelResults with uncertainty metadata
            threshold: Highlight correlations above this threshold
            figsize: Figure size
            save_path: Path to save figure
        """
        if 'uncertainty' not in results.metadata:
            raise ValueError("Uncertainty information not available")

        unc = results.metadata.get('uncertainty')
        if unc is None or not hasattr(unc, 'correlation_matrix'):
            raise ValueError("Correlation matrix not computed")

        corr_matrix = unc.correlation_matrix

        fig, ax = plt.subplots(figsize=figsize)

        im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')

        # Colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Correlation', fontsize=12)

        # Highlight high correlations
        n = corr_matrix.shape[0]
        for i in range(n):
            for j in range(i + 1, n):
                if np.abs(corr_matrix[i, j]) > threshold:
                    ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                               fill=False, edgecolor='yellow',
                                               linewidth=2))

        ax.set_xlabel('Parameter Index', fontsize=12)
        ax.set_ylabel('Parameter Index', fontsize=12)
        ax.set_title(f'Coefficient Correlation Matrix (|r| > {threshold} highlighted)',
                     fontsize=14, fontweight='bold')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig, ax

    @staticmethod
    def plot_acf(results, figsize: Tuple[int, int] = (10, 4),
                 max_lag=50, save_path: Optional[str] = None):
        """Plot autocorrelation of residuals."""
        from scipy.stats import pearsonr
        residuals = results.residuals
        residuals_clean = residuals[~np.isnan(residuals)]

        acf = []
        lags = range(max_lag + 1)

        for lag in lags:
            if lag == 0:
                acf.append(1.0)
            else:
                r1 = residuals_clean[:-lag]
                r2 = residuals_clean[lag:]
                acf.append(np.corrcoef(r1, r2)[0, 1])

        # Plot
        fig, ax = plt.subplots(figsize=figsize)
        ax.stem(lags, acf, basefmt=' ')
        ax.axhline(0, color='k', linestyle='-', linewidth=0.5)

        # 95% confidence interval
        conf = 1.96 / np.sqrt(len(residuals_clean))
        ax.axhline(conf, color='r', linestyle='--', linewidth=1)
        ax.axhline(-conf, color='r', linestyle='--', linewidth=1)

        ax.set_xlabel('Lag (days)')
        ax.set_ylabel('Autocorrelation')
        ax.set_title('Residual Autocorrelation')
        ax.grid(alpha=0.3)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig, ax

    @staticmethod
    def plot_periodogram(results, sampling_interval=1.0,
                         figsize: Tuple[int, int] = (12, 8),
                         save_path: Optional[str] = None):
        """Plot power spectral density of residuals."""
        from scipy import signal

        residuals = results.residuals
        residuals_clean = residuals[~np.isnan(residuals)]

        if len(residuals_clean) < 10:
            raise ValueError("Need at least 10 valid residuals for periodogram")

        # Compute periodogram
        try:
            frequencies, psd = signal.periodogram(
                residuals_clean,
                fs=1.0 / sampling_interval,
                window='hann',
                scaling='density'
            )
        except Exception as e:
            logger.error(f"Failed to compute periodogram: {e}")
            raise

        # Convert to periods (skip DC component)
        periods = 1.0 / frequencies[1:]
        psd = psd[1:]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)

        # Linear frequency scale
        ax1.semilogy(frequencies[1:], psd, linewidth=1.5, color='steelblue')
        ax1.set_xlabel('Frequency (cycles/day)', fontsize=11)
        ax1.set_ylabel('Power Spectral Density', fontsize=11)
        ax1.set_title('Residual Periodogram (Frequency Domain)',
                      fontsize=12, fontweight='bold')
        ax1.grid(alpha=0.3, which='both')

        # Period scale
        ax2.loglog(periods, psd, linewidth=1.5, color='steelblue')
        ax2.axvline(365.25, color='r', linestyle='--', alpha=0.6,
                    linewidth=2, label='Annual (365.25 d)')
        ax2.axvline(182.625, color='orange', linestyle='--', alpha=0.6,
                    linewidth=2, label='Semi-annual (182.6 d)')
        '''ax2.axvline(91.3, color='green', linestyle='--', alpha=0.5,
                    linewidth=1.5, label='Quarterly (91.3 d)')'''
        ax2.set_xlabel('Period (days)', fontsize=11)
        ax2.set_ylabel('Power Spectral Density', fontsize=11)
        ax2.set_title('Residual Periodogram (Period Domain)',
                      fontsize=12, fontweight='bold')
        ax2.legend(loc='best', fontsize=9)
        ax2.grid(alpha=0.3, which='both')
        ax2.set_xlim([periods.min(), periods.max()])

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved periodogram to {save_path}")

        return fig, (ax1, ax2)

    @staticmethod
    def plot_residuals_vs_fitted(results, figsize: Tuple[int, int] = (10, 6),
                                 save_path: Optional[str] = None):
        """Plot residuals vs fitted values to check heteroscedasticity.

        Args:
            results: ModelResults object
            figsize: Figure size
            save_path: Path to save figure

        Returns:
            fig, ax: Matplotlib figure and axes
        """
        fitted = results.fitted
        residuals = results.residuals

        # Remove NaNs
        valid = ~(np.isnan(fitted) | np.isnan(residuals))
        fitted_clean = fitted[valid]
        residuals_clean = residuals[valid]

        fig, ax = plt.subplots(figsize=figsize)

        # Scatter plot
        ax.scatter(fitted_clean, residuals_clean, alpha=0.5, s=10,
                   color='steelblue', edgecolor='none')
        ax.axhline(0, color='r', linestyle='--', linewidth=1.5)

        # Add smoothed trend line (LOWESS approximation)
        try:
            from scipy.signal import savgol_filter
            idx_sorted = np.argsort(fitted_clean)

            if len(fitted_clean) > 51:  # Need enough points for smoothing
                window = min(51, len(fitted_clean) // 3)
                if window % 2 == 0:  # Must be odd
                    window += 1
                smooth = savgol_filter(residuals_clean[idx_sorted], window, 3)
                ax.plot(fitted_clean[idx_sorted], smooth, 'orange',
                        linewidth=2, label='Smoothed trend', alpha=0.8)
        except Exception as e:
            logger.debug(f"Could not add smoothed trend: {e}")

        # Add ±3σ lines
        sigma = results.rms
        ax.axhline(3 * sigma, color='orange', linestyle=':', linewidth=1, alpha=0.7)
        ax.axhline(-3 * sigma, color='orange', linestyle=':', linewidth=1, alpha=0.7)

        ax.set_xlabel('Fitted values (mm)', fontsize=12)
        ax.set_ylabel('Residuals (mm)', fontsize=12)
        ax.set_title('Residuals vs Fitted Values\n(Should show random scatter around zero)',
                     fontsize=13, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved residuals vs fitted plot to {save_path}")

        return fig, ax
