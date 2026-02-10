"""Visualization utilities for trajectory models."""
import logging
from typing import Optional, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


class TrajectoryVisualizer:
    """Comprehensive visualization for trajectory models."""

    @staticmethod
    def plot_fit(model, figsize: Tuple[int, int] = (14, 10),
                 save_path: Optional[str] = None, show_uncertainty: bool = True):
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
        ax2 = fig.add_subplot(gs[2, :])
        ax2.plot(model.t, model.results.residuals, 'k.', alpha=0.5, markersize=2)
        ax2.axhline(0, color='r', linestyle='--', linewidth=1)
        ax2.axhline(3 * model.results.rms, color='orange', linestyle=':',
                    linewidth=1, alpha=0.7, label='±3σ')
        ax2.axhline(-3 * model.results.rms, color='orange', linestyle=':',
                    linewidth=1, alpha=0.7)
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

        return fig

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
        """Plot individual model components.

        Args:
            model: Fitted TrajectoryModel
            components: List of component types to plot ['baseline', 'seasonal', 'sse', 'eq']
            figsize: Figure size
            save_path: Path to save figure
        """
        if model.results is None:
            raise ValueError("Model must be fit before plotting")

        if components is None:
            components = ['baseline', 'seasonal', 'sse', 'eq']

        n_components = len(components)
        fig, axes = plt.subplots(n_components + 1, 1, figsize=figsize, sharex=True)

        if n_components == 1:
            axes = [axes]

        # Plot total fit
        axes[0].plot(model.t, model.y, 'k.', alpha=0.2, markersize=1, label='Data')
        axes[0].plot(model.t, model.results.fitted, 'r-', linewidth=1.5, label='Total Fit')
        axes[0].set_ylabel('Displacement (mm)')
        axes[0].set_title('Total Signal', fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Plot individual components
        for i, comp_type in enumerate(components, 1):
            # Find indices for this component type
            indices = []
            for j, name in enumerate(model.results.template_names):
                if comp_type == 'baseline' and name in ['offset', 'trend', 'acceleration']:
                    indices.append(j)
                elif comp_type == 'seasonal' and ('sin' in name or 'cos' in name or 'envelope' in name):
                    indices.append(j)
                elif comp_type == 'sse' and name.startswith('SSE_'):
                    indices.append(j)
                elif comp_type == 'eq' and name.startswith('EQ_'):
                    indices.append(j)

            if not indices:
                axes[i].text(0.5, 0.5, f'No {comp_type} components',
                             ha='center', va='center', transform=axes[i].transAxes)
                axes[i].set_ylabel(comp_type.capitalize())
                continue

            # Compute component signal
            G_comp = model.G[:, indices]
            coeffs_comp = model.results.coeffs[indices]
            signal_comp = G_comp @ coeffs_comp

            axes[i].plot(model.t, signal_comp, linewidth=1.5)
            axes[i].axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
            axes[i].set_ylabel(f'{comp_type.capitalize()} (mm)')
            axes[i].set_title(f'{comp_type.capitalize()} Component ({len(indices)} terms)',
                              fontweight='bold')
            axes[i].grid(True, alpha=0.3)

        axes[-1].set_xlabel('Time')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

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
        if not hasattr(results.metadata, 'uncertainty'):
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
