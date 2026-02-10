"""Uncertainty quantification for trajectory models."""
import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class UncertaintyResults:
    """Container for uncertainty quantification results."""

    covariance_matrix: np.ndarray
    standard_errors: np.ndarray
    confidence_intervals_95: np.ndarray  # Shape (n_params, 2)
    correlation_matrix: np.ndarray
    variance_inflation_factors: Optional[np.ndarray] = None


class UncertaintyQuantifier:
    """Compute uncertainty estimates for fitted models."""

    @staticmethod
    def compute_covariance(G: np.ndarray, residuals: np.ndarray,
                           valid_mask: np.ndarray,
                           use_hc: bool = False) -> np.ndarray:
        """Compute covariance matrix for OLS estimates.

        Args:
            G: Design matrix (N x P)
            residuals: Residuals (N,)
            valid_mask: Valid observation mask
            use_hc: Whether to use heteroskedasticity-consistent (HC) estimator

        Returns:
            Covariance matrix (P x P)
        """
        Gsub = G[valid_mask, :]
        residuals_sub = residuals[valid_mask]

        n = Gsub.shape[0]
        p = Gsub.shape[1]
        dof = max(1, n - p)

        # Compute (G^T G)^{-1}
        GTG = Gsub.T @ Gsub

        try:
            GTG_inv = np.linalg.inv(GTG)
        except np.linalg.LinAlgError:
            logger.warning("Singular GTG matrix, using pseudo-inverse")
            GTG_inv = np.linalg.pinv(GTG)

        if use_hc:
            # Heteroskedasticity-consistent (White/Huber) covariance
            # Cov = (G^T G)^{-1} G^T diag(e^2) G (G^T G)^{-1}
            diag_e2 = np.diag(residuals_sub ** 2)
            middle = Gsub.T @ diag_e2 @ Gsub
            cov_matrix = GTG_inv @ middle @ GTG_inv
        else:
            # Standard OLS covariance
            # Cov = sigma^2 * (G^T G)^{-1}
            sigma2 = np.sum(residuals_sub ** 2) / dof
            cov_matrix = sigma2 * GTG_inv

        return cov_matrix

    @staticmethod
    def compute_standard_errors(cov_matrix: np.ndarray) -> np.ndarray:
        """Compute standard errors from covariance matrix.

        Args:
            cov_matrix: Covariance matrix

        Returns:
            Array of standard errors
        """
        return np.sqrt(np.maximum(np.diag(cov_matrix), 0))

    @staticmethod
    def compute_correlation_matrix(cov_matrix: np.ndarray) -> np.ndarray:
        """Compute correlation matrix from covariance matrix.

        Args:
            cov_matrix: Covariance matrix

        Returns:
            Correlation matrix
        """
        std_devs = np.sqrt(np.diag(cov_matrix))
        std_devs[std_devs == 0] = 1  # Avoid division by zero

        corr_matrix = cov_matrix / np.outer(std_devs, std_devs)

        # Ensure diagonal is exactly 1
        np.fill_diagonal(corr_matrix, 1.0)

        return corr_matrix

    @staticmethod
    def compute_vif(G: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
        """Compute Variance Inflation Factors for multicollinearity detection.

        VIF_j = 1 / (1 - R_j^2) where R_j^2 is the R-squared from regressing
        column j on all other columns.

        Args:
            G: Design matrix
            valid_mask: Valid observation mask

        Returns:
            Array of VIF values
        """
        Gsub = G[valid_mask, :]
        n, p = Gsub.shape

        if p <= 1:
            return np.ones(p)

        vif = np.zeros(p)

        for j in range(p):
            # Regress column j on all other columns
            y_j = Gsub[:, j]
            X_j = np.delete(Gsub, j, axis=1)

            try:
                # Fit regression
                coeffs, _, _, _ = np.linalg.lstsq(X_j, y_j, rcond=None)
                y_pred = X_j @ coeffs

                # Compute R-squared
                ss_tot = np.sum((y_j - np.mean(y_j)) ** 2)
                ss_res = np.sum((y_j - y_pred) ** 2)

                if ss_tot > 0:
                    r_squared = 1 - (ss_res / ss_tot)
                    # VIF = 1 / (1 - R^2)
                    vif[j] = 1.0 / max(1 - r_squared, 1e-10)
                else:
                    vif[j] = np.inf

            except np.linalg.LinAlgError:
                vif[j] = np.inf

        return vif

    @classmethod
    def quantify(cls, G: np.ndarray, residuals: np.ndarray,
                 valid_mask: np.ndarray, coeffs: np.ndarray,
                 use_hc: bool = False, compute_vif: bool = True) -> UncertaintyResults:
        """Complete uncertainty quantification.

        Args:
            G: Design matrix
            residuals: Residuals
            valid_mask: Valid observation mask
            coeffs: Fitted coefficients
            use_hc: Use heteroskedasticity-consistent estimator
            compute_vif: Compute VIF for multicollinearity detection

        Returns:
            UncertaintyResults object
        """
        from scipy.stats import t as t_dist

        # Covariance matrix
        cov_matrix = cls.compute_covariance(G, residuals, valid_mask, use_hc)

        # Standard errors
        se = cls.compute_standard_errors(cov_matrix)

        # Confidence intervals (95%)
        n = np.sum(valid_mask)
        p = G.shape[1]
        dof = max(1, n - p)
        t_crit = t_dist.ppf(0.975, dof)  # 95% CI

        ci_95 = np.zeros((len(coeffs), 2))
        ci_95[:, 0] = coeffs - t_crit * se
        ci_95[:, 1] = coeffs + t_crit * se

        # Correlation matrix
        corr_matrix = cls.compute_correlation_matrix(cov_matrix)

        # VIF (optional)
        vif = cls.compute_vif(G, valid_mask) if compute_vif else None

        return UncertaintyResults(
            covariance_matrix=cov_matrix,
            standard_errors=se,
            confidence_intervals_95=ci_95,
            correlation_matrix=corr_matrix,
            variance_inflation_factors=vif
        )


class BootstrapUncertainty:
    """Bootstrap-based uncertainty quantification."""

    def __init__(self, n_bootstrap: int = 1000, confidence: float = 0.95):
        """Initialize bootstrap uncertainty quantifier.

        Args:
            n_bootstrap: Number of bootstrap samples
            confidence: Confidence level
        """
        self.n_bootstrap = n_bootstrap
        self.confidence = confidence

    def quantify(self, G: np.ndarray, y: np.ndarray, valid_mask: np.ndarray,
                 fitter_func) -> Dict:
        """Compute bootstrap confidence intervals.

        Args:
            G: Design matrix
            y: Observations
            valid_mask: Valid observation mask
            fitter_func: Function that takes (G, y) and returns coefficients

        Returns:
            Dictionary with bootstrap results
        """
        Gsub = G[valid_mask, :]
        ysub = y[valid_mask]
        n = len(ysub)
        p = G.shape[1]

        # Original fit
        coeffs_original = fitter_func(Gsub, ysub)

        # Bootstrap samples
        coeffs_bootstrap = np.zeros((self.n_bootstrap, p))

        for b in range(self.n_bootstrap):
            # Resample with replacement
            indices = np.random.choice(n, size=n, replace=True)
            Gsub_boot = Gsub[indices, :]
            ysub_boot = ysub[indices]

            try:
                coeffs_bootstrap[b, :] = fitter_func(Gsub_boot, ysub_boot)
            except:
                coeffs_bootstrap[b, :] = np.nan

        # Compute percentile confidence intervals
        alpha = 1 - self.confidence
        lower_percentile = 100 * alpha / 2
        upper_percentile = 100 * (1 - alpha / 2)

        ci_lower = np.nanpercentile(coeffs_bootstrap, lower_percentile, axis=0)
        ci_upper = np.nanpercentile(coeffs_bootstrap, upper_percentile, axis=0)

        # Bootstrap standard errors
        se_bootstrap = np.nanstd(coeffs_bootstrap, axis=0)

        return {
            'coeffs': coeffs_original,
            'coeffs_bootstrap': coeffs_bootstrap,
            'standard_errors': se_bootstrap,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'confidence': self.confidence
        }
