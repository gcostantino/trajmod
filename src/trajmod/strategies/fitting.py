"""Fitting strategies for trajectory models."""
import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
from sklearn.linear_model import ElasticNetCV, LassoCV

logger = logging.getLogger(__name__)


def _prepare(G: np.ndarray,
             y: np.ndarray,
             weights: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Return (w, mask).

    w: full-length weights (ones if None, zero where not finite).
    mask: observations that contribute (finite y, finite positive weight).
    """
    y = np.asarray(y, dtype=float)
    if weights is None:
        w = np.ones(len(y))
    else:
        w = np.array(weights, dtype=float)
        w[~np.isfinite(w)] = 0.0
    mask = np.isfinite(y) & (w > 0)
    return w, mask


def _result(G: np.ndarray, y: np.ndarray, coeffs: np.ndarray, **extra) -> Dict[str, Any]:
    """Standard result dict. Residuals are UNWEIGHTED (y - G @ coeffs)."""
    fitted = G @ coeffs
    out = {'coeffs': coeffs, 'residuals': y - fitted, 'fitted': fitted}
    out.update(extra)
    return out


class FittingStrategy(ABC):
    @abstractmethod
    def fit(self, G: np.ndarray, y: np.ndarray,
            weights: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Fit y ~ G @ coeffs minimising sum (weights * residual)^2.

        Returns a dict with at least 'coeffs', 'fitted', 'residuals'
        (residuals are unweighted).
        """


class OLSFitter(FittingStrategy):
    """(Weighted) least squares. weights = 1/sigma."""

    def fit(self, G: np.ndarray, y: np.ndarray,
            weights: Optional[np.ndarray] = None) -> Dict[str, Any]:
        w, mask = _prepare(G, y, weights)
        if mask.sum() == 0:
            raise ValueError("No valid observations to fit")

        Gsub = G[mask, :] * w[mask, None]
        ysub = y[mask] * w[mask]
        coeffs, _, rank, _ = np.linalg.lstsq(Gsub, ysub, rcond=None)

        return _result(G, y, coeffs, rank=rank, n_used=int(mask.sum()))


def _penalized_fit(G: np.ndarray,
                   y: np.ndarray,
                   w: np.ndarray,
                   mask: np.ndarray,
                   estimator: Any) -> Tuple[np.ndarray, Any]:
    """Weighted penalised fit that keeps the constant (offset) column unpenalised.

    The offset column is removed by weighted centering, the remaining columns are
    scaled to unit weighted variance, the penalised model is fitted without
    intercept, and the intercept is then put back in the constant column. (The
    old implementation standardised the offset column to zero, so the fitted
    series lost its mean.)
    """
    Gm, ym, wm = G[mask, :], y[mask], w[mask]
    w2 = wm ** 2
    sw = w2.sum()

    spread = np.ptp(Gm, axis=0)
    const = spread <= 1e-12 * np.maximum(1.0, np.abs(Gm).max(axis=0))
    nc = ~const

    xbar = (w2[:, None] * Gm[:, nc]).sum(axis=0) / sw
    ybar = float((w2 * ym).sum() / sw)

    Xc = Gm[:, nc] - xbar
    yc = ym - ybar

    scale = np.sqrt((w2[:, None] * Xc ** 2).sum(axis=0) / sw)
    scale = np.where(scale > 1e-12, scale, 1.0)
    Xs = Xc / scale

    # loss = sum (w * (yc - Xs b))^2
    estimator.fit(Xs * wm[:, None], yc * wm)

    beta = estimator.coef_ / scale
    coeffs = np.zeros(G.shape[1])
    coeffs[nc] = beta
    intercept = ybar - float(xbar @ beta)

    if const.any():
        coeffs[np.flatnonzero(const)[0]] = intercept
    else:
        logger.debug("No constant column in G: intercept of the penalised fit is dropped")

    return coeffs, estimator


class LassoFitter(FittingStrategy):
    """Weighted LassoCV; the offset column is not penalised."""

    def __init__(self, cv: int = 5, positive: bool = False):
        self.cv = cv
        self.positive = positive

    def fit(self, G: np.ndarray, y: np.ndarray,
            weights: Optional[np.ndarray] = None) -> Dict[str, Any]:
        w, mask = _prepare(G, y, weights)
        if mask.sum() <= self.cv:
            raise ValueError("Too few valid observations for cross-validated Lasso")

        lasso = LassoCV(cv=self.cv, n_jobs=-1, positive=self.positive, fit_intercept=False)
        coeffs, lasso = _penalized_fit(G, y, w, mask, lasso)

        selected = np.abs(coeffs) > 1e-8
        return _result(G, y, coeffs,
                       selected_mask=selected, alpha=lasso.alpha_,
                       n_selected=int(selected.sum()), n_used=int(mask.sum()))


class ElasticNetFitter(FittingStrategy):
    """Weighted ElasticNetCV; the offset column is not penalised."""

    def __init__(self, cv: int = 5, l1_ratio: float = 0.5):
        self.cv = cv
        self.l1_ratio = l1_ratio

    def fit(self, G: np.ndarray, y: np.ndarray,
            weights: Optional[np.ndarray] = None) -> Dict[str, Any]:
        w, mask = _prepare(G, y, weights)
        if mask.sum() <= self.cv:
            raise ValueError("Too few valid observations for cross-validated ElasticNet")

        enet = ElasticNetCV(cv=self.cv, l1_ratio=self.l1_ratio, n_jobs=-1, fit_intercept=False)
        coeffs, enet = _penalized_fit(G, y, w, mask, enet)

        selected = np.abs(coeffs) > 1e-8
        return _result(G, y, coeffs,
                       selected_mask=selected, alpha=enet.alpha_,
                       n_selected=int(selected.sum()), n_used=int(mask.sum()))


class IterativeRefinementFitter(FittingStrategy):
    """Iteratively down-weights outliers (robust, MAD-based).

    Outliers are detected on standardized residuals, i.e. weighted residuals
    w * r divided by their robust scatter, so noisy epochs are not flagged just
    because their sigma is large. Flagged points stay excluded (cumulative).
    """

    def __init__(self, max_iterations: int = 10, threshold: float = 3.0):
        self.max_iterations = max_iterations
        self.threshold = threshold
        self.base_fitter = OLSFitter()

    def fit(self, G: np.ndarray, y: np.ndarray,
            weights: Optional[np.ndarray] = None) -> Dict[str, Any]:
        y = np.asarray(y, dtype=float)
        base_w, _ = _prepare(G, y, weights)
        current_weights = base_w.copy()
        outlier_mask = np.zeros(len(y), dtype=bool)

        result: Optional[Dict[str, Any]] = None
        n_iter = 0

        for iteration in range(self.max_iterations):
            n_iter = iteration + 1
            result = self.base_fitter.fit(G, y, current_weights)

            # standardized residuals on the points still in the fit
            wr = current_weights * result['residuals']
            active = np.isfinite(wr) & (current_weights > 0)
            if active.sum() < 3:
                break

            med = np.median(wr[active])
            sigma = 1.4826 * np.median(np.abs(wr[active] - med))
            if sigma <= 0:
                break

            z = np.abs(wr - med) / sigma
            new_outliers = active & (z > self.threshold)
            if not new_outliers.any():
                break

            outlier_mask |= new_outliers
            current_weights[new_outliers] = 0.0

        if result is None:
            result = self.base_fitter.fit(G, y, current_weights)

        result['outlier_mask'] = outlier_mask
        result['n_outliers'] = int(outlier_mask.sum())
        result['n_iterations'] = n_iter
        result['effective_weights'] = current_weights
        return result
