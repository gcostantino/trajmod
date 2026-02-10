"""Fitting strategies for trajectory models."""
import logging
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any

import numpy as np
from sklearn.linear_model import LassoCV, ElasticNetCV
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class FittingStrategy(ABC):
    @abstractmethod
    def fit(self, G: np.ndarray, y: np.ndarray, weights: Optional[np.ndarray] = None) -> Dict[str, Any]:
        pass


class OLSFitter(FittingStrategy):
    def fit(self, G: np.ndarray, y: np.ndarray, weights: Optional[np.ndarray] = None) -> Dict[str, Any]:
        mask = ~np.isnan(y)
        Gsub = G[mask, :]
        ysub = y[mask]

        if weights is None:
            coeffs, residuals, rank, s = np.linalg.lstsq(Gsub, ysub, rcond=None)
        else:
            w = weights[mask]
            coeffs, residuals, rank, s = np.linalg.lstsq(Gsub * w[:, None], ysub * w, rcond=None)

        return {'coeffs': coeffs, 'residuals': y - G @ coeffs, 'fitted': G @ coeffs, 'rank': rank}


class LassoFitter(FittingStrategy):
    def __init__(self, cv: int = 5, positive: bool = False):
        self.cv = cv
        self.positive = positive

    def fit(self, G: np.ndarray, y: np.ndarray, weights: Optional[np.ndarray] = None) -> Dict[str, Any]:
        mask = ~np.isnan(y)
        scaler = StandardScaler()
        Gsub_scaled = scaler.fit_transform(G[mask, :])

        lasso = LassoCV(cv=self.cv, n_jobs=-1, positive=self.positive)
        lasso.fit(Gsub_scaled, y[mask])

        coeffs = lasso.coef_ / np.where(scaler.scale_ > 1e-10, scaler.scale_, 1.0)
        fitted = G @ coeffs
        return {'coeffs': coeffs, 'residuals': y - fitted, 'fitted': fitted,
                'selected_mask': np.abs(coeffs) > 1e-8, 'alpha': lasso.alpha_,
                'n_selected': np.sum(np.abs(coeffs) > 1e-8)}


class ElasticNetFitter(FittingStrategy):
    def __init__(self, cv: int = 5, l1_ratio: float = 0.5):
        self.cv = cv
        self.l1_ratio = l1_ratio

    def fit(self, G: np.ndarray, y: np.ndarray, weights: Optional[np.ndarray] = None) -> Dict[str, Any]:
        mask = ~np.isnan(y)
        scaler = StandardScaler()
        Gsub_scaled = scaler.fit_transform(G[mask, :])

        enet = ElasticNetCV(cv=self.cv, l1_ratio=self.l1_ratio, n_jobs=-1)
        enet.fit(Gsub_scaled, y[mask])

        coeffs = enet.coef_ / np.where(scaler.scale_ > 1e-10, scaler.scale_, 1.0)
        return {'coeffs': coeffs, 'residuals': y - G @ coeffs, 'fitted': G @ coeffs, 'alpha': enet.alpha_}


class IterativeRefinementFitter(FittingStrategy):
    def __init__(self, max_iterations: int = 10, threshold: float = 3.0):
        self.max_iterations = max_iterations
        self.threshold = threshold
        self.base_fitter = OLSFitter()

    def fit(self, G: np.ndarray, y: np.ndarray, weights: Optional[np.ndarray] = None) -> Dict[str, Any]:
        current_weights = weights.copy() if weights is not None else np.ones(len(y))
        outlier_mask = np.zeros(len(y), dtype=bool)

        for iteration in range(self.max_iterations):
            result = self.base_fitter.fit(G, y, current_weights)
            residuals = result['residuals']
            valid = ~np.isnan(residuals)
            sigma = 1.4826 * np.median(np.abs(residuals[valid] - np.median(residuals[valid])))

            if sigma == 0:
                break

            new_outliers = np.abs(residuals) / sigma > self.threshold
            if not np.any(new_outliers[~outlier_mask]):
                break

            outlier_mask = new_outliers
            current_weights[outlier_mask] = 0.0

        result['outlier_mask'] = outlier_mask
        result['n_iterations'] = iteration + 1
        return result
