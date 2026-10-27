"""Results container for model fits."""
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import numpy as np


@dataclass
class ModelResults:
    """Container for trajectory model fit results."""

    coeffs: np.ndarray
    fitted: np.ndarray
    residuals: np.ndarray
    template_names: List[str]
    rms: float
    wrms: float
    n_observations: int
    n_parameters: int
    metadata: Dict[str, Any]

    @classmethod
    def from_fit(cls, coeffs: np.ndarray, fitted: np.ndarray,
                 residuals: np.ndarray, template_names: List[str],
                 weights: Optional[np.ndarray] = None,
                 **metadata) -> 'ModelResults':
        """Create results from fit output."""
        valid = ~np.isnan(residuals)
        rms = np.sqrt(np.mean(residuals[valid] ** 2))

        if weights is not None:
            w2 = weights[valid] ** 2  # w = weights[valid]
            # wrms = np.sqrt(np.sum(w * residuals[valid] ** 2) / np.sum(w))
            wrms = np.sqrt(np.sum(w2 * residuals[valid] ** 2) / np.sum(w2))
        else:
            wrms = rms

        return cls(
            coeffs=coeffs,
            fitted=fitted,
            residuals=residuals,
            template_names=template_names,
            rms=rms,
            wrms=wrms,
            n_observations=np.sum(valid),
            n_parameters=len(coeffs),
            metadata=metadata
        )

    def get_coefficient(self, name: str) -> float:
        """Get coefficient by template name."""
        idx = self.template_names.index(name)
        return self.coeffs[idx]

    def summary(self) -> str:
        """Generate summary string."""
        valid = ~np.isnan(self.residuals)
        y_obs = (self.fitted + self.residuals)[valid]
        vr = (1 - np.var(self.residuals[valid]) / np.var(y_obs)) * 100

        return f"""
Model Fit Summary
================
RMS: {self.rms:.3f} mm
WRMS: {self.wrms:.3f} mm
Observations: {self.n_observations}
Parameters: {self.n_parameters}
Variance Reduction: {vr:.1f}%
        """.strip()
