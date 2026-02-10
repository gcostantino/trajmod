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
            w = weights[valid]
            wrms = np.sqrt(np.sum(w * residuals[valid] ** 2) / np.sum(w))
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
        return f"""
Model Fit Summary
================
RMS: {self.rms:.3f} mm
WRMS: {self.wrms:.3f} mm
Observations: {self.n_observations}
Parameters: {self.n_parameters}
Variance Reduction: {(1 - self.rms ** 2 / np.var(self.fitted)) * 100:.1f}%
        """.strip()
