"""Model validation and diagnostics tools.

This package provides utilities for validating CLV models, including:
- Model performance metrics (MAE, MAPE, RMSE, ARPE, R²)
- Temporal train/test splitting for time-series validation
- Time-series cross-validation with expanding windows
- MCMC convergence diagnostics
- Posterior predictive checks
- Trace plot visualization
"""

from customer_base_audit.validation.diagnostics import (
    check_mcmc_convergence,
    plot_trace_diagnostics,
    posterior_predictive_check,
)
from customer_base_audit.validation.validation import (
    ValidationMetrics,
    calculate_clv_metrics,
    cross_validate_clv,
    temporal_train_test_split,
)

__all__ = [
    # Diagnostics
    "check_mcmc_convergence",
    "posterior_predictive_check",
    "plot_trace_diagnostics",
    # Validation
    "ValidationMetrics",
    "temporal_train_test_split",
    "calculate_clv_metrics",
    "cross_validate_clv",
]
