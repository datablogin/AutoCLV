"""Model validation and diagnostics tools.

This package provides utilities for validating CLV models, including:
- MCMC convergence diagnostics
- Posterior predictive checks
- Trace plot visualization
"""

from customer_base_audit.validation.diagnostics import (
    check_mcmc_convergence,
    plot_trace_diagnostics,
    posterior_predictive_check,
)

__all__ = [
    "check_mcmc_convergence",
    "posterior_predictive_check",
    "plot_trace_diagnostics",
]
