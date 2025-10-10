"""Model diagnostics for PyMC-based CLV models.

This module provides diagnostic tools for assessing model quality, convergence,
and predictive performance for Bayesian CLV models fitted with MCMC.

Key diagnostics:
- R-hat convergence statistics for MCMC chains
- Posterior predictive checks comparing observed vs predicted data
- Trace plots for visual convergence inspection
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import arviz as az
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for headless environments
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ConvergenceDiagnostics:
    """MCMC convergence diagnostic results.

    Attributes
    ----------
    converged:
        True if all parameters passed convergence checks (R-hat and ESS)
    max_r_hat:
        Maximum R-hat value across all parameters
    failed_parameters:
        Dict of {parameter_name: r_hat} for parameters that failed convergence
    r_hat_threshold:
        Threshold used for convergence check (default: 1.1)
    min_ess_bulk:
        Minimum bulk effective sample size across all parameters
    min_ess_tail:
        Minimum tail effective sample size across all parameters
    ess_threshold:
        Minimum acceptable ESS value (default: 400)
    summary:
        Full ArviZ summary statistics DataFrame
    """

    converged: bool
    max_r_hat: float
    failed_parameters: Dict[str, float]
    r_hat_threshold: float
    min_ess_bulk: float
    min_ess_tail: float
    ess_threshold: float
    summary: pd.DataFrame


@dataclass(frozen=True)
class PosteriorPredictiveStats:
    """Posterior predictive check statistics.

    Attributes
    ----------
    observed_mean:
        Mean of observed data
    predicted_mean:
        Mean of posterior predictive samples
    observed_std:
        Standard deviation of observed data
    predicted_std:
        Standard deviation of posterior predictive samples
    coverage_95:
        Fraction of observed values within 95% credible interval
    median_abs_error:
        Median absolute difference between observed and predicted
    """

    observed_mean: float
    predicted_mean: float
    observed_std: float
    predicted_std: float
    coverage_95: float
    median_abs_error: float


def check_mcmc_convergence(
    idata: az.InferenceData,
    r_hat_threshold: float = 1.1,
    ess_threshold: float = 400.0,
) -> ConvergenceDiagnostics:
    """Check MCMC convergence using R-hat and effective sample size statistics.

    The R-hat (Gelman-Rubin) statistic measures convergence by comparing
    within-chain and between-chain variance. Values close to 1.0 indicate
    convergence, while values > 1.1 suggest chains have not converged.

    Effective sample size (ESS) measures the number of independent samples
    in the chains. Low ESS indicates high autocorrelation.

    Parameters
    ----------
    idata:
        ArviZ InferenceData object from fitted PyMC model
    r_hat_threshold:
        Maximum acceptable R-hat value (default: 1.1)
    ess_threshold:
        Minimum acceptable ESS value (default: 400)

    Returns
    -------
    ConvergenceDiagnostics
        Convergence diagnostic results including max R-hat, min ESS, and failed parameters

    Notes
    -----
    Interpretation guidelines:
    - R-hat < 1.01: Excellent convergence
    - R-hat 1.01-1.05: Good convergence
    - R-hat 1.05-1.1: Acceptable convergence
    - R-hat > 1.1: Poor convergence, increase draws/chains

    - ESS > 400: Sufficient for most applications
    - ESS 100-400: Marginal, consider more draws
    - ESS < 100: Insufficient, increase draws significantly

    Examples
    --------
    >>> from customer_base_audit.models.bg_nbd import BGNBDModelWrapper, BGNBDConfig
    >>> import pandas as pd
    >>> # Fit MCMC model
    >>> data = pd.DataFrame({
    ...     'customer_id': ['C1', 'C2'],
    ...     'frequency': [2, 5],
    ...     'recency': [30.0, 60.0],
    ...     'T': [90.0, 90.0]
    ... })
    >>> config = BGNBDConfig(method='mcmc', draws=500, tune=500)
    >>> wrapper = BGNBDModelWrapper(config)
    >>> wrapper.fit(data)
    >>> # Check convergence
    >>> from customer_base_audit.validation.diagnostics import check_mcmc_convergence
    >>> diagnostics = check_mcmc_convergence(wrapper.model.idata)
    >>> diagnostics.converged
    True
    >>> diagnostics.max_r_hat < 1.1
    True
    >>> diagnostics.min_ess_bulk > 400
    True
    """
    # Generate summary statistics with ArviZ
    summary = az.summary(idata)

    # Check if r_hat column exists
    if "r_hat" not in summary.columns:
        raise ValueError(
            "R-hat statistics not found in summary. "
            "Ensure model was fitted with MCMC (not MAP)."
        )

    # Find maximum R-hat value, handling NaN values from single-chain MCMC
    r_hat_values = summary["r_hat"].dropna()
    if len(r_hat_values) == 0:
        # All R-hat values are NaN (e.g., single chain)
        max_r_hat = float("inf")
        failed_r_hat = summary.index.tolist()  # All parameters failed
    else:
        max_r_hat = float(r_hat_values.max())
        # Identify parameters that failed R-hat convergence (including NaN as failures)
        failed_r_hat = summary[
            (summary["r_hat"] > r_hat_threshold) | (summary["r_hat"].isna())
        ].index.tolist()

    # Check effective sample sizes if available
    min_ess_bulk = float("inf")
    min_ess_tail = float("inf")
    failed_ess = []

    if "ess_bulk" in summary.columns:
        ess_bulk_values = summary["ess_bulk"].dropna()
        if len(ess_bulk_values) > 0:
            min_ess_bulk = float(ess_bulk_values.min())
            failed_ess.extend(
                summary[
                    (summary["ess_bulk"] < ess_threshold) | (summary["ess_bulk"].isna())
                ].index.tolist()
            )

    if "ess_tail" in summary.columns:
        ess_tail_values = summary["ess_tail"].dropna()
        if len(ess_tail_values) > 0:
            min_ess_tail = float(ess_tail_values.min())
            failed_ess.extend(
                summary[
                    (summary["ess_tail"] < ess_threshold) | (summary["ess_tail"].isna())
                ].index.tolist()
            )

    # Combine all failed parameters
    all_failed = set(failed_r_hat + failed_ess)

    # Create failed_parameters dict with R-hat values
    failed_parameters = {}
    for param in all_failed:
        r_hat_val = summary.loc[param, "r_hat"]
        failed_parameters[param] = (
            float(r_hat_val) if not pd.isna(r_hat_val) else float("nan")
        )

    # Overall convergence status: passed R-hat AND ESS checks
    converged = len(all_failed) == 0

    return ConvergenceDiagnostics(
        converged=converged,
        max_r_hat=max_r_hat,
        failed_parameters=failed_parameters,
        r_hat_threshold=r_hat_threshold,
        min_ess_bulk=min_ess_bulk,
        min_ess_tail=min_ess_tail,
        ess_threshold=ess_threshold,
        summary=summary,
    )


def posterior_predictive_check(
    observed_data: pd.Series, posterior_samples: np.ndarray
) -> PosteriorPredictiveStats:
    """Compare observed data to posterior predictive distribution.

    Posterior predictive checks assess model fit by comparing actual observed
    data to simulated data from the fitted model. Good fit shows similar
    distributions between observed and predicted.

    Parameters
    ----------
    observed_data:
        Series of observed values (e.g., actual purchase frequencies)
    posterior_samples:
        Array of posterior predictive samples from model.sample_posterior_predictive(),
        shape (n_samples, n_observations) or (n_observations,) for point predictions

    Returns
    -------
    PosteriorPredictiveStats
        Statistics comparing observed vs predicted distributions

    Warnings
    --------
    This function expects samples from model.sample_posterior_predictive(),
    not simulated/fake data. Using simulated data will not validate model fit.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> # Simulate observed data
    >>> observed = pd.Series([2, 5, 1, 3, 4])
    >>> # Get posterior predictive samples from fitted model
    >>> # predicted = model.sample_posterior_predictive()  # Real usage
    >>> # For demo: simulate what model might produce
    >>> predicted = np.random.normal(3, 1, size=(1000, 5))
    >>> # Run posterior predictive check
    >>> from customer_base_audit.validation.diagnostics import posterior_predictive_check
    >>> stats = posterior_predictive_check(observed, predicted)
    >>> stats.observed_mean
    3.0
    >>> stats.coverage_95  # Fraction within 95% CI
    0.8
    """
    # Validate inputs
    if len(observed_data) == 0:
        raise ValueError("observed_data cannot be empty")

    observed_values = observed_data.values

    # Handle both 1D and 2D posterior samples
    if posterior_samples.ndim == 1:
        # Point predictions - convert to 2D with single sample
        posterior_samples = posterior_samples.reshape(1, -1)

    # Validate dimensions match
    if posterior_samples.shape[-1] != len(observed_values):
        raise ValueError(
            f"Shape mismatch: posterior_samples has {posterior_samples.shape[-1]} observations "
            f"but observed_data has {len(observed_values)}"
        )

    # Calculate observed statistics (using unbiased estimator with ddof=1)
    observed_mean = float(np.mean(observed_values))
    observed_std = float(np.std(observed_values, ddof=1))

    # Calculate posterior predictive statistics (using unbiased estimator)
    # Mean across samples for each observation, then mean across observations
    predicted_mean = float(np.mean(posterior_samples))
    predicted_std = float(np.std(posterior_samples, ddof=1))

    # Calculate 95% credible interval coverage
    # For each observation, check if it falls within 95% CI of predictions
    lower_95 = np.percentile(posterior_samples, 2.5, axis=0)
    upper_95 = np.percentile(posterior_samples, 97.5, axis=0)
    within_ci = np.logical_and(observed_values >= lower_95, observed_values <= upper_95)
    coverage_95 = float(np.mean(within_ci))

    # Calculate median absolute error
    # Use median of posterior samples as point prediction
    predicted_median = np.median(posterior_samples, axis=0)
    abs_errors = np.abs(observed_values - predicted_median)
    median_abs_error = float(np.median(abs_errors))

    return PosteriorPredictiveStats(
        observed_mean=observed_mean,
        predicted_mean=predicted_mean,
        observed_std=observed_std,
        predicted_std=predicted_std,
        coverage_95=coverage_95,
        median_abs_error=median_abs_error,
    )


def plot_trace_diagnostics(
    idata: az.InferenceData,
    output_path: str,
    var_names: Optional[list[str]] = None,
    figsize: tuple[int, int] = (12, 8),
) -> None:
    """Generate trace plots for MCMC diagnostic inspection.

    Trace plots visualize the sampling trajectory of MCMC chains, helping
    identify convergence issues like:
    - Poor mixing (chains stuck in different regions)
    - Autocorrelation (slow exploration)
    - Divergences (numerical instability)

    Parameters
    ----------
    idata:
        ArviZ InferenceData object from fitted PyMC model
    output_path:
        Path to save the plot (e.g., 'diagnostics/trace_plot.png')
    var_names:
        List of parameter names to plot. If None, plots all parameters.
    figsize:
        Figure size in inches (width, height)

    Examples
    --------
    >>> from customer_base_audit.models.bg_nbd import BGNBDModelWrapper, BGNBDConfig
    >>> import pandas as pd
    >>> # Fit MCMC model
    >>> data = pd.DataFrame({
    ...     'customer_id': ['C1', 'C2'],
    ...     'frequency': [2, 5],
    ...     'recency': [30.0, 60.0],
    ...     'T': [90.0, 90.0]
    ... })
    >>> config = BGNBDConfig(method='mcmc', draws=500, tune=500)
    >>> wrapper = BGNBDModelWrapper(config)
    >>> wrapper.fit(data)
    >>> # Generate trace plots
    >>> from customer_base_audit.validation.diagnostics import plot_trace_diagnostics
    >>> plot_trace_diagnostics(wrapper.model.idata, 'trace_plot.png')
    """
    # Create output directory if it doesn't exist
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Generate trace plot using ArviZ (it creates its own figure)
    try:
        az.plot_trace(idata, var_names=var_names, figsize=figsize)

        # Add title
        plt.suptitle("MCMC Trace Diagnostics", fontsize=14, y=1.02)

        # Save plot
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    finally:
        # Always close figure to prevent memory leaks
        plt.close()
