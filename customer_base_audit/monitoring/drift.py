"""Drift detection for monitoring model performance over time.

This module implements statistical tests to detect distribution shifts between
baseline (training) and current (production) data distributions.

Key Metrics:
- PSI (Population Stability Index): Measures overall distribution shift
- KS (Kolmogorov-Smirnov): Non-parametric test for distribution equality

Typical workflow:
1. Calculate baseline statistics from training data
2. Periodically calculate current statistics from production data
3. Compare using PSI and/or KS test
4. Alert if drift exceeds threshold

References:
- Yurdakul, B. (2018). "Statistical Properties of Population Stability Index"
- Massey, F. J. (1951). "The Kolmogorov-Smirnov Test for Goodness of Fit"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DriftResult:
    """Results from drift detection analysis.

    Attributes
    ----------
    metric_name:
        Name of the feature or prediction being monitored
    test_type:
        Type of drift test performed ("psi" or "ks")
    drift_score:
        Numerical drift score (PSI value or KS statistic)
    p_value:
        Statistical significance (only for KS test, None for PSI)
    drift_detected:
        Whether drift exceeds threshold
    threshold:
        Threshold used for detection
    interpretation:
        Human-readable interpretation of the result
    """

    metric_name: str
    test_type: Literal["psi", "ks"]
    drift_score: float
    p_value: float | None
    drift_detected: bool
    threshold: float
    interpretation: str


def calculate_psi(
    baseline: np.ndarray | pd.Series,
    current: np.ndarray | pd.Series,
    bins: int = 10,
    epsilon: float = 1e-10,
) -> float:
    """Calculate Population Stability Index (PSI) between two distributions.

    PSI measures the shift between a baseline (expected) and current (actual)
    distribution. It's commonly used to detect feature drift in production models.

    Formula:
        PSI = Σ [(Actual% - Expected%) * ln(Actual% / Expected%)]

    Interpretation:
        PSI < 0.1: No significant change
        0.1 ≤ PSI < 0.25: Small change, investigation recommended
        PSI ≥ 0.25: Large change, model retraining likely needed

    Parameters
    ----------
    baseline:
        Baseline (training/reference) distribution
    current:
        Current (production/test) distribution
    bins:
        Number of bins for discretization (default: 10)
    epsilon:
        Small value added to avoid log(0) errors (default: 1e-10)

    Returns
    -------
    float
        PSI value (non-negative, typically 0-3)

    Raises
    ------
    ValueError
        If inputs are empty or have insufficient data

    Examples
    --------
    >>> baseline = np.random.normal(0, 1, 1000)
    >>> current_stable = np.random.normal(0, 1, 1000)
    >>> current_shifted = np.random.normal(0.5, 1, 1000)
    >>> calculate_psi(baseline, current_stable)
    0.05  # No significant drift
    >>> calculate_psi(baseline, current_shifted)
    0.35  # Significant drift detected
    """
    if len(baseline) == 0 or len(current) == 0:
        raise ValueError("Baseline and current distributions must not be empty")

    # Convert to numpy arrays if needed
    baseline_arr = np.asarray(baseline)
    current_arr = np.asarray(current)

    # Remove NaN values BEFORE size validation
    # This ensures validation happens on actual usable data
    baseline_arr = baseline_arr[~np.isnan(baseline_arr)]
    current_arr = current_arr[~np.isnan(current_arr)]

    if len(baseline_arr) == 0 or len(current_arr) == 0:
        raise ValueError("Distributions contain only NaN values")

    # Validate sufficient data after NaN removal
    if len(baseline_arr) < bins or len(current_arr) < bins:
        raise ValueError(
            f"Need at least {bins} samples in each distribution after NaN removal "
            f"(got baseline={len(baseline_arr)}, current={len(current_arr)})"
        )

    # Create bins based on baseline distribution percentiles
    # Use quantiles to ensure roughly equal-sized bins
    percentiles = np.linspace(0, 100, bins + 1)
    bin_edges = np.percentile(baseline_arr, percentiles)

    # Ensure unique bin edges (handle constant values)
    bin_edges = np.unique(bin_edges)
    if len(bin_edges) < 2:
        logger.warning(
            "Baseline distribution has constant values. PSI may not be meaningful."
        )
        return 0.0

    # Adjust edges to include all values
    bin_edges[0] = min(bin_edges[0], baseline_arr.min(), current_arr.min()) - epsilon
    bin_edges[-1] = max(bin_edges[-1], baseline_arr.max(), current_arr.max()) + epsilon

    # Calculate frequencies in each bin
    baseline_counts, _ = np.histogram(baseline_arr, bins=bin_edges)
    current_counts, _ = np.histogram(current_arr, bins=bin_edges)

    # Convert to proportions
    baseline_props = baseline_counts / len(baseline_arr)
    current_props = current_counts / len(current_arr)

    # Add epsilon to avoid division by zero
    baseline_props = np.where(baseline_props == 0, epsilon, baseline_props)
    current_props = np.where(current_props == 0, epsilon, current_props)

    # Calculate PSI
    psi = np.sum(
        (current_props - baseline_props) * np.log(current_props / baseline_props)
    )

    return float(psi)


def kolmogorov_smirnov_test(
    baseline: np.ndarray | pd.Series,
    current: np.ndarray | pd.Series,
    alpha: float = 0.05,
) -> tuple[float, float, bool]:
    """Perform two-sample Kolmogorov-Smirnov test for drift detection.

    The KS test measures the maximum distance between two cumulative distribution
    functions (CDFs). It's a non-parametric test that makes no assumptions about
    the underlying distributions.

    Parameters
    ----------
    baseline:
        Baseline (training/reference) distribution
    current:
        Current (production/test) distribution
    alpha:
        Significance level for hypothesis test (default: 0.05)

    Returns
    -------
    statistic: float
        KS statistic (maximum distance between CDFs, range [0, 1])
    p_value: float
        Two-tailed p-value
    drift_detected: bool
        True if p_value < alpha (distributions are significantly different)

    Raises
    ------
    ValueError
        If inputs are empty

    Examples
    --------
    >>> baseline = np.random.normal(0, 1, 1000)
    >>> current_stable = np.random.normal(0, 1, 1000)
    >>> current_shifted = np.random.normal(0.5, 1, 1000)
    >>> _, p_val_stable, drift = kolmogorov_smirnov_test(baseline, current_stable)
    >>> p_val_stable > 0.05  # No drift detected
    True
    >>> _, p_val_shifted, drift = kolmogorov_smirnov_test(baseline, current_shifted)
    >>> drift  # Drift detected
    True
    """
    if len(baseline) == 0 or len(current) == 0:
        raise ValueError("Baseline and current distributions must not be empty")

    # Convert to numpy arrays if needed
    baseline_arr = np.asarray(baseline)
    current_arr = np.asarray(current)

    # Remove NaN values
    baseline_arr = baseline_arr[~np.isnan(baseline_arr)]
    current_arr = current_arr[~np.isnan(current_arr)]

    if len(baseline_arr) == 0 or len(current_arr) == 0:
        raise ValueError("Distributions contain only NaN values")

    # Perform two-sample KS test
    statistic, p_value = stats.ks_2samp(baseline_arr, current_arr)

    drift_detected = p_value < alpha

    return float(statistic), float(p_value), drift_detected


def detect_feature_drift(
    baseline_features: pd.DataFrame,
    current_features: pd.DataFrame,
    method: Literal["psi", "ks", "both"] = "both",
    psi_threshold: float = 0.25,
    ks_alpha: float = 0.05,
    bins: int = 10,
    psi_interpretation_thresholds: tuple[float, float] = (0.1, 0.25),
) -> dict[str, DriftResult]:
    """Detect drift across multiple features.

    Compares feature distributions between baseline (training) and current
    (production) datasets. Useful for monitoring data quality and identifying
    when model retraining may be needed.

    Parameters
    ----------
    baseline_features:
        Baseline feature DataFrame (from training data)
    current_features:
        Current feature DataFrame (from production data)
    method:
        Drift detection method: "psi", "ks", or "both" (default: "both")
    psi_threshold:
        PSI threshold for drift detection (default: 0.25)
    ks_alpha:
        Significance level for KS test (default: 0.05)
    bins:
        Number of bins for PSI calculation (default: 10)
    psi_interpretation_thresholds:
        Tuple of (low, high) thresholds for PSI interpretation.
        Default: (0.1, 0.25) where PSI < low = "No significant drift",
        low ≤ PSI < high = "Small drift", PSI ≥ high = "Significant drift"

    Returns
    -------
    dict[str, DriftResult]
        Dictionary mapping feature names to drift detection results

    Raises
    ------
    ValueError
        If DataFrames have different columns or are empty

    Examples
    --------
    >>> baseline = pd.DataFrame({"age": np.random.normal(35, 10, 1000),
    ...                          "income": np.random.normal(50000, 15000, 1000)})
    >>> current = pd.DataFrame({"age": np.random.normal(37, 10, 1000),
    ...                         "income": np.random.normal(55000, 15000, 1000)})
    >>> results = detect_feature_drift(baseline, current, method="psi")
    >>> results["age"].drift_detected
    False
    >>> results["income"].drift_detected  # Significant shift
    True
    """
    if baseline_features.empty or current_features.empty:
        raise ValueError("Feature DataFrames must not be empty")

    # Check for column mismatch
    baseline_cols = set(baseline_features.columns)
    current_cols = set(current_features.columns)

    if baseline_cols != current_cols:
        missing_in_current = baseline_cols - current_cols
        missing_in_baseline = current_cols - baseline_cols
        raise ValueError(
            f"Column mismatch. Missing in current: {missing_in_current}. "
            f"Missing in baseline: {missing_in_baseline}"
        )

    results = {}

    for column in baseline_features.columns:
        baseline_values = baseline_features[column].dropna()
        current_values = current_features[column].dropna()

        if len(baseline_values) == 0 or len(current_values) == 0:
            logger.warning(
                f"Column '{column}' has no non-null values, skipping drift detection"
            )
            continue

        # Perform drift detection based on method
        if method in ["psi", "both"]:
            try:
                psi_score = calculate_psi(baseline_values, current_values, bins=bins)
                drift_detected = psi_score >= psi_threshold

                # Determine interpretation using configurable thresholds
                low_threshold, high_threshold = psi_interpretation_thresholds
                if psi_score < low_threshold:
                    interpretation = "No significant drift"
                elif psi_score < high_threshold:
                    interpretation = "Small drift, investigation recommended"
                else:
                    interpretation = "Significant drift, retraining recommended"

                results[f"{column}_psi"] = DriftResult(
                    metric_name=column,
                    test_type="psi",
                    drift_score=float(psi_score),  # Ensure native Python float
                    p_value=None,
                    drift_detected=bool(drift_detected),  # Ensure native Python bool
                    threshold=float(psi_threshold),  # Ensure native Python float
                    interpretation=interpretation,
                )
            except ValueError as e:
                logger.warning(f"PSI calculation failed for column '{column}': {e}")

        if method in ["ks", "both"]:
            try:
                ks_stat, p_value, drift_detected = kolmogorov_smirnov_test(
                    baseline_values, current_values, alpha=ks_alpha
                )

                interpretation = (
                    f"Distributions {'differ' if drift_detected else 'are similar'} "
                    f"(p={p_value:.4f}, {'<' if drift_detected else '>='} {ks_alpha})"
                )

                results[f"{column}_ks"] = DriftResult(
                    metric_name=column,
                    test_type="ks",
                    drift_score=float(ks_stat),  # Ensure native Python float
                    p_value=float(p_value),  # Ensure native Python float
                    drift_detected=bool(drift_detected),  # Ensure native Python bool
                    threshold=float(ks_alpha),  # Ensure native Python float
                    interpretation=interpretation,
                )
            except ValueError as e:
                logger.warning(f"KS test failed for column '{column}': {e}")

    return results


def detect_prediction_drift(
    baseline_predictions: np.ndarray | pd.Series,
    current_predictions: np.ndarray | pd.Series,
    method: Literal["psi", "ks", "both"] = "both",
    psi_threshold: float = 0.25,
    ks_alpha: float = 0.05,
    bins: int = 10,
    prediction_name: str = "prediction",
    psi_interpretation_thresholds: tuple[float, float] = (0.1, 0.25),
) -> dict[str, DriftResult]:
    """Detect drift in model predictions.

    Compares prediction distributions between baseline (training/validation) and
    current (production) periods. Prediction drift can indicate model degradation
    even when feature drift is minimal.

    Parameters
    ----------
    baseline_predictions:
        Baseline predictions (from training/validation)
    current_predictions:
        Current predictions (from production)
    method:
        Drift detection method: "psi", "ks", or "both" (default: "both")
    psi_threshold:
        PSI threshold for drift detection (default: 0.25)
    ks_alpha:
        Significance level for KS test (default: 0.05)
    bins:
        Number of bins for PSI calculation (default: 10)
    prediction_name:
        Name for the prediction metric (default: "prediction")

    Returns
    -------
    dict[str, DriftResult]
        Dictionary with drift detection results for predictions

    Raises
    ------
    ValueError
        If prediction arrays are empty

    Examples
    --------
    >>> baseline_pred = np.random.beta(2, 5, 1000)  # CLV predictions
    >>> current_pred_stable = np.random.beta(2, 5, 1000)
    >>> current_pred_shifted = np.random.beta(3, 4, 1000)  # Distribution shift
    >>> results_stable = detect_prediction_drift(baseline_pred, current_pred_stable)
    >>> results_stable[f"{prediction_name}_psi"].drift_detected
    False
    >>> results_shifted = detect_prediction_drift(baseline_pred, current_pred_shifted)
    >>> results_shifted[f"{prediction_name}_psi"].drift_detected
    True
    """
    if len(baseline_predictions) == 0 or len(current_predictions) == 0:
        raise ValueError("Prediction arrays must not be empty")

    # Convert to DataFrame for consistency with detect_feature_drift
    baseline_df = pd.DataFrame({prediction_name: baseline_predictions})
    current_df = pd.DataFrame({prediction_name: current_predictions})

    return detect_feature_drift(
        baseline_df,
        current_df,
        method=method,
        psi_threshold=psi_threshold,
        ks_alpha=ks_alpha,
        bins=bins,
        psi_interpretation_thresholds=psi_interpretation_thresholds,
    )
