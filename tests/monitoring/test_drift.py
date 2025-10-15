"""Unit tests for drift detection module."""

import numpy as np
import pandas as pd
import pytest

from customer_base_audit.monitoring.drift import (
    DriftResult,
    calculate_psi,
    detect_feature_drift,
    detect_prediction_drift,
    kolmogorov_smirnov_test,
)


class TestCalculatePSI:
    """Test PSI (Population Stability Index) calculation."""

    def test_psi_identical_distributions(self):
        """PSI should be near zero for identical distributions."""
        np.random.seed(42)
        baseline = np.random.normal(0, 1, 1000)
        current = baseline.copy()

        psi = calculate_psi(baseline, current)
        assert psi < 0.01, f"Expected PSI near 0 for identical distributions, got {psi}"

    def test_psi_stable_distributions(self):
        """PSI should be low for similar distributions (no drift)."""
        np.random.seed(42)
        baseline = np.random.normal(0, 1, 5000)  # Larger sample for stability
        np.random.seed(43)  # Different seed for current
        current = np.random.normal(0, 1, 5000)  # Same distribution, different sample

        psi = calculate_psi(baseline, current)
        assert psi < 0.1, f"Expected PSI < 0.1 for stable distributions, got {psi}"

    def test_psi_shifted_mean(self):
        """PSI should detect mean shift (drift)."""
        np.random.seed(42)
        baseline = np.random.normal(0, 1, 1000)
        current = np.random.normal(0.5, 1, 1000)  # Mean shifted by 0.5 std

        psi = calculate_psi(baseline, current)
        assert psi > 0.1, f"Expected PSI > 0.1 for shifted mean, got {psi}"

    def test_psi_large_shift(self):
        """PSI should be high for large distribution shift."""
        np.random.seed(42)
        baseline = np.random.normal(0, 1, 1000)
        current = np.random.normal(1.5, 1, 1000)  # Large mean shift

        psi = calculate_psi(baseline, current)
        assert psi >= 0.25, f"Expected PSI >= 0.25 for large shift, got {psi}"

    def test_psi_variance_change(self):
        """PSI should detect variance change."""
        np.random.seed(42)
        baseline = np.random.normal(0, 1, 1000)
        current = np.random.normal(0, 2, 1000)  # Same mean, doubled variance

        psi = calculate_psi(baseline, current)
        assert psi > 0.1, f"Expected PSI > 0.1 for variance change, got {psi}"

    def test_psi_with_pandas_series(self):
        """PSI should work with pandas Series."""
        np.random.seed(42)
        baseline = pd.Series(np.random.normal(0, 1, 1000))
        current = pd.Series(np.random.normal(0.5, 1, 1000))

        psi = calculate_psi(baseline, current)
        assert psi > 0.1

    def test_psi_custom_bins(self):
        """PSI should accept custom bin count."""
        np.random.seed(42)
        baseline = np.random.normal(0, 1, 1000)
        current = np.random.normal(0.5, 1, 1000)

        psi_10 = calculate_psi(baseline, current, bins=10)
        psi_20 = calculate_psi(baseline, current, bins=20)

        # Both should detect drift, but values may differ slightly
        assert psi_10 > 0.05
        assert psi_20 > 0.05

    def test_psi_empty_input_raises_error(self):
        """PSI should raise error for empty input."""
        with pytest.raises(ValueError, match="must not be empty"):
            calculate_psi(np.array([]), np.array([1, 2, 3]))

    def test_psi_insufficient_data_raises_error(self):
        """PSI should raise error if not enough data for bins."""
        with pytest.raises(ValueError, match="Need at least"):
            calculate_psi(np.array([1, 2, 3]), np.array([4, 5, 6]), bins=10)

    def test_psi_nan_values_handled(self):
        """PSI should handle NaN values by removing them."""
        np.random.seed(42)
        baseline = np.array([np.nan, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 100)
        current = np.array([1, 2, np.nan, 3, 4, 5, 6, 7, 8, 9, 10] * 100)

        psi = calculate_psi(baseline, current)
        assert psi < 0.1  # Should be low since distributions are similar

    def test_psi_constant_values(self):
        """PSI should handle constant values gracefully."""
        baseline = np.ones(100)
        current = np.ones(100)

        psi = calculate_psi(baseline, current)
        assert psi == 0.0


class TestKolmogorovSmirnovTest:
    """Test Kolmogorov-Smirnov drift test."""

    def test_ks_identical_distributions(self):
        """KS test should not detect drift for identical distributions."""
        np.random.seed(42)
        baseline = np.random.normal(0, 1, 1000)
        current = baseline.copy()

        statistic, p_value, drift_detected = kolmogorov_smirnov_test(baseline, current)

        assert statistic < 0.05, f"Expected low KS statistic, got {statistic}"
        assert p_value > 0.05, f"Expected high p-value, got {p_value}"
        assert not drift_detected

    def test_ks_stable_distributions(self):
        """KS test should not detect drift for similar distributions."""
        np.random.seed(42)
        baseline = np.random.normal(0, 1, 5000)  # Larger sample for stability
        np.random.seed(43)  # Different seed for current
        current = np.random.normal(0, 1, 5000)  # Different sample, same distribution

        _, p_value, drift_detected = kolmogorov_smirnov_test(baseline, current)

        assert p_value > 0.05, (
            f"Expected p > 0.05 for stable distributions, got {p_value}"
        )
        assert not drift_detected

    def test_ks_shifted_distribution(self):
        """KS test should detect drift for shifted distributions."""
        np.random.seed(42)
        baseline = np.random.normal(0, 1, 1000)
        current = np.random.normal(0.5, 1, 1000)  # Mean shifted

        statistic, p_value, drift_detected = kolmogorov_smirnov_test(baseline, current)

        assert p_value < 0.05, (
            f"Expected p < 0.05 for shifted distribution, got {p_value}"
        )
        assert drift_detected

    def test_ks_large_shift(self):
        """KS test should strongly detect large distribution shift."""
        np.random.seed(42)
        baseline = np.random.normal(0, 1, 1000)
        current = np.random.normal(2, 1, 1000)  # Large shift

        statistic, p_value, drift_detected = kolmogorov_smirnov_test(baseline, current)

        assert statistic > 0.2, f"Expected high KS statistic, got {statistic}"
        assert p_value < 0.001, f"Expected very low p-value, got {p_value}"
        assert drift_detected

    def test_ks_custom_alpha(self):
        """KS test should respect custom significance level."""
        np.random.seed(42)
        baseline = np.random.normal(0, 1, 1000)
        current = np.random.normal(0.3, 1, 1000)  # Moderate shift

        # With alpha=0.05, might detect drift
        _, p_value_005, drift_005 = kolmogorov_smirnov_test(
            baseline, current, alpha=0.05
        )

        # With alpha=0.01, might not detect drift (more conservative)
        _, p_value_001, drift_001 = kolmogorov_smirnov_test(
            baseline, current, alpha=0.01
        )

        # Both should have same p-value, but different drift_detected flags
        assert p_value_005 == p_value_001
        # If p-value is between 0.01 and 0.05, they should differ
        if 0.01 < p_value_005 < 0.05:
            assert drift_005 != drift_001

    def test_ks_with_pandas_series(self):
        """KS test should work with pandas Series."""
        np.random.seed(42)
        baseline = pd.Series(np.random.normal(0, 1, 1000))
        current = pd.Series(np.random.normal(0.5, 1, 1000))

        _, p_value, drift_detected = kolmogorov_smirnov_test(baseline, current)

        assert p_value < 0.05
        assert drift_detected

    def test_ks_empty_input_raises_error(self):
        """KS test should raise error for empty input."""
        with pytest.raises(ValueError, match="must not be empty"):
            kolmogorov_smirnov_test(np.array([]), np.array([1, 2, 3]))

    def test_ks_nan_values_handled(self):
        """KS test should handle NaN values."""
        np.random.seed(42)
        baseline = np.array([np.nan, 1, 2, 3, 4, 5] * 200)
        current = np.array([1, 2, np.nan, 3, 4, 5] * 200)

        _, p_value, drift_detected = kolmogorov_smirnov_test(baseline, current)

        assert p_value > 0.05  # Should not detect drift in similar distributions
        assert not drift_detected


class TestDetectFeatureDrift:
    """Test feature drift detection across multiple features."""

    def test_detect_drift_multiple_features_stable(self):
        """Should not detect drift in stable features."""
        np.random.seed(42)
        baseline = pd.DataFrame(
            {
                "age": np.random.normal(35, 10, 1000),
                "income": np.random.normal(50000, 15000, 1000),
                "score": np.random.uniform(0, 100, 1000),
            }
        )
        current = pd.DataFrame(
            {
                "age": np.random.normal(35, 10, 1000),
                "income": np.random.normal(50000, 15000, 1000),
                "score": np.random.uniform(0, 100, 1000),
            }
        )

        results = detect_feature_drift(baseline, current, method="psi")

        # All features should be stable
        for key, result in results.items():
            assert not result.drift_detected, (
                f"Unexpected drift in {key}: PSI={result.drift_score}"
            )

    def test_detect_drift_one_feature_shifted(self):
        """Should detect drift in shifted feature."""
        np.random.seed(42)
        baseline = pd.DataFrame(
            {
                "age": np.random.normal(35, 10, 1000),
                "income": np.random.normal(50000, 15000, 1000),
            }
        )
        current = pd.DataFrame(
            {
                "age": np.random.normal(35, 10, 1000),  # Stable
                "income": np.random.normal(60000, 15000, 1000),  # Shifted +20%
            }
        )

        results = detect_feature_drift(
            baseline, current, method="psi", psi_threshold=0.25
        )

        # Age should be stable
        assert not results["age_psi"].drift_detected
        # Income should show drift
        assert results["income_psi"].drift_detected

    def test_detect_drift_both_methods(self):
        """Should run both PSI and KS tests when method='both'."""
        np.random.seed(42)
        baseline = pd.DataFrame({"feature": np.random.normal(0, 1, 1000)})
        current = pd.DataFrame({"feature": np.random.normal(0.5, 1, 1000)})

        results = detect_feature_drift(baseline, current, method="both")

        # Should have results for both methods
        assert "feature_psi" in results
        assert "feature_ks" in results

        # Both should detect drift
        assert results["feature_psi"].drift_detected
        assert results["feature_ks"].drift_detected

    def test_detect_drift_column_mismatch_raises_error(self):
        """Should raise error if columns don't match."""
        baseline = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        current = pd.DataFrame({"a": [7, 8, 9], "c": [10, 11, 12]})

        with pytest.raises(ValueError, match="Column mismatch"):
            detect_feature_drift(baseline, current)

    def test_detect_drift_empty_dataframe_raises_error(self):
        """Should raise error for empty DataFrames."""
        baseline = pd.DataFrame({"a": [1, 2, 3]})
        current = pd.DataFrame()

        with pytest.raises(ValueError, match="must not be empty"):
            detect_feature_drift(baseline, current)

    def test_detect_drift_all_nan_column_skipped(self):
        """Should skip columns with all NaN values."""
        baseline = pd.DataFrame({"a": [1, 2, 3, 4, 5] * 20, "b": [np.nan] * 100})
        current = pd.DataFrame({"a": [6, 7, 8, 9, 10] * 20, "b": [np.nan] * 100})

        results = detect_feature_drift(baseline, current, method="psi")

        # Should have result for 'a' but not 'b'
        assert "a_psi" in results
        assert "b_psi" not in results


class TestDetectPredictionDrift:
    """Test prediction drift detection."""

    def test_detect_prediction_drift_stable(self):
        """Should not detect drift in stable predictions."""
        np.random.seed(42)
        baseline = np.random.beta(2, 5, 1000)  # CLV-like distribution
        current = np.random.beta(2, 5, 1000)

        results = detect_prediction_drift(baseline, current, method="psi")

        assert not results["prediction_psi"].drift_detected

    def test_detect_prediction_drift_shifted(self):
        """Should detect drift in shifted predictions."""
        np.random.seed(42)
        baseline = np.random.beta(2, 5, 1000)
        current = np.random.beta(3, 4, 1000)  # Distribution shift

        results = detect_prediction_drift(
            baseline, current, method="psi", psi_threshold=0.1
        )

        assert results["prediction_psi"].drift_detected

    def test_detect_prediction_drift_custom_name(self):
        """Should use custom prediction name."""
        np.random.seed(42)
        baseline = np.random.normal(100, 20, 1000)
        current = np.random.normal(120, 20, 1000)

        results = detect_prediction_drift(
            baseline,
            current,
            method="psi",
            prediction_name="clv_score",
        )

        assert "clv_score_psi" in results
        assert results["clv_score_psi"].metric_name == "clv_score"

    def test_detect_prediction_drift_with_pandas_series(self):
        """Should work with pandas Series."""
        np.random.seed(42)
        baseline = pd.Series(np.random.normal(0, 1, 1000))
        current = pd.Series(np.random.normal(0.5, 1, 1000))

        results = detect_prediction_drift(baseline, current)

        assert "prediction_psi" in results


class TestDriftResult:
    """Test DriftResult dataclass."""

    def test_drift_result_creation(self):
        """Should create DriftResult with all fields."""
        result = DriftResult(
            metric_name="age",
            test_type="psi",
            drift_score=0.15,
            p_value=None,
            drift_detected=True,
            threshold=0.1,
            interpretation="Small drift detected",
        )

        assert result.metric_name == "age"
        assert result.test_type == "psi"
        assert result.drift_score == 0.15
        assert result.p_value is None
        assert result.drift_detected
        assert result.threshold == 0.1

    def test_drift_result_immutable(self):
        """DriftResult should be immutable (frozen dataclass)."""
        result = DriftResult(
            metric_name="age",
            test_type="psi",
            drift_score=0.15,
            p_value=None,
            drift_detected=True,
            threshold=0.1,
            interpretation="Small drift",
        )

        with pytest.raises(AttributeError):
            result.drift_score = 0.20  # Should fail
