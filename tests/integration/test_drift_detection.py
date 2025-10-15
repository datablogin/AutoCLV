"""Integration tests for drift detection with CLV models.

Tests the full drift detection workflow using realistic customer data and
CLV model outputs.
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from customer_base_audit.monitoring.drift import (
    detect_feature_drift,
    detect_prediction_drift,
)
from customer_base_audit.monitoring.exports import (
    export_drift_report_json,
    export_drift_report_markdown,
    get_drift_summary,
)


@pytest.fixture
def baseline_clv_features():
    """Generate baseline customer features for CLV model."""
    np.random.seed(42)
    n_customers = 1000

    return pd.DataFrame(
        {
            "frequency": np.random.poisson(3, n_customers),  # Number of purchases
            "recency": np.random.uniform(
                0, 365, n_customers
            ),  # Days since first purchase
            "T": np.random.uniform(180, 730, n_customers),  # Customer age in days
            "monetary_value": np.random.lognormal(
                4, 0.5, n_customers
            ),  # Average order value
        }
    )


@pytest.fixture
def current_clv_features_stable(baseline_clv_features):
    """Generate current features with no drift (stable)."""
    np.random.seed(100)
    n_customers = 1000

    return pd.DataFrame(
        {
            "frequency": np.random.poisson(3, n_customers),
            "recency": np.random.uniform(0, 365, n_customers),
            "T": np.random.uniform(180, 730, n_customers),
            "monetary_value": np.random.lognormal(4, 0.5, n_customers),
        }
    )


@pytest.fixture
def current_clv_features_drifted():
    """Generate current features with distribution shift (drifted)."""
    np.random.seed(200)
    n_customers = 1000

    # Simulate drift: customers making more purchases with higher value
    return pd.DataFrame(
        {
            "frequency": np.random.poisson(4, n_customers),  # Increased from 3 to 4
            "recency": np.random.uniform(0, 365, n_customers),  # Stable
            "T": np.random.uniform(180, 730, n_customers),  # Stable
            "monetary_value": np.random.lognormal(
                4.3, 0.5, n_customers
            ),  # Increased mean (~20% higher AOV)
        }
    )


@pytest.fixture
def baseline_clv_predictions(baseline_clv_features):
    """Generate baseline CLV predictions."""
    # Simplified CLV calculation for testing
    # Real CLV = E[frequency] * E[monetary_value] * customer_lifetime
    np.random.seed(42)

    clv = (
        baseline_clv_features["frequency"]
        * baseline_clv_features["monetary_value"]
        * (baseline_clv_features["T"] / 365)
    )

    # Add some noise
    clv = clv * np.random.lognormal(0, 0.2, len(clv))

    return clv.values


@pytest.fixture
def current_clv_predictions_stable(current_clv_features_stable):
    """Generate current CLV predictions with no drift."""
    np.random.seed(100)

    clv = (
        current_clv_features_stable["frequency"]
        * current_clv_features_stable["monetary_value"]
        * (current_clv_features_stable["T"] / 365)
    )

    clv = clv * np.random.lognormal(0, 0.2, len(clv))

    return clv.values


@pytest.fixture
def current_clv_predictions_drifted(current_clv_features_drifted):
    """Generate current CLV predictions with drift."""
    np.random.seed(200)

    clv = (
        current_clv_features_drifted["frequency"]
        * current_clv_features_drifted["monetary_value"]
        * (current_clv_features_drifted["T"] / 365)
    )

    clv = clv * np.random.lognormal(0, 0.2, len(clv))

    return clv.values


class TestFeatureDriftIntegration:
    """Integration tests for feature drift detection."""

    def test_no_drift_in_stable_features(
        self, baseline_clv_features, current_clv_features_stable
    ):
        """Should not detect drift when features are stable (Issue #37 acceptance criteria)."""
        results = detect_feature_drift(
            baseline_clv_features,
            current_clv_features_stable,
            method="both",
            psi_threshold=0.25,
        )

        # Count metrics with drift
        drifted_count = sum(1 for r in results.values() if r.drift_detected)

        # Should have minimal or no drift detected
        assert drifted_count <= 1, (
            f"Expected no drift in stable features, but {drifted_count} metrics show drift"
        )

    def test_drift_detected_in_shifted_features(
        self, baseline_clv_features, current_clv_features_drifted
    ):
        """Should detect drift when features shift significantly (Issue #37 acceptance criteria)."""
        results = detect_feature_drift(
            baseline_clv_features,
            current_clv_features_drifted,
            method="psi",
            psi_threshold=0.20,  # Lower threshold to ensure drift is detected
        )

        # Frequency and monetary_value should show drift
        assert results["frequency_psi"].drift_detected, (
            f"Expected drift in frequency (PSI={results['frequency_psi'].drift_score:.3f})"
        )
        assert results["monetary_value_psi"].drift_detected, (
            "Expected drift in monetary_value"
        )

        # Recency and T should be stable
        assert not results["recency_psi"].drift_detected, "Recency should be stable"
        assert not results["T_psi"].drift_detected, "T should be stable"

    def test_psi_values_increase_with_drift(
        self,
        baseline_clv_features,
        current_clv_features_stable,
        current_clv_features_drifted,
    ):
        """PSI scores should be higher for drifted distributions."""
        results_stable = detect_feature_drift(
            baseline_clv_features, current_clv_features_stable, method="psi"
        )

        results_drifted = detect_feature_drift(
            baseline_clv_features, current_clv_features_drifted, method="psi"
        )

        # Frequency PSI should be higher for drifted data
        psi_stable = results_stable["frequency_psi"].drift_score
        psi_drifted = results_drifted["frequency_psi"].drift_score

        assert psi_drifted > psi_stable, (
            f"Expected higher PSI for drifted data (stable={psi_stable:.3f}, drifted={psi_drifted:.3f})"
        )


class TestPredictionDriftIntegration:
    """Integration tests for prediction drift detection."""

    def test_no_drift_in_stable_predictions(
        self, baseline_clv_predictions, current_clv_predictions_stable
    ):
        """Should not detect drift in stable predictions (Issue #37 acceptance criteria)."""
        results = detect_prediction_drift(
            baseline_clv_predictions,
            current_clv_predictions_stable,
            method="both",
            psi_threshold=0.25,
        )

        # Should not detect drift
        assert not results["prediction_psi"].drift_detected
        assert not results["prediction_ks"].drift_detected

    def test_drift_detected_in_shifted_predictions(
        self, baseline_clv_predictions, current_clv_predictions_drifted
    ):
        """Should detect drift when predictions shift significantly (Issue #37 acceptance criteria)."""
        results = detect_prediction_drift(
            baseline_clv_predictions,
            current_clv_predictions_drifted,
            method="psi",
            psi_threshold=0.20,  # Slightly lower threshold for predictions
        )

        # Should detect drift in predictions
        assert results["prediction_psi"].drift_detected, (
            "Expected drift in CLV predictions"
        )

        # PSI should indicate significant shift
        assert results["prediction_psi"].drift_score > 0.20


class TestEndToEndDriftWorkflow:
    """Test complete drift detection and reporting workflow."""

    def test_full_drift_detection_pipeline(
        self,
        baseline_clv_features,
        current_clv_features_drifted,
        baseline_clv_predictions,
        current_clv_predictions_drifted,
    ):
        """Test full drift detection and reporting workflow."""
        # Step 1: Detect feature drift
        feature_results = detect_feature_drift(
            baseline_clv_features,
            current_clv_features_drifted,
            method="both",
        )

        # Step 2: Detect prediction drift
        prediction_results = detect_prediction_drift(
            baseline_clv_predictions,
            current_clv_predictions_drifted,
            method="both",
            prediction_name="clv",
        )

        # Step 3: Combine results
        all_results = {**feature_results, **prediction_results}

        # Step 4: Get summary
        summary = get_drift_summary(all_results)

        # Verify summary
        assert summary["total_metrics"] > 0
        assert summary["drifted_count"] > 0
        assert len(summary["metrics_with_drift"]) > 0

        # Step 5: Export reports
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "drift_report.json"
            md_path = Path(tmpdir) / "drift_report.md"

            export_drift_report_json(
                all_results,
                json_path,
                metadata={
                    "model": "BG/NBD + Gamma-Gamma",
                    "baseline_period": "2024-01-01 to 2024-01-31",
                    "current_period": "2024-02-01 to 2024-02-28",
                },
            )

            export_drift_report_markdown(
                all_results,
                md_path,
                title="CLV Model Drift Report - February 2024",
            )

            # Verify files were created
            assert json_path.exists()
            assert md_path.exists()

            # Verify markdown contains key sections
            md_content = md_path.read_text()
            assert "CLV Model Drift Report" in md_content
            assert "Summary" in md_content
            assert "frequency" in md_content.lower()

    def test_drift_detection_with_mean_shift_20_percent(self, baseline_clv_features):
        """Test drift detection with 20% mean shift (Issue #37 acceptance criteria)."""
        # Create current features with 20% mean shift in monetary value
        np.random.seed(300)
        n_customers = 1000

        current_features = pd.DataFrame(
            {
                "frequency": np.random.poisson(3, n_customers),  # Same
                "recency": np.random.uniform(0, 365, n_customers),  # Same
                "T": np.random.uniform(180, 730, n_customers),  # Same
                "monetary_value": np.random.lognormal(
                    4.18, 0.5, n_customers
                ),  # +20% mean shift
            }
        )

        results = detect_feature_drift(
            baseline_clv_features,
            current_features,
            method="psi",
            psi_threshold=0.1,  # Lower threshold for detection
        )

        # Should detect drift in monetary_value
        assert results["monetary_value_psi"].drift_detected, (
            "Expected drift with 20% mean shift"
        )

        # Other features should be stable
        assert not results["frequency_psi"].drift_detected
        assert not results["recency_psi"].drift_detected
        assert not results["T_psi"].drift_detected


class TestNoFalsePositives:
    """Test that stable distributions don't trigger false drift alerts (Issue #37 acceptance criteria)."""

    def test_repeated_stable_distribution_checks(self):
        """Multiple checks on stable distributions should not produce false positives."""
        np.random.seed(42)

        # Generate baseline
        baseline = pd.DataFrame(
            {
                "feature1": np.random.normal(100, 15, 1000),
                "feature2": np.random.exponential(50, 1000),
                "feature3": np.random.uniform(0, 1, 1000),
            }
        )

        # Run drift detection 10 times with different stable samples
        false_positive_count = 0

        for i in range(10):
            np.random.seed(100 + i)

            # Generate current from same distribution
            current = pd.DataFrame(
                {
                    "feature1": np.random.normal(100, 15, 1000),
                    "feature2": np.random.exponential(50, 1000),
                    "feature3": np.random.uniform(0, 1, 1000),
                }
            )

            results = detect_feature_drift(
                baseline, current, method="psi", psi_threshold=0.25
            )

            # Count any detected drift as false positive
            false_positive_count += sum(1 for r in results.values() if r.drift_detected)

        # With proper implementation, false positive rate should be very low
        # Allow at most 3 false positives out of 30 tests (10 runs × 3 features)
        assert false_positive_count <= 3, (
            f"Too many false positives: {false_positive_count}/30 tests"
        )
