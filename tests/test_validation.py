"""Tests for CLV model validation framework.

Tests validation metrics calculation, temporal data splitting, and cross-validation.
"""

from datetime import datetime
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from customer_base_audit.validation.validation import (
    ValidationMetrics,
    calculate_clv_metrics,
    cross_validate_clv,
    temporal_train_test_split,
)


class TestValidationMetrics:
    """Tests for ValidationMetrics dataclass."""

    def test_valid_metrics(self):
        """Test ValidationMetrics with valid values."""
        metrics = ValidationMetrics(
            mae=Decimal("10.50"),
            mape=Decimal("15.30"),
            rmse=Decimal("12.75"),
            arpe=Decimal("5.20"),
            r_squared=Decimal("0.850"),
            sample_size=100,
        )

        assert metrics.mae == Decimal("10.50")
        assert metrics.mape == Decimal("15.30")
        assert metrics.rmse == Decimal("12.75")
        assert metrics.arpe == Decimal("5.20")
        assert metrics.r_squared == Decimal("0.850")
        assert metrics.sample_size == 100

    def test_negative_mae_raises_error(self):
        """Test that negative MAE raises ValueError."""
        with pytest.raises(ValueError, match="mae must be non-negative"):
            ValidationMetrics(
                mae=Decimal("-10.0"),
                mape=Decimal("15.0"),
                rmse=Decimal("12.0"),
                arpe=Decimal("5.0"),
                r_squared=Decimal("0.8"),
                sample_size=100,
            )

    def test_negative_sample_size_raises_error(self):
        """Test that sample_size < 1 raises ValueError."""
        with pytest.raises(ValueError, match="sample_size must be >= 1"):
            ValidationMetrics(
                mae=Decimal("10.0"),
                mape=Decimal("15.0"),
                rmse=Decimal("12.0"),
                arpe=Decimal("5.0"),
                r_squared=Decimal("0.8"),
                sample_size=0,
            )

    def test_immutability(self):
        """Test that ValidationMetrics is immutable."""
        metrics = ValidationMetrics(
            mae=Decimal("10.0"),
            mape=Decimal("15.0"),
            rmse=Decimal("12.0"),
            arpe=Decimal("5.0"),
            r_squared=Decimal("0.8"),
            sample_size=100,
        )

        with pytest.raises(AttributeError):
            metrics.mae = Decimal("20.0")


class TestTemporalTrainTestSplit:
    """Tests for temporal_train_test_split()."""

    def test_basic_split(self):
        """Test basic temporal splitting."""
        transactions = pd.DataFrame(
            {
                "customer_id": ["C1", "C1", "C1", "C2", "C2"],
                "event_ts": pd.to_datetime(
                    [
                        "2023-01-15",
                        "2023-06-20",
                        "2023-12-10",
                        "2023-03-10",
                        "2023-11-05",
                    ]
                ),
                "amount": [50.0, 75.0, 100.0, 30.0, 45.0],
            }
        )

        train, obs, test = temporal_train_test_split(
            transactions,
            train_end_date=datetime(2023, 9, 1),
            observation_end_date=datetime(2023, 12, 31),
        )

        # Train: before Sept 1 (C1: Jan+Jun, C2: Mar = 3 txns)
        assert len(train) == 3
        assert set(train["customer_id"]) == {"C1", "C2"}

        # Observation: all through Dec 31
        assert len(obs) == 5

        # Test: Sept 1 - Dec 31 (C1: Dec, C2: Nov = 2 txns)
        assert len(test) == 2
        assert all(test["event_ts"] >= datetime(2023, 9, 1))

    def test_string_dates_converted(self):
        """Test that string dates are converted to datetime."""
        transactions = pd.DataFrame(
            {
                "customer_id": ["C1", "C1"],
                "event_ts": ["2023-01-15", "2023-06-20"],
                "amount": [50.0, 75.0],
            }
        )

        train, obs, test = temporal_train_test_split(
            transactions,
            train_end_date=datetime(2023, 4, 1),
            observation_end_date=datetime(2023, 12, 31),
        )

        assert len(train) == 1
        assert len(obs) == 2
        assert len(test) == 1

    def test_custom_column_names(self):
        """Test with custom column names."""
        transactions = pd.DataFrame(
            {
                "cust_id": ["C1", "C1"],
                "txn_date": pd.to_datetime(["2023-01-15", "2023-06-20"]),
                "amount": [50.0, 75.0],
            }
        )

        train, obs, test = temporal_train_test_split(
            transactions,
            train_end_date=datetime(2023, 4, 1),
            observation_end_date=datetime(2023, 12, 31),
            customer_id_col="cust_id",
            date_col="txn_date",
        )

        assert len(train) == 1
        assert "cust_id" in train.columns
        assert "txn_date" in train.columns

    def test_invalid_date_order_raises_error(self):
        """Test that train_end_date >= observation_end_date raises error."""
        transactions = pd.DataFrame(
            {
                "customer_id": ["C1"],
                "event_ts": pd.to_datetime(["2023-01-15"]),
                "amount": [50.0],
            }
        )

        with pytest.raises(ValueError, match="train_end_date .* must be before"):
            temporal_train_test_split(
                transactions,
                train_end_date=datetime(2023, 12, 31),
                observation_end_date=datetime(2023, 6, 1),
            )

    def test_missing_columns_raises_error(self):
        """Test that missing required columns raises error."""
        transactions = pd.DataFrame(
            {
                "customer_id": ["C1"],
                "amount": [50.0],
            }
        )

        with pytest.raises(ValueError, match="missing required columns"):
            temporal_train_test_split(
                transactions,
                train_end_date=datetime(2023, 6, 1),
                observation_end_date=datetime(2023, 12, 31),
            )

    def test_empty_test_set(self):
        """Test when test set is empty (no transactions in test period)."""
        transactions = pd.DataFrame(
            {
                "customer_id": ["C1", "C1"],
                "event_ts": pd.to_datetime(["2023-01-15", "2023-02-20"]),
                "amount": [50.0, 75.0],
            }
        )

        train, obs, test = temporal_train_test_split(
            transactions,
            train_end_date=datetime(2023, 12, 1),  # After all transactions
            observation_end_date=datetime(2023, 12, 31),
        )

        assert len(train) == 2
        assert len(obs) == 2
        assert len(test) == 0


class TestCalculateCLVMetrics:
    """Tests for calculate_clv_metrics()."""

    def test_perfect_predictions(self):
        """Test metrics when predictions are perfect."""
        actual = pd.Series([100.0, 150.0, 200.0, 50.0])
        predicted = pd.Series([100.0, 150.0, 200.0, 50.0])

        metrics = calculate_clv_metrics(actual, predicted)

        assert metrics.mae == Decimal("0.00")
        assert metrics.mape == Decimal("0.00")
        assert metrics.rmse == Decimal("0.00")
        assert metrics.arpe == Decimal("0.00")
        assert metrics.r_squared == Decimal("1.000")
        assert metrics.sample_size == 4

    def test_realistic_predictions(self):
        """Test metrics with realistic prediction errors."""
        actual = pd.Series([100.0, 150.0, 200.0, 50.0])
        predicted = pd.Series([95.0, 160.0, 190.0, 55.0])

        metrics = calculate_clv_metrics(actual, predicted)

        # MAE should be average of |5, 10, 10, 5| = 7.5
        assert metrics.mae == Decimal("7.50")

        # MAPE should be average of |5%, 6.67%, 5%, 10%| ≈ 6.67%
        assert 6.0 < float(metrics.mape) < 7.5

        # RMSE should be sqrt(mean([25, 100, 100, 25])) = sqrt(62.5) ≈ 7.91
        assert 7.5 < float(metrics.rmse) < 8.5

        # ARPE: |500 - 500| / 500 = 0%
        assert metrics.arpe == Decimal("0.00")

        # R² should be high (close to 1)
        assert float(metrics.r_squared) > 0.9

    def test_systematic_overestimation(self):
        """Test ARPE captures systematic bias."""
        actual = pd.Series([100.0, 150.0, 200.0, 50.0])
        predicted = pd.Series([120.0, 170.0, 220.0, 70.0])  # All +20

        metrics = calculate_clv_metrics(actual, predicted)

        # Total actual: 500, Total predicted: 580
        # ARPE = |500 - 580| / 500 * 100 = 16%
        assert 15.0 < float(metrics.arpe) < 17.0

    def test_length_mismatch_raises_error(self):
        """Test that different lengths raise error."""
        actual = pd.Series([100.0, 150.0])
        predicted = pd.Series([95.0, 160.0, 190.0])

        with pytest.raises(ValueError, match="must have same length"):
            calculate_clv_metrics(actual, predicted)

    def test_empty_series_raises_error(self):
        """Test that empty series raise error."""
        actual = pd.Series([])
        predicted = pd.Series([])

        with pytest.raises(ValueError, match="cannot be empty"):
            calculate_clv_metrics(actual, predicted)

    def test_nan_values_raise_error(self):
        """Test that NaN values raise error."""
        actual = pd.Series([100.0, np.nan, 200.0])
        predicted = pd.Series([95.0, 160.0, 190.0])

        with pytest.raises(ValueError, match="contains NaN or inf"):
            calculate_clv_metrics(actual, predicted)

    def test_zero_actual_values(self):
        """Test metrics when some actual values are zero."""
        actual = pd.Series([100.0, 0.0, 200.0, 0.0])
        predicted = pd.Series([95.0, 10.0, 190.0, 5.0])

        metrics = calculate_clv_metrics(actual, predicted)

        # MAPE only calculated on non-zero actual values
        # Should be average of |5%/100, 5%/200| = |5%, 2.5%| ≈ 3.75%
        assert metrics.mape > Decimal("0.0")
        assert float(metrics.mape) < 10.0

        # Other metrics should still be calculated
        assert metrics.mae > Decimal("0.0")
        assert metrics.sample_size == 4

    def test_all_zero_actual_values(self):
        """Test when all actual values are zero."""
        actual = pd.Series([0.0, 0.0, 0.0])
        predicted = pd.Series([10.0, 15.0, 20.0])

        metrics = calculate_clv_metrics(actual, predicted)

        # MAPE and ARPE undefined, set to 0
        assert metrics.mape == Decimal("0.00")
        assert metrics.arpe == Decimal("0.00")

        # MAE and RMSE should still be calculated
        assert metrics.mae > Decimal("0.0")
        assert metrics.rmse > Decimal("0.0")


class TestCrossValidateCLV:
    """Tests for cross_validate_clv()."""

    def test_basic_cross_validation(self):
        """Test basic cross-validation with simple pipeline."""
        # Create transactions spanning 24 months
        np.random.seed(42)
        n_customers = 10
        dates = pd.date_range("2023-01-01", periods=720, freq="D")  # 2 years

        transactions = []
        for cust_id in range(n_customers):
            # Each customer has random transactions
            n_txns = np.random.randint(5, 20)
            cust_dates = np.random.choice(dates, size=n_txns, replace=False)
            for txn_date in cust_dates:
                transactions.append(
                    {
                        "customer_id": f"C{cust_id}",
                        "event_ts": txn_date,
                        "amount": np.random.uniform(10, 100),
                    }
                )

        txn_df = pd.DataFrame(transactions)

        # Simple pipeline: predict average historical CLV
        def simple_pipeline(train_txns, obs_end_date):
            clv_by_customer = train_txns.groupby("customer_id")["amount"].sum()
            return pd.DataFrame(
                {
                    "customer_id": clv_by_customer.index,
                    "clv": clv_by_customer.values,
                }
            )

        # Run 3-fold validation
        metrics_list = cross_validate_clv(
            txn_df,
            simple_pipeline,
            n_folds=3,
            time_increment_months=3,
            initial_train_months=12,
        )

        # Should get 3 validation metrics
        assert len(metrics_list) == 3

        # All should be ValidationMetrics
        assert all(isinstance(m, ValidationMetrics) for m in metrics_list)

        # All should have non-zero samples
        assert all(m.sample_size > 0 for m in metrics_list)

    def test_invalid_n_folds_raises_error(self):
        """Test that n_folds < 1 raises error."""
        txn_df = pd.DataFrame(
            {
                "customer_id": ["C1"],
                "event_ts": pd.to_datetime(["2023-01-01"]),
                "amount": [50.0],
            }
        )

        def dummy_pipeline(train_txns, obs_end_date):
            return pd.DataFrame({"customer_id": ["C1"], "clv": [50.0]})

        with pytest.raises(ValueError, match="n_folds must be >= 1"):
            cross_validate_clv(txn_df, dummy_pipeline, n_folds=0)

    def test_pipeline_missing_customer_id_raises_error(self):
        """Test that pipeline output without customer_id raises error."""
        txn_df = pd.DataFrame(
            {
                "customer_id": ["C1", "C1"],
                "event_ts": pd.date_range("2023-01-01", periods=2, freq="6ME"),
                "amount": [50.0, 75.0],
            }
        )

        def bad_pipeline(train_txns, obs_end_date):
            return pd.DataFrame({"clv": [50.0]})  # Missing customer_id

        with pytest.raises(ValueError, match="missing 'customer_id'"):
            cross_validate_clv(
                txn_df,
                bad_pipeline,
                n_folds=1,
                time_increment_months=3,
                initial_train_months=3,
            )

    def test_pipeline_missing_clv_raises_error(self):
        """Test that pipeline output without clv raises error."""
        txn_df = pd.DataFrame(
            {
                "customer_id": ["C1", "C1"],
                "event_ts": pd.date_range("2023-01-01", periods=2, freq="6ME"),
                "amount": [50.0, 75.0],
            }
        )

        def bad_pipeline(train_txns, obs_end_date):
            return pd.DataFrame({"customer_id": ["C1"]})  # Missing clv

        with pytest.raises(ValueError, match="missing 'clv'"):
            cross_validate_clv(
                txn_df,
                bad_pipeline,
                n_folds=1,
                time_increment_months=3,
                initial_train_months=3,
            )

    def test_insufficient_data_returns_fewer_folds(self):
        """Test that insufficient data results in fewer folds."""
        # Only 6 months of data
        txn_df = pd.DataFrame(
            {
                "customer_id": ["C1", "C1"],
                "event_ts": pd.to_datetime(["2023-01-15", "2023-05-20"]),
                "amount": [50.0, 75.0],
            }
        )

        def simple_pipeline(train_txns, obs_end_date):
            return pd.DataFrame({"customer_id": ["C1"], "clv": [50.0]})

        # Request 5 folds but only have data for 0-1 folds
        metrics_list = cross_validate_clv(
            txn_df,
            simple_pipeline,
            n_folds=5,
            time_increment_months=3,
            initial_train_months=12,  # 12 months initial + 3*5 = 27 months needed
        )

        # Should get 0 folds (not enough data for even first fold)
        assert len(metrics_list) == 0
