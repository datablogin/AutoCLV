"""Tests for model input preparation utilities (BG/NBD and Gamma-Gamma)."""

from datetime import datetime
from decimal import Decimal

import pandas as pd
import pytest

from customer_base_audit.foundation.data_mart import PeriodAggregation
from customer_base_audit.models.model_prep import (
    BGNBDInput,
    GammaGammaInput,
    prepare_bg_nbd_inputs,
    prepare_gamma_gamma_inputs,
)


class TestBGNBDInput:
    """Test BGNBDInput dataclass validation."""

    def test_valid_bgnbd_input(self):
        """Valid BG/NBD input should be created successfully."""
        input_data = BGNBDInput(customer_id="C1", frequency=5, recency=30.0, T=90.0)
        assert input_data.customer_id == "C1"
        assert input_data.frequency == 5
        assert input_data.recency == 30.0
        assert input_data.T == 90.0

    def test_negative_frequency_raises_error(self):
        """Negative frequency should raise ValueError."""
        with pytest.raises(ValueError, match="Frequency cannot be negative"):
            BGNBDInput(customer_id="C1", frequency=-1, recency=30.0, T=90.0)

    def test_negative_recency_raises_error(self):
        """Negative recency should raise ValueError."""
        with pytest.raises(ValueError, match="Recency cannot be negative"):
            BGNBDInput(customer_id="C1", frequency=5, recency=-10.0, T=90.0)

    def test_zero_T_raises_error(self):
        """Zero T should raise ValueError."""
        with pytest.raises(ValueError, match="T must be positive"):
            BGNBDInput(customer_id="C1", frequency=5, recency=30.0, T=0.0)

    def test_negative_T_raises_error(self):
        """Negative T should raise ValueError."""
        with pytest.raises(ValueError, match="T must be positive"):
            BGNBDInput(customer_id="C1", frequency=5, recency=30.0, T=-90.0)

    def test_recency_exceeds_T_raises_error(self):
        """Recency exceeding T should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid BG/NBD input: recency"):
            BGNBDInput(customer_id="C1", frequency=5, recency=100.0, T=90.0)

    def test_recency_equals_T_is_valid(self):
        """Recency equal to T should be valid (edge case)."""
        input_data = BGNBDInput(customer_id="C1", frequency=5, recency=90.0, T=90.0)
        assert input_data.recency == input_data.T


class TestGammaGammaInput:
    """Test GammaGammaInput dataclass validation."""

    def test_valid_gamma_gamma_input(self):
        """Valid Gamma-Gamma input should be created successfully."""
        input_data = GammaGammaInput(
            customer_id="C1", frequency=5, monetary_value=Decimal("50.00")
        )
        assert input_data.customer_id == "C1"
        assert input_data.frequency == 5
        assert input_data.monetary_value == Decimal("50.00")

    def test_frequency_less_than_2_raises_error(self):
        """Frequency < 2 should raise ValueError."""
        with pytest.raises(ValueError, match="Frequency must be >= 2"):
            GammaGammaInput(
                customer_id="C1", frequency=1, monetary_value=Decimal("50.00")
            )

    def test_negative_monetary_value_raises_error(self):
        """Negative monetary value should raise ValueError."""
        with pytest.raises(ValueError, match="Monetary value must be positive"):
            GammaGammaInput(
                customer_id="C1", frequency=5, monetary_value=Decimal("-50.00")
            )

    def test_frequency_exactly_2_is_valid(self):
        """Frequency exactly 2 should be valid (minimum for Gamma-Gamma)."""
        input_data = GammaGammaInput(
            customer_id="C1", frequency=2, monetary_value=Decimal("50.00")
        )
        assert input_data.frequency == 2


class TestPrepareBGNBDInputs:
    """Test prepare_bg_nbd_inputs function."""

    def test_empty_input_returns_empty_dataframe(self):
        """Empty period aggregations should return empty DataFrame with correct columns."""
        df = prepare_bg_nbd_inputs([], datetime(2023, 1, 1), datetime(2023, 12, 31))
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
        assert list(df.columns) == ["customer_id", "frequency", "recency", "T"]

    def test_single_customer_single_period(self):
        """Single customer with single period should have frequency=0."""
        periods = [
            PeriodAggregation(
                customer_id="C1",
                period_start=datetime(2023, 1, 1),
                period_end=datetime(2023, 2, 1),
                total_orders=1,
                total_spend=100.0,
                total_margin=30.0,
                total_quantity=5,
            )
        ]
        df = prepare_bg_nbd_inputs(periods, datetime(2023, 1, 1), datetime(2023, 6, 1))
        assert len(df) == 1
        assert df.loc[0, "customer_id"] == "C1"
        assert df.loc[0, "frequency"] == 0  # 1 order - 1 = 0 repeat purchases
        assert df.loc[0, "recency"] == 0.0  # No repeat purchases
        # T = from first period start (2023-01-01) to observation end (2023-06-01)
        expected_T = (datetime(2023, 6, 1) - datetime(2023, 1, 1)).days
        assert df.loc[0, "T"] == pytest.approx(expected_T, abs=1.0)

    def test_single_customer_multiple_periods(self):
        """Single customer with multiple periods should calculate frequency, recency, T."""
        periods = [
            PeriodAggregation(
                customer_id="C1",
                period_start=datetime(2023, 1, 1),
                period_end=datetime(2023, 2, 1),
                total_orders=2,
                total_spend=150.0,
                total_margin=45.0,
                total_quantity=10,
            ),
            PeriodAggregation(
                customer_id="C1",
                period_start=datetime(2023, 3, 1),
                period_end=datetime(2023, 4, 1),
                total_orders=1,
                total_spend=50.0,
                total_margin=15.0,
                total_quantity=3,
            ),
        ]
        df = prepare_bg_nbd_inputs(periods, datetime(2023, 1, 1), datetime(2023, 6, 1))
        assert len(df) == 1
        assert df.loc[0, "customer_id"] == "C1"
        assert df.loc[0, "frequency"] == 2  # 3 total orders - 1 = 2 repeat purchases
        # Recency = from first period start (2023-01-01) to last period end (2023-04-01)
        expected_recency = (datetime(2023, 4, 1) - datetime(2023, 1, 1)).days
        assert df.loc[0, "recency"] == pytest.approx(expected_recency, abs=1.0)
        # T = from first period start (2023-01-01) to observation end (2023-06-01)
        expected_T = (datetime(2023, 6, 1) - datetime(2023, 1, 1)).days
        assert df.loc[0, "T"] == pytest.approx(expected_T, abs=1.0)

    def test_multiple_customers(self):
        """Multiple customers should be correctly grouped and sorted."""
        periods = [
            PeriodAggregation(
                "C2",
                datetime(2023, 2, 1),
                datetime(2023, 3, 1),
                1,
                75.0,
                20.0,
                3,
            ),
            PeriodAggregation(
                "C1",
                datetime(2023, 1, 1),
                datetime(2023, 2, 1),
                2,
                100.0,
                30.0,
                5,
            ),
            PeriodAggregation(
                "C1",
                datetime(2023, 3, 1),
                datetime(2023, 4, 1),
                1,
                50.0,
                15.0,
                2,
            ),
        ]
        df = prepare_bg_nbd_inputs(periods, datetime(2023, 1, 1), datetime(2023, 6, 1))
        assert len(df) == 2
        # Should be sorted by customer_id
        assert df.loc[0, "customer_id"] == "C1"
        assert df.loc[1, "customer_id"] == "C2"

        # C1: 3 total orders - 1 = 2 repeat purchases
        assert df.loc[0, "frequency"] == 2
        # C2: 1 total order - 1 = 0 repeat purchases
        assert df.loc[1, "frequency"] == 0

    def test_customer_with_exactly_2_transactions(self):
        """Customer with exactly 2 transactions (edge case for Gamma-Gamma threshold)."""
        periods = [
            PeriodAggregation(
                "C1",
                datetime(2023, 1, 1),
                datetime(2023, 2, 1),
                1,
                50.0,
                15.0,
                3,
            ),
            PeriodAggregation(
                "C1",
                datetime(2023, 2, 1),
                datetime(2023, 3, 1),
                1,
                75.0,
                20.0,
                4,
            ),
        ]
        df = prepare_bg_nbd_inputs(periods, datetime(2023, 1, 1), datetime(2023, 6, 1))
        assert len(df) == 1
        assert df.loc[0, "frequency"] == 1  # 2 orders - 1 = 1 repeat purchase

    def test_non_overlapping_periods(self):
        """Customers with non-overlapping activity periods."""
        periods = [
            PeriodAggregation(
                "C1",
                datetime(2023, 1, 1),
                datetime(2023, 2, 1),
                2,
                100.0,
                30.0,
                5,
            ),
            PeriodAggregation(
                "C1",
                datetime(2023, 5, 1),
                datetime(2023, 6, 1),
                1,
                50.0,
                15.0,
                3,
            ),
        ]
        df = prepare_bg_nbd_inputs(
            periods, datetime(2023, 1, 1), datetime(2023, 12, 31)
        )
        assert len(df) == 1
        assert df.loc[0, "frequency"] == 2  # 3 total orders - 1
        # Recency should span from first period start to last period end
        expected_recency = (datetime(2023, 6, 1) - datetime(2023, 1, 1)).days
        assert df.loc[0, "recency"] == pytest.approx(expected_recency, abs=1.0)


class TestPrepareGammaGammaInputs:
    """Test prepare_gamma_gamma_inputs function."""

    def test_empty_input_returns_empty_dataframe(self):
        """Empty period aggregations should return empty DataFrame with correct columns."""
        df = prepare_gamma_gamma_inputs([])
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
        assert list(df.columns) == ["customer_id", "frequency", "monetary_value"]

    def test_single_customer_above_threshold(self):
        """Customer with frequency >= min_frequency should be included."""
        periods = [
            PeriodAggregation(
                "C1",
                datetime(2023, 1, 1),
                datetime(2023, 2, 1),
                3,
                150.0,
                45.0,
                10,
            )
        ]
        df = prepare_gamma_gamma_inputs(periods, min_frequency=2)
        assert len(df) == 1
        assert df.loc[0, "customer_id"] == "C1"
        assert df.loc[0, "frequency"] == 3
        # Monetary value = 150.0 / 3 = 50.0
        assert df.loc[0, "monetary_value"] == pytest.approx(50.0, abs=0.01)

    def test_one_time_buyer_excluded(self):
        """Customer with frequency < min_frequency should be excluded."""
        periods = [
            PeriodAggregation(
                "C1",
                datetime(2023, 1, 1),
                datetime(2023, 2, 1),
                1,
                50.0,
                15.0,
                3,
            )
        ]
        df = prepare_gamma_gamma_inputs(periods, min_frequency=2)
        assert len(df) == 0  # One-time buyer excluded

    def test_exactly_min_frequency_included(self):
        """Customer with frequency exactly equal to min_frequency should be included."""
        periods = [
            PeriodAggregation(
                "C1",
                datetime(2023, 1, 1),
                datetime(2023, 2, 1),
                2,
                100.0,
                30.0,
                5,
            )
        ]
        df = prepare_gamma_gamma_inputs(periods, min_frequency=2)
        assert len(df) == 1
        assert df.loc[0, "frequency"] == 2
        # Monetary value = 100.0 / 2 = 50.0
        assert df.loc[0, "monetary_value"] == pytest.approx(50.0, abs=0.01)

    def test_multiple_customers_mixed_frequencies(self):
        """Multiple customers with different frequencies (some excluded)."""
        periods = [
            # C1: 3 orders (included)
            PeriodAggregation(
                "C1",
                datetime(2023, 1, 1),
                datetime(2023, 2, 1),
                3,
                150.0,
                45.0,
                10,
            ),
            # C2: 1 order (excluded)
            PeriodAggregation(
                "C2",
                datetime(2023, 1, 1),
                datetime(2023, 2, 1),
                1,
                50.0,
                15.0,
                3,
            ),
            # C3: 5 orders across 2 periods (included)
            PeriodAggregation(
                "C3",
                datetime(2023, 1, 1),
                datetime(2023, 2, 1),
                2,
                100.0,
                30.0,
                5,
            ),
            PeriodAggregation(
                "C3",
                datetime(2023, 3, 1),
                datetime(2023, 4, 1),
                3,
                150.0,
                45.0,
                8,
            ),
        ]
        df = prepare_gamma_gamma_inputs(periods, min_frequency=2)
        assert len(df) == 2  # C1 and C3 included, C2 excluded
        # Should be sorted by customer_id
        assert df.loc[0, "customer_id"] == "C1"
        assert df.loc[1, "customer_id"] == "C3"

        # C1: 3 orders, 150.0 spend -> 50.0 avg
        assert df.loc[0, "frequency"] == 3
        assert df.loc[0, "monetary_value"] == pytest.approx(50.0, abs=0.01)

        # C3: 5 orders, 250.0 total spend -> 50.0 avg
        assert df.loc[1, "frequency"] == 5
        assert df.loc[1, "monetary_value"] == pytest.approx(50.0, abs=0.01)

    def test_custom_min_frequency_threshold(self):
        """Test with custom min_frequency threshold."""
        periods = [
            PeriodAggregation(
                "C1",
                datetime(2023, 1, 1),
                datetime(2023, 2, 1),
                2,
                100.0,
                30.0,
                5,
            ),
            PeriodAggregation(
                "C2",
                datetime(2023, 1, 1),
                datetime(2023, 2, 1),
                5,
                250.0,
                75.0,
                12,
            ),
        ]
        # With min_frequency=3, only C2 should be included
        df = prepare_gamma_gamma_inputs(periods, min_frequency=3)
        assert len(df) == 1
        assert df.loc[0, "customer_id"] == "C2"
        assert df.loc[0, "frequency"] == 5

    def test_monetary_value_precision(self):
        """Test monetary value calculation with Decimal precision."""
        periods = [
            PeriodAggregation(
                "C1",
                datetime(2023, 1, 1),
                datetime(2023, 2, 1),
                3,
                100.0,
                30.0,
                6,
            )
        ]
        df = prepare_gamma_gamma_inputs(periods, min_frequency=2)
        # 100.0 / 3 = 33.333... -> should round to 33.33
        assert df.loc[0, "monetary_value"] == pytest.approx(33.33, abs=0.01)

    def test_aggregation_across_multiple_periods(self):
        """Customer with purchases in multiple periods should aggregate correctly."""
        periods = [
            PeriodAggregation(
                "C1",
                datetime(2023, 1, 1),
                datetime(2023, 2, 1),
                2,
                80.0,
                24.0,
                4,
            ),
            PeriodAggregation(
                "C1",
                datetime(2023, 2, 1),
                datetime(2023, 3, 1),
                1,
                40.0,
                12.0,
                2,
            ),
            PeriodAggregation(
                "C1",
                datetime(2023, 3, 1),
                datetime(2023, 4, 1),
                2,
                80.0,
                24.0,
                4,
            ),
        ]
        df = prepare_gamma_gamma_inputs(periods, min_frequency=2)
        assert len(df) == 1
        assert df.loc[0, "frequency"] == 5  # 2 + 1 + 2 = 5
        # Total spend: 80 + 40 + 80 = 200
        # Monetary value: 200 / 5 = 40.0
        assert df.loc[0, "monetary_value"] == pytest.approx(40.0, abs=0.01)


class TestEdgeCases:
    """Test edge cases identified in code review."""

    def test_zero_monetary_value_raises_error(self):
        """Zero monetary value should raise ValueError (Gamma-Gamma requires positive values)."""
        with pytest.raises(
            ValueError, match="Monetary value must be positive \\(>0\\)"
        ):
            GammaGammaInput(
                customer_id="C1", frequency=2, monetary_value=Decimal("0.00")
            )

    def test_very_small_monetary_value_preserves_precision(self):
        """Very small monetary values should maintain decimal precision."""
        periods = [
            PeriodAggregation(
                "C1",
                datetime(2023, 1, 1),
                datetime(2023, 2, 1),
                100,
                1.00,  # 100 orders at $0.01 each
                0.30,
                100,
            )
        ]
        df = prepare_gamma_gamma_inputs(periods, min_frequency=2)
        assert len(df) == 1
        # Monetary value: 1.00 / 100 = 0.01
        assert df.loc[0, "monetary_value"] == pytest.approx(0.01, abs=0.001)

    def test_decimal_precision_preserved_through_rounding(self):
        """Decimal precision should be preserved with ROUND_HALF_UP."""
        periods = [
            PeriodAggregation(
                "C1",
                datetime(2023, 1, 1),
                datetime(2023, 2, 1),
                3,
                10.00,  # 3 orders, $10 total -> $3.33... per order
                3.00,
                6,
            )
        ]
        df = prepare_gamma_gamma_inputs(periods, min_frequency=2)
        assert len(df) == 1
        # Monetary value: 10.00 / 3 = 3.333... -> should round to 3.33
        assert df.loc[0, "monetary_value"] == pytest.approx(3.33, abs=0.01)

    def test_gamma_gamma_input_zero_monetary_value_validation(self):
        """GammaGammaInput should reject zero monetary values."""
        with pytest.raises(ValueError, match="must be positive"):
            GammaGammaInput(
                customer_id="C1", frequency=5, monetary_value=Decimal("0.00")
            )
