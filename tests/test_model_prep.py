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

    def test_recency_exceeds_T_is_capped(self):
        """Recency exceeding T should be capped at T (Issue #4)."""
        input_data = BGNBDInput(customer_id="C1", frequency=5, recency=100.0, T=90.0)
        # recency should be capped at T
        assert input_data.recency == 90.0
        assert input_data.T == 90.0

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
        assert df.loc[0, "monetary_value"] == Decimal("50.00")

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
        assert df.loc[0, "monetary_value"] == Decimal("50.00")

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
        assert df.loc[0, "monetary_value"] == Decimal("50.00")

        # C3: 5 orders, 250.0 total spend -> 50.0 avg
        assert df.loc[1, "frequency"] == 5
        assert df.loc[1, "monetary_value"] == Decimal("50.00")

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
        assert df.loc[0, "monetary_value"] == Decimal("33.33")

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
        assert df.loc[0, "monetary_value"] == Decimal("40.00")


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
        assert df.loc[0, "monetary_value"] == Decimal("0.01")

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
        assert df.loc[0, "monetary_value"] == Decimal("3.33")

    def test_gamma_gamma_input_zero_monetary_value_validation(self):
        """GammaGammaInput should reject zero monetary values."""
        with pytest.raises(ValueError, match="must be positive"):
            GammaGammaInput(
                customer_id="C1", frequency=5, monetary_value=Decimal("0.00")
            )


class TestIssue61Validations:
    """Test validations added for Issue #61."""

    def test_negative_total_orders_in_bg_nbd(self):
        """prepare_bg_nbd_inputs should reject negative total_orders."""
        periods = [
            PeriodAggregation(
                "C1",
                datetime(2023, 1, 1),
                datetime(2023, 2, 1),
                -1,  # Invalid negative total_orders
                100.0,
                30.0,
                5,
            )
        ]
        with pytest.raises(ValueError, match="Invalid total_orders.*must be non-negative"):
            prepare_bg_nbd_inputs(periods, datetime(2023, 1, 1), datetime(2023, 6, 1))

    def test_negative_total_spend_in_bg_nbd(self):
        """prepare_bg_nbd_inputs should reject negative total_spend."""
        periods = [
            PeriodAggregation(
                "C1",
                datetime(2023, 1, 1),
                datetime(2023, 2, 1),
                1,
                -100.0,  # Invalid negative total_spend
                -30.0,
                5,
            )
        ]
        with pytest.raises(ValueError, match="Invalid total_spend.*must be non-negative"):
            prepare_bg_nbd_inputs(periods, datetime(2023, 1, 1), datetime(2023, 6, 1))

    def test_negative_total_orders_in_gamma_gamma(self):
        """prepare_gamma_gamma_inputs should reject negative total_orders."""
        periods = [
            PeriodAggregation(
                "C1",
                datetime(2023, 1, 1),
                datetime(2023, 2, 1),
                -5,  # Invalid negative total_orders
                100.0,
                30.0,
                10,
            )
        ]
        with pytest.raises(ValueError, match="Invalid total_orders.*must be non-negative"):
            prepare_gamma_gamma_inputs(periods, min_frequency=2)

    def test_negative_total_spend_in_gamma_gamma(self):
        """prepare_gamma_gamma_inputs should reject negative total_spend."""
        periods = [
            PeriodAggregation(
                "C1",
                datetime(2023, 1, 1),
                datetime(2023, 2, 1),
                5,
                -200.0,  # Invalid negative total_spend
                -60.0,
                10,
            )
        ]
        with pytest.raises(ValueError, match="Invalid total_spend.*must be non-negative"):
            prepare_gamma_gamma_inputs(periods, min_frequency=2)

    def test_zero_frequency_division_check(self):
        """Defensive check: division by zero should be caught (though prevented by design)."""
        # This test verifies the defensive programming in prepare_gamma_gamma_inputs
        # With min_frequency=0 (unusual but technically possible), a customer with 0 orders
        # would pass the frequency < min_frequency check and trigger the division check
        periods = [
            PeriodAggregation(
                "C1",
                datetime(2023, 1, 1),
                datetime(2023, 2, 1),
                0,  # Zero orders
                0.0,
                0.0,
                0,
            )
        ]
        with pytest.raises(ValueError, match="Cannot calculate monetary value with zero frequency"):
            prepare_gamma_gamma_inputs(periods, min_frequency=0)

    def test_zero_values_are_valid(self):
        """Zero total_orders and total_spend should be valid (not negative)."""
        # BG/NBD: Zero orders is valid (results in frequency=0)
        periods = [
            PeriodAggregation(
                "C1",
                datetime(2023, 1, 1),
                datetime(2023, 2, 1),
                0,  # Zero orders is valid
                0.0,  # Zero spend is valid
                0.0,
                0,
            )
        ]
        df = prepare_bg_nbd_inputs(periods, datetime(2023, 1, 1), datetime(2023, 6, 1))
        assert len(df) == 1
        assert df.loc[0, "frequency"] == 0  # 0 - 1 = -1, but max(0, -1) = 0

        # Gamma-Gamma: Zero orders will be excluded by min_frequency filter
        df_gg = prepare_gamma_gamma_inputs(periods, min_frequency=2)
        assert len(df_gg) == 0  # Excluded by min_frequency

    def test_period_before_observation_start(self):
        """prepare_bg_nbd_inputs should reject periods starting before observation_start."""
        periods = [
            PeriodAggregation(
                "C1",
                datetime(2022, 12, 1),  # Before observation_start
                datetime(2023, 1, 15),
                1,
                100.0,
                30.0,
                5,
            )
        ]
        with pytest.raises(ValueError, match="Period start.*is before.*observation_start"):
            prepare_bg_nbd_inputs(periods, datetime(2023, 1, 1), datetime(2023, 6, 1))

    def test_period_after_observation_end(self):
        """prepare_bg_nbd_inputs should reject periods ending after observation_end."""
        periods = [
            PeriodAggregation(
                "C1",
                datetime(2023, 5, 1),
                datetime(2023, 7, 1),  # After observation_end
                1,
                100.0,
                30.0,
                5,
            )
        ]
        with pytest.raises(ValueError, match="Period end.*is after.*observation_end"):
            prepare_bg_nbd_inputs(periods, datetime(2023, 1, 1), datetime(2023, 6, 1))

    def test_period_exactly_at_boundaries(self):
        """Periods exactly at observation window boundaries should be valid."""
        periods = [
            PeriodAggregation(
                "C1",
                datetime(2023, 1, 1),  # Exactly at observation_start
                datetime(2023, 2, 1),
                1,
                100.0,
                30.0,
                5,
            ),
            PeriodAggregation(
                "C2",
                datetime(2023, 5, 1),
                datetime(2023, 6, 1),  # Exactly at observation_end
                1,
                50.0,
                15.0,
                3,
            ),
        ]
        df = prepare_bg_nbd_inputs(periods, datetime(2023, 1, 1), datetime(2023, 6, 1))
        assert len(df) == 2  # Both should be valid

    def test_decimal_type_preserved_in_dataframe(self):
        """monetary_value should be Decimal type in DataFrame (Issue #3)."""
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

        # Verify monetary_value is Decimal type, not float
        monetary_value = df.loc[0, "monetary_value"]
        assert isinstance(monetary_value, Decimal)
        assert monetary_value == Decimal("50.00")

        # Verify column dtype is 'object' (which holds Decimal)
        assert df["monetary_value"].dtype == "object"

    def test_decimal_precision_not_lost(self):
        """High-precision Decimal values should not lose precision (Issue #3)."""
        periods = [
            PeriodAggregation(
                "C1",
                datetime(2023, 1, 1),
                datetime(2023, 2, 1),
                7,
                123.456789,  # High precision input
                37.0,
                20,
            )
        ]
        df = prepare_gamma_gamma_inputs(periods, min_frequency=2)
        assert len(df) == 1

        # Should round to 2 decimal places: 123.456789 / 7 = 17.636827... -> 17.64
        monetary_value = df.loc[0, "monetary_value"]
        assert isinstance(monetary_value, Decimal)
        assert monetary_value == Decimal("17.64")

    def test_dataframe_to_gamma_gamma_input_roundtrip(self):
        """DataFrame → GammaGammaInput should work with Decimal values (Issue #3)."""
        periods = [
            PeriodAggregation(
                "C1",
                datetime(2023, 1, 1),
                datetime(2023, 2, 1),
                5,
                250.0,
                75.0,
                15,
            )
        ]
        df = prepare_gamma_gamma_inputs(periods, min_frequency=2)

        # Create GammaGammaInput from DataFrame row
        row = df.iloc[0]
        gamma_input = GammaGammaInput(
            customer_id=row["customer_id"],
            frequency=row["frequency"],
            monetary_value=row["monetary_value"],  # Should be Decimal
        )

        # Verify successful creation
        assert gamma_input.customer_id == "C1"
        assert gamma_input.frequency == 5
        assert gamma_input.monetary_value == Decimal("50.00")

    def test_recency_slightly_exceeds_T(self):
        """Small recency > T (e.g., 0.1 days) should be capped (Issue #4)."""
        input_data = BGNBDInput(customer_id="C1", frequency=5, recency=90.1, T=90.0)
        assert input_data.recency == 90.0
        assert input_data.T == 90.0

    def test_recency_significantly_exceeds_T(self):
        """Large recency > T should still be capped (Issue #4)."""
        input_data = BGNBDInput(customer_id="C1", frequency=5, recency=200.0, T=90.0)
        assert input_data.recency == 90.0
        assert input_data.T == 90.0

    def test_recency_capping_preserves_other_fields(self):
        """Recency capping should not affect other fields (Issue #4)."""
        input_data = BGNBDInput(customer_id="C1", frequency=5, recency=100.0, T=90.0)
        assert input_data.customer_id == "C1"
        assert input_data.frequency == 5
        assert input_data.recency == 90.0  # Capped
        assert input_data.T == 90.0
