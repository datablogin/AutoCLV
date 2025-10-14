"""Tests for model input preparation utilities (BG/NBD and Gamma-Gamma)."""

import time
from datetime import datetime, timedelta, timezone
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

# Timezone-aware datetime helper for tests (Issue #62)
UTC = timezone.utc


def generate_large_dataset(num_customers: int) -> list[PeriodAggregation]:
    """Generate realistic synthetic dataset for performance testing.

    Creates customer purchase history with:
    - 1-5 periods per customer
    - 1-10 orders per period
    - Varying spend amounts ($10-$500 per period)
    - Monthly periods spanning Jan-Dec 2023
    """
    periods = []
    base_date = datetime(2023, 1, 1, tzinfo=UTC)

    for customer_idx in range(num_customers):
        customer_id = f"C{customer_idx:07d}"  # C0000000, C0000001, etc.
        num_periods = (customer_idx % 5) + 1  # 1-5 periods per customer

        for period_idx in range(num_periods):
            period_start = base_date + timedelta(days=30 * period_idx)
            period_end = period_start + timedelta(days=30)

            # Vary orders and spend realistically
            total_orders = ((customer_idx + period_idx) % 10) + 1  # 1-10 orders
            total_spend = float(
                total_orders * (((customer_idx + period_idx) % 50) + 10)
            )  # $10-$500
            total_margin = total_spend * 0.3
            total_quantity = total_orders * 2

            periods.append(
                PeriodAggregation(
                    customer_id,
                    period_start,
                    period_end,
                    total_orders,
                    total_spend,
                    total_margin,
                    total_quantity,
                )
            )

    return periods


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
        df = prepare_bg_nbd_inputs(
            [], datetime(2023, 1, 1, tzinfo=UTC), datetime(2023, 12, 31, tzinfo=UTC)
        )
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
        assert list(df.columns) == ["customer_id", "frequency", "recency", "T"]

    def test_single_customer_single_period(self):
        """Single customer with single period should have frequency=0."""
        periods = [
            PeriodAggregation(
                customer_id="C1",
                period_start=datetime(2023, 1, 1, tzinfo=UTC),
                period_end=datetime(2023, 2, 1, tzinfo=UTC),
                total_orders=1,
                total_spend=100.0,
                total_margin=30.0,
                total_quantity=5,
            )
        ]
        df = prepare_bg_nbd_inputs(
            periods, datetime(2023, 1, 1, tzinfo=UTC), datetime(2023, 6, 1, tzinfo=UTC)
        )
        assert len(df) == 1
        assert df.loc[0, "customer_id"] == "C1"
        assert df.loc[0, "frequency"] == 0  # 1 order - 1 = 0 repeat purchases
        assert df.loc[0, "recency"] == 0.0  # No repeat purchases
        # T = from first period start (2023-01-01) to observation end (2023-06-01)
        expected_T = (
            datetime(2023, 6, 1, tzinfo=UTC) - datetime(2023, 1, 1, tzinfo=UTC)
        ).days
        assert df.loc[0, "T"] == pytest.approx(expected_T, abs=1.0)

    def test_single_customer_multiple_periods(self):
        """Single customer with multiple periods should calculate frequency, recency, T."""
        periods = [
            PeriodAggregation(
                customer_id="C1",
                period_start=datetime(2023, 1, 1, tzinfo=UTC),
                period_end=datetime(2023, 2, 1, tzinfo=UTC),
                total_orders=2,
                total_spend=150.0,
                total_margin=45.0,
                total_quantity=10,
            ),
            PeriodAggregation(
                customer_id="C1",
                period_start=datetime(2023, 3, 1, tzinfo=UTC),
                period_end=datetime(2023, 4, 1, tzinfo=UTC),
                total_orders=1,
                total_spend=50.0,
                total_margin=15.0,
                total_quantity=3,
            ),
        ]
        df = prepare_bg_nbd_inputs(
            periods, datetime(2023, 1, 1, tzinfo=UTC), datetime(2023, 6, 1, tzinfo=UTC)
        )
        assert len(df) == 1
        assert df.loc[0, "customer_id"] == "C1"
        assert df.loc[0, "frequency"] == 2  # 3 total orders - 1 = 2 repeat purchases
        # Recency = from first period start (2023-01-01) to last period end (2023-04-01)
        expected_recency = (
            datetime(2023, 4, 1, tzinfo=UTC) - datetime(2023, 1, 1, tzinfo=UTC)
        ).days
        assert df.loc[0, "recency"] == pytest.approx(expected_recency, abs=1.0)
        # T = from first period start (2023-01-01) to observation end (2023-06-01)
        expected_T = (
            datetime(2023, 6, 1, tzinfo=UTC) - datetime(2023, 1, 1, tzinfo=UTC)
        ).days
        assert df.loc[0, "T"] == pytest.approx(expected_T, abs=1.0)

    def test_multiple_customers(self):
        """Multiple customers should be correctly grouped and sorted."""
        periods = [
            PeriodAggregation(
                "C2",
                datetime(2023, 2, 1, tzinfo=UTC),
                datetime(2023, 3, 1, tzinfo=UTC),
                1,
                75.0,
                20.0,
                3,
            ),
            PeriodAggregation(
                "C1",
                datetime(2023, 1, 1, tzinfo=UTC),
                datetime(2023, 2, 1, tzinfo=UTC),
                2,
                100.0,
                30.0,
                5,
            ),
            PeriodAggregation(
                "C1",
                datetime(2023, 3, 1, tzinfo=UTC),
                datetime(2023, 4, 1, tzinfo=UTC),
                1,
                50.0,
                15.0,
                2,
            ),
        ]
        df = prepare_bg_nbd_inputs(
            periods, datetime(2023, 1, 1, tzinfo=UTC), datetime(2023, 6, 1, tzinfo=UTC)
        )
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
                datetime(2023, 1, 1, tzinfo=UTC),
                datetime(2023, 2, 1, tzinfo=UTC),
                1,
                50.0,
                15.0,
                3,
            ),
            PeriodAggregation(
                "C1",
                datetime(2023, 2, 1, tzinfo=UTC),
                datetime(2023, 3, 1, tzinfo=UTC),
                1,
                75.0,
                20.0,
                4,
            ),
        ]
        df = prepare_bg_nbd_inputs(
            periods, datetime(2023, 1, 1, tzinfo=UTC), datetime(2023, 6, 1, tzinfo=UTC)
        )
        assert len(df) == 1
        assert df.loc[0, "frequency"] == 1  # 2 orders - 1 = 1 repeat purchase

    def test_non_overlapping_periods(self):
        """Customers with non-overlapping activity periods."""
        periods = [
            PeriodAggregation(
                "C1",
                datetime(2023, 1, 1, tzinfo=UTC),
                datetime(2023, 2, 1, tzinfo=UTC),
                2,
                100.0,
                30.0,
                5,
            ),
            PeriodAggregation(
                "C1",
                datetime(2023, 5, 1, tzinfo=UTC),
                datetime(2023, 6, 1, tzinfo=UTC),
                1,
                50.0,
                15.0,
                3,
            ),
        ]
        df = prepare_bg_nbd_inputs(
            periods,
            datetime(2023, 1, 1, tzinfo=UTC),
            datetime(2023, 12, 31, tzinfo=UTC),
        )
        assert len(df) == 1
        assert df.loc[0, "frequency"] == 2  # 3 total orders - 1
        # Recency should span from first period start to last period end
        expected_recency = (
            datetime(2023, 6, 1, tzinfo=UTC) - datetime(2023, 1, 1, tzinfo=UTC)
        ).days
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
                datetime(2023, 1, 1, tzinfo=UTC),
                datetime(2023, 2, 1, tzinfo=UTC),
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
                datetime(2023, 1, 1, tzinfo=UTC),
                datetime(2023, 2, 1, tzinfo=UTC),
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
                datetime(2023, 1, 1, tzinfo=UTC),
                datetime(2023, 2, 1, tzinfo=UTC),
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
                datetime(2023, 1, 1, tzinfo=UTC),
                datetime(2023, 2, 1, tzinfo=UTC),
                3,
                150.0,
                45.0,
                10,
            ),
            # C2: 1 order (excluded)
            PeriodAggregation(
                "C2",
                datetime(2023, 1, 1, tzinfo=UTC),
                datetime(2023, 2, 1, tzinfo=UTC),
                1,
                50.0,
                15.0,
                3,
            ),
            # C3: 5 orders across 2 periods (included)
            PeriodAggregation(
                "C3",
                datetime(2023, 1, 1, tzinfo=UTC),
                datetime(2023, 2, 1, tzinfo=UTC),
                2,
                100.0,
                30.0,
                5,
            ),
            PeriodAggregation(
                "C3",
                datetime(2023, 3, 1, tzinfo=UTC),
                datetime(2023, 4, 1, tzinfo=UTC),
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
                datetime(2023, 1, 1, tzinfo=UTC),
                datetime(2023, 2, 1, tzinfo=UTC),
                2,
                100.0,
                30.0,
                5,
            ),
            PeriodAggregation(
                "C2",
                datetime(2023, 1, 1, tzinfo=UTC),
                datetime(2023, 2, 1, tzinfo=UTC),
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
                datetime(2023, 1, 1, tzinfo=UTC),
                datetime(2023, 2, 1, tzinfo=UTC),
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
                datetime(2023, 1, 1, tzinfo=UTC),
                datetime(2023, 2, 1, tzinfo=UTC),
                2,
                80.0,
                24.0,
                4,
            ),
            PeriodAggregation(
                "C1",
                datetime(2023, 2, 1, tzinfo=UTC),
                datetime(2023, 3, 1, tzinfo=UTC),
                1,
                40.0,
                12.0,
                2,
            ),
            PeriodAggregation(
                "C1",
                datetime(2023, 3, 1, tzinfo=UTC),
                datetime(2023, 4, 1, tzinfo=UTC),
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
                datetime(2023, 1, 1, tzinfo=UTC),
                datetime(2023, 2, 1, tzinfo=UTC),
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
                datetime(2023, 1, 1, tzinfo=UTC),
                datetime(2023, 2, 1, tzinfo=UTC),
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


class TestInputDataValidation:
    """Test validation of negative values, boundary conditions, and parameter validation (Issue #61)."""

    def test_negative_total_orders_in_bg_nbd(self):
        """prepare_bg_nbd_inputs should reject negative total_orders."""
        periods = [
            PeriodAggregation(
                "C1",
                datetime(2023, 1, 1, tzinfo=UTC),
                datetime(2023, 2, 1, tzinfo=UTC),
                -1,  # Invalid negative total_orders
                100.0,
                30.0,
                5,
            )
        ]
        with pytest.raises(
            ValueError, match="Invalid total_orders.*must be non-negative"
        ):
            prepare_bg_nbd_inputs(
                periods,
                datetime(2023, 1, 1, tzinfo=UTC),
                datetime(2023, 6, 1, tzinfo=UTC),
            )

    def test_negative_total_spend_in_bg_nbd(self):
        """prepare_bg_nbd_inputs should reject negative total_spend."""
        periods = [
            PeriodAggregation(
                "C1",
                datetime(2023, 1, 1, tzinfo=UTC),
                datetime(2023, 2, 1, tzinfo=UTC),
                1,
                -100.0,  # Invalid negative total_spend
                -30.0,
                5,
            )
        ]
        with pytest.raises(
            ValueError, match="Invalid total_spend.*must be non-negative"
        ):
            prepare_bg_nbd_inputs(
                periods,
                datetime(2023, 1, 1, tzinfo=UTC),
                datetime(2023, 6, 1, tzinfo=UTC),
            )

    def test_negative_total_orders_in_gamma_gamma(self):
        """prepare_gamma_gamma_inputs should reject negative total_orders."""
        periods = [
            PeriodAggregation(
                "C1",
                datetime(2023, 1, 1, tzinfo=UTC),
                datetime(2023, 2, 1, tzinfo=UTC),
                -5,  # Invalid negative total_orders
                100.0,
                30.0,
                10,
            )
        ]
        with pytest.raises(
            ValueError, match="Invalid total_orders.*must be non-negative"
        ):
            prepare_gamma_gamma_inputs(periods, min_frequency=2)

    def test_negative_total_spend_in_gamma_gamma(self):
        """prepare_gamma_gamma_inputs should reject negative total_spend."""
        periods = [
            PeriodAggregation(
                "C1",
                datetime(2023, 1, 1, tzinfo=UTC),
                datetime(2023, 2, 1, tzinfo=UTC),
                5,
                -200.0,  # Invalid negative total_spend
                -60.0,
                10,
            )
        ]
        with pytest.raises(
            ValueError, match="Invalid total_spend.*must be non-negative"
        ):
            prepare_gamma_gamma_inputs(periods, min_frequency=2)

    def test_zero_frequency_division_check(self):
        """min_frequency=0 should be caught at function entry (prevents division by zero)."""
        # This test verifies that invalid min_frequency is caught early at function entry,
        # which prevents division by zero downstream. This is better than catching it later.
        # Updated in response to Claude Review recommendation to validate min_frequency early.
        periods = [
            PeriodAggregation(
                "C1",
                datetime(2023, 1, 1, tzinfo=UTC),
                datetime(2023, 2, 1, tzinfo=UTC),
                0,  # Zero orders
                0.0,
                0.0,
                0,
            )
        ]
        # Now caught at function entry instead of during division
        with pytest.raises(ValueError, match="min_frequency must be >= 1"):
            prepare_gamma_gamma_inputs(periods, min_frequency=0)

    def test_zero_values_are_valid(self):
        """Zero total_orders and total_spend should be valid (not negative)."""
        # BG/NBD: Zero orders is valid (results in frequency=0)
        periods = [
            PeriodAggregation(
                "C1",
                datetime(2023, 1, 1, tzinfo=UTC),
                datetime(2023, 2, 1, tzinfo=UTC),
                0,  # Zero orders is valid
                0.0,  # Zero spend is valid
                0.0,
                0,
            )
        ]
        df = prepare_bg_nbd_inputs(
            periods, datetime(2023, 1, 1, tzinfo=UTC), datetime(2023, 6, 1, tzinfo=UTC)
        )
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
                datetime(2022, 12, 1, tzinfo=UTC),  # Before observation_start
                datetime(2023, 1, 15, tzinfo=UTC),
                1,
                100.0,
                30.0,
                5,
            )
        ]
        with pytest.raises(
            ValueError, match="Period start.*is before.*observation_start"
        ):
            prepare_bg_nbd_inputs(
                periods,
                datetime(2023, 1, 1, tzinfo=UTC),
                datetime(2023, 6, 1, tzinfo=UTC),
            )

    def test_period_after_observation_end(self):
        """prepare_bg_nbd_inputs should reject periods ending after observation_end."""
        periods = [
            PeriodAggregation(
                "C1",
                datetime(2023, 5, 1, tzinfo=UTC),
                datetime(2023, 7, 1, tzinfo=UTC),  # After observation_end
                1,
                100.0,
                30.0,
                5,
            )
        ]
        with pytest.raises(ValueError, match="Period end.*is after.*observation_end"):
            prepare_bg_nbd_inputs(
                periods,
                datetime(2023, 1, 1, tzinfo=UTC),
                datetime(2023, 6, 1, tzinfo=UTC),
            )

    def test_period_exactly_at_boundaries(self):
        """Periods exactly at observation window boundaries should be valid."""
        periods = [
            PeriodAggregation(
                "C1",
                datetime(2023, 1, 1, tzinfo=UTC),  # Exactly at observation_start
                datetime(2023, 2, 1, tzinfo=UTC),
                1,
                100.0,
                30.0,
                5,
            ),
            PeriodAggregation(
                "C2",
                datetime(2023, 5, 1, tzinfo=UTC),
                datetime(2023, 6, 1, tzinfo=UTC),  # Exactly at observation_end
                1,
                50.0,
                15.0,
                3,
            ),
        ]
        df = prepare_bg_nbd_inputs(
            periods, datetime(2023, 1, 1, tzinfo=UTC), datetime(2023, 6, 1, tzinfo=UTC)
        )
        assert len(df) == 2  # Both should be valid

    def test_decimal_type_preserved_in_dataframe(self):
        """monetary_value should be Decimal type in DataFrame (Issue #3)."""
        periods = [
            PeriodAggregation(
                "C1",
                datetime(2023, 1, 1, tzinfo=UTC),
                datetime(2023, 2, 1, tzinfo=UTC),
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
                datetime(2023, 1, 1, tzinfo=UTC),
                datetime(2023, 2, 1, tzinfo=UTC),
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
                datetime(2023, 1, 1, tzinfo=UTC),
                datetime(2023, 2, 1, tzinfo=UTC),
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

    def test_min_frequency_validation(self):
        """prepare_gamma_gamma_inputs should reject min_frequency < 1."""
        periods = [
            PeriodAggregation(
                "C1",
                datetime(2023, 1, 1, tzinfo=UTC),
                datetime(2023, 2, 1, tzinfo=UTC),
                3,
                150.0,
                45.0,
                10,
            )
        ]
        # min_frequency must be >= 1
        with pytest.raises(ValueError, match="min_frequency must be >= 1"):
            prepare_gamma_gamma_inputs(periods, min_frequency=0)

        with pytest.raises(ValueError, match="min_frequency must be >= 1"):
            prepare_gamma_gamma_inputs(periods, min_frequency=-1)


class TestTimezoneValidation:
    """Test timezone validation added for Issue #62."""

    def test_observation_end_must_be_timezone_aware(self):
        """prepare_bg_nbd_inputs should reject timezone-naive observation_end."""
        from datetime import timezone

        periods = [
            PeriodAggregation(
                "C1",
                datetime(2023, 1, 1, tzinfo=timezone.utc),
                datetime(2023, 2, 1, tzinfo=timezone.utc),
                1,
                100.0,
                30.0,
                5,
            )
        ]
        # observation_end is timezone-naive (no tzinfo)
        with pytest.raises(ValueError, match="observation_end must be timezone-aware"):
            prepare_bg_nbd_inputs(
                periods,
                datetime(2023, 1, 1, tzinfo=timezone.utc),
                datetime(2023, 6, 1),  # Missing tzinfo
            )

    def test_observation_start_must_be_timezone_aware(self):
        """prepare_bg_nbd_inputs should reject timezone-naive observation_start."""
        from datetime import timezone

        periods = [
            PeriodAggregation(
                "C1",
                datetime(2023, 1, 1, tzinfo=timezone.utc),
                datetime(2023, 2, 1, tzinfo=timezone.utc),
                1,
                100.0,
                30.0,
                5,
            )
        ]
        # observation_start is timezone-naive (no tzinfo)
        with pytest.raises(
            ValueError, match="observation_start must be timezone-aware"
        ):
            prepare_bg_nbd_inputs(
                periods,
                datetime(2023, 1, 1),  # Missing tzinfo
                datetime(2023, 6, 1, tzinfo=timezone.utc),
            )

    def test_period_start_must_be_timezone_aware(self):
        """prepare_bg_nbd_inputs should reject timezone-naive period dates."""
        from datetime import timezone

        periods = [
            PeriodAggregation(
                "C1",
                datetime(2023, 1, 1),  # Missing tzinfo
                datetime(2023, 2, 1, tzinfo=timezone.utc),
                1,
                100.0,
                30.0,
                5,
            )
        ]
        with pytest.raises(
            ValueError, match="Period start must be timezone-aware for customer C1"
        ):
            prepare_bg_nbd_inputs(
                periods,
                datetime(2023, 1, 1, tzinfo=timezone.utc),
                datetime(2023, 6, 1, tzinfo=timezone.utc),
            )

    def test_period_end_must_be_timezone_aware(self):
        """prepare_bg_nbd_inputs should reject timezone-naive period dates."""
        from datetime import timezone

        periods = [
            PeriodAggregation(
                "C1",
                datetime(2023, 1, 1, tzinfo=timezone.utc),
                datetime(2023, 2, 1),  # Missing tzinfo
                1,
                100.0,
                30.0,
                5,
            )
        ]
        with pytest.raises(
            ValueError, match="Period end must be timezone-aware for customer C1"
        ):
            prepare_bg_nbd_inputs(
                periods,
                datetime(2023, 1, 1, tzinfo=timezone.utc),
                datetime(2023, 6, 1, tzinfo=timezone.utc),
            )

    def test_inconsistent_timezones_rejected(self):
        """prepare_bg_nbd_inputs should reject inconsistent timezones."""
        from datetime import timezone
        from zoneinfo import ZoneInfo

        periods = [
            PeriodAggregation(
                "C1",
                datetime(2023, 1, 1, tzinfo=ZoneInfo("America/New_York")),
                datetime(2023, 2, 1, tzinfo=ZoneInfo("America/New_York")),
                1,
                100.0,
                30.0,
                5,
            )
        ]
        # observation dates use UTC, periods use America/New_York
        with pytest.raises(ValueError, match="Inconsistent timezones"):
            prepare_bg_nbd_inputs(
                periods,
                datetime(2023, 1, 1, tzinfo=timezone.utc),
                datetime(2023, 6, 1, tzinfo=timezone.utc),
            )

    def test_timezone_aware_datetimes_accepted(self):
        """prepare_bg_nbd_inputs should accept timezone-aware datetimes."""
        from datetime import timezone

        periods = [
            PeriodAggregation(
                "C1",
                datetime(2023, 1, 1, tzinfo=timezone.utc),
                datetime(2023, 2, 1, tzinfo=timezone.utc),
                2,
                100.0,
                30.0,
                5,
            ),
            PeriodAggregation(
                "C1",
                datetime(2023, 3, 1, tzinfo=timezone.utc),
                datetime(2023, 4, 1, tzinfo=timezone.utc),
                1,
                50.0,
                15.0,
                3,
            ),
        ]
        df = prepare_bg_nbd_inputs(
            periods,
            datetime(2023, 1, 1, tzinfo=timezone.utc),
            datetime(2023, 6, 1, tzinfo=timezone.utc),
        )
        # Should succeed and produce valid results
        assert len(df) == 1
        assert df.loc[0, "customer_id"] == "C1"
        assert df.loc[0, "frequency"] == 2  # 3 orders - 1

    def test_equivalent_utc_timezones_accepted(self):
        """prepare_bg_nbd_inputs should accept equivalent UTC timezones (Issue #62 Claude Review)."""
        from datetime import timezone
        from zoneinfo import ZoneInfo

        # Test that datetime.timezone.utc and ZoneInfo("UTC") are accepted as equivalent
        periods = [
            PeriodAggregation(
                "C1",
                datetime(2023, 1, 1, tzinfo=ZoneInfo("UTC")),
                datetime(2023, 2, 1, tzinfo=ZoneInfo("UTC")),
                2,
                100.0,
                30.0,
                5,
            ),
            PeriodAggregation(
                "C1",
                datetime(2023, 3, 1, tzinfo=ZoneInfo("UTC")),
                datetime(2023, 4, 1, tzinfo=ZoneInfo("UTC")),
                1,
                50.0,
                15.0,
                3,
            ),
        ]
        # Should not fail even though ZoneInfo("UTC") != datetime.timezone.utc by identity
        # but they have the same UTC offset (0:00:00)
        df = prepare_bg_nbd_inputs(
            periods,
            datetime(2023, 1, 1, tzinfo=timezone.utc),
            datetime(2023, 6, 1, tzinfo=timezone.utc),
        )
        assert len(df) == 1
        assert df.loc[0, "customer_id"] == "C1"
        assert df.loc[0, "frequency"] == 2  # 3 orders - 1


class TestOverlappingPeriods:
    """Test behavior with overlapping periods (Issue #64.14)."""

    def test_overlapping_periods_aggregate_correctly(self):
        """Overlapping periods for same customer should aggregate total orders."""
        # Scenario: Customer has overlapping activity periods (e.g., subscription
        # and one-time purchases tracked separately but overlapping in time)
        periods = [
            PeriodAggregation(
                "C1",
                datetime(2023, 1, 1, tzinfo=UTC),
                datetime(2023, 2, 1, tzinfo=UTC),
                2,
                100.0,
                30.0,
                5,
            ),
            PeriodAggregation(
                "C1",
                datetime(2023, 1, 15, tzinfo=UTC),  # Overlaps with previous period
                datetime(2023, 2, 15, tzinfo=UTC),
                1,
                50.0,
                15.0,
                2,
            ),
        ]
        # Expected behavior: Sum orders, use earliest start and latest end
        df = prepare_bg_nbd_inputs(
            periods, datetime(2023, 1, 1, tzinfo=UTC), datetime(2023, 6, 1, tzinfo=UTC)
        )
        assert len(df) == 1
        assert df.loc[0, "customer_id"] == "C1"
        # Total orders: 2 + 1 = 3, frequency = 3 - 1 = 2
        assert df.loc[0, "frequency"] == 2
        # Recency: from 2023-01-01 (earliest start) to 2023-02-15 (latest end)
        expected_recency = (
            datetime(2023, 2, 15, tzinfo=UTC) - datetime(2023, 1, 1, tzinfo=UTC)
        ).days
        assert df.loc[0, "recency"] == pytest.approx(expected_recency, abs=1.0)

    def test_overlapping_periods_gamma_gamma(self):
        """Overlapping periods should aggregate spend and orders for Gamma-Gamma."""
        periods = [
            PeriodAggregation(
                "C1",
                datetime(2023, 1, 1, tzinfo=UTC),
                datetime(2023, 2, 1, tzinfo=UTC),
                2,
                80.0,  # $40 per order
                24.0,
                5,
            ),
            PeriodAggregation(
                "C1",
                datetime(2023, 1, 15, tzinfo=UTC),
                datetime(2023, 2, 15, tzinfo=UTC),
                1,
                60.0,  # $60 per order
                18.0,
                2,
            ),
        ]
        df = prepare_gamma_gamma_inputs(periods, min_frequency=2)
        assert len(df) == 1
        assert df.loc[0, "customer_id"] == "C1"
        assert df.loc[0, "frequency"] == 3  # 2 + 1 = 3 orders
        # Monetary value: (80 + 60) / 3 = 140 / 3 = 46.67 (rounded)
        assert df.loc[0, "monetary_value"] == Decimal("46.67")

    def test_completely_overlapping_periods(self):
        """Completely overlapping periods (same dates) should aggregate correctly."""
        # Edge case: Two records for same period (e.g., from different data sources)
        periods = [
            PeriodAggregation(
                "C1",
                datetime(2023, 1, 1, tzinfo=UTC),
                datetime(2023, 2, 1, tzinfo=UTC),
                1,
                50.0,
                15.0,
                3,
            ),
            PeriodAggregation(
                "C1",
                datetime(2023, 1, 1, tzinfo=UTC),  # Exact same dates
                datetime(2023, 2, 1, tzinfo=UTC),
                1,
                50.0,
                15.0,
                3,
            ),
        ]
        df = prepare_bg_nbd_inputs(
            periods, datetime(2023, 1, 1, tzinfo=UTC), datetime(2023, 6, 1, tzinfo=UTC)
        )
        assert len(df) == 1
        assert df.loc[0, "frequency"] == 1  # 2 orders - 1 = 1 repeat purchase
        # Recency: from 2023-01-01 to 2023-02-01 (same start/end)
        expected_recency = (
            datetime(2023, 2, 1, tzinfo=UTC) - datetime(2023, 1, 1, tzinfo=UTC)
        ).days
        assert df.loc[0, "recency"] == pytest.approx(expected_recency, abs=1.0)


@pytest.mark.slow
class TestPerformance:
    """Performance benchmarks with large datasets (Issue #64.13)."""

    def test_bg_nbd_performance_100k_customers(self):
        """Verify acceptable performance with 100k customers (Issue #64.13)."""
        num_customers = 100_000
        periods = generate_large_dataset(num_customers)

        start = time.time()
        df = prepare_bg_nbd_inputs(
            periods,
            datetime(2023, 1, 1, tzinfo=UTC),
            datetime(2023, 12, 31, tzinfo=UTC),
        )
        duration = time.time() - start

        # Verify correctness
        assert len(df) == num_customers
        assert list(df.columns) == ["customer_id", "frequency", "recency", "T"]
        assert df["frequency"].min() >= 0
        assert df["recency"].min() >= 0
        assert df["T"].min() > 0

        # Performance assertion: should complete in < 5 seconds
        # Note: This is a guideline from Issue #63.10. Adjust if needed based on hardware.
        assert duration < 5.0, f"Took {duration:.2f}s (expected < 5s)"
        print(f"\n✓ BG/NBD 100k customers: {duration:.2f}s")

    def test_gamma_gamma_performance_100k_customers(self):
        """Verify acceptable performance with 100k customers (Issue #64.13)."""
        num_customers = 100_000
        periods = generate_large_dataset(num_customers)

        start = time.time()
        df = prepare_gamma_gamma_inputs(periods, min_frequency=2)
        duration = time.time() - start

        # Verify correctness
        assert len(df) > 0  # Many customers should have frequency >= 2
        assert list(df.columns) == ["customer_id", "frequency", "monetary_value"]
        assert df["frequency"].min() >= 2
        assert all(isinstance(v, Decimal) for v in df["monetary_value"])

        # Performance assertion: should complete in < 5 seconds
        assert duration < 5.0, f"Took {duration:.2f}s (expected < 5s)"
        print(f"\n✓ Gamma-Gamma 100k customers: {duration:.2f}s")

    @pytest.mark.skipif(
        True, reason="Skip 500k benchmark by default (runs in CI with --run-slow)"
    )
    def test_bg_nbd_performance_500k_customers(self):
        """Benchmark with 500k customers (Issue #63.10)."""
        num_customers = 500_000
        periods = generate_large_dataset(num_customers)

        start = time.time()
        df = prepare_bg_nbd_inputs(
            periods,
            datetime(2023, 1, 1, tzinfo=UTC),
            datetime(2023, 12, 31, tzinfo=UTC),
        )
        duration = time.time() - start

        assert len(df) == num_customers
        print(f"\n✓ BG/NBD 500k customers: {duration:.2f}s")

    @pytest.mark.skipif(
        True, reason="Skip 1M benchmark by default (runs in CI with --run-slow)"
    )
    def test_bg_nbd_performance_1m_customers(self):
        """Benchmark with 1M customers (Issue #63.10)."""
        num_customers = 1_000_000
        periods = generate_large_dataset(num_customers)

        start = time.time()
        df = prepare_bg_nbd_inputs(
            periods,
            datetime(2023, 1, 1, tzinfo=UTC),
            datetime(2023, 12, 31, tzinfo=UTC),
        )
        duration = time.time() - start

        assert len(df) == num_customers
        print(f"\n✓ BG/NBD 1M customers: {duration:.2f}s")
