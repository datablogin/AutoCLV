"""Tests for Lens 1: Single Period Analysis."""

from datetime import datetime
from decimal import Decimal

import pytest

from customer_base_audit.analyses.lens1 import (
    Lens1Metrics,
    analyze_single_period,
    calculate_revenue_concentration,
)
from customer_base_audit.foundation.rfm import RFMMetrics, RFMScore


class TestLens1Metrics:
    """Test Lens1Metrics dataclass validation."""

    def test_valid_lens1_metrics(self):
        """Valid Lens 1 metrics should be created successfully."""
        metrics = Lens1Metrics(
            total_customers=100,
            one_time_buyers=30,
            one_time_buyer_pct=Decimal("30.00"),
            total_revenue=Decimal("10000.00"),
            top_10pct_revenue_contribution=Decimal("45.0"),
            top_20pct_revenue_contribution=Decimal("65.0"),
            avg_orders_per_customer=Decimal("3.50"),
            median_customer_value=Decimal("75.00"),
            rfm_distribution={"555": 5, "111": 10},
        )
        assert metrics.total_customers == 100
        assert metrics.one_time_buyers == 30

    def test_negative_total_customers_raises_error(self):
        """Negative total customers should raise ValueError."""
        with pytest.raises(ValueError, match="Total customers cannot be negative"):
            Lens1Metrics(
                total_customers=-1,
                one_time_buyers=0,
                one_time_buyer_pct=Decimal("0"),
                total_revenue=Decimal("0"),
                top_10pct_revenue_contribution=Decimal("0"),
                top_20pct_revenue_contribution=Decimal("0"),
                avg_orders_per_customer=Decimal("0"),
                median_customer_value=Decimal("0"),
                rfm_distribution={},
            )

    def test_one_time_buyers_exceeding_total_raises_error(self):
        """One-time buyers exceeding total customers should raise ValueError."""
        with pytest.raises(
            ValueError, match="One-time buyers .* cannot exceed total customers"
        ):
            Lens1Metrics(
                total_customers=10,
                one_time_buyers=15,
                one_time_buyer_pct=Decimal("150.00"),
                total_revenue=Decimal("1000.00"),
                top_10pct_revenue_contribution=Decimal("50.0"),
                top_20pct_revenue_contribution=Decimal("70.0"),
                avg_orders_per_customer=Decimal("2.00"),
                median_customer_value=Decimal("50.00"),
                rfm_distribution={},
            )

    def test_invalid_one_time_buyer_pct_raises_error(self):
        """One-time buyer percentage outside 0-100 should raise ValueError."""
        with pytest.raises(ValueError, match="One-time buyer percentage must be 0-100"):
            Lens1Metrics(
                total_customers=10,
                one_time_buyers=3,
                one_time_buyer_pct=Decimal("150.00"),
                total_revenue=Decimal("1000.00"),
                top_10pct_revenue_contribution=Decimal("50.0"),
                top_20pct_revenue_contribution=Decimal("70.0"),
                avg_orders_per_customer=Decimal("2.00"),
                median_customer_value=Decimal("50.00"),
                rfm_distribution={},
            )

    def test_invalid_revenue_contribution_raises_error(self):
        """Revenue contribution percentage outside 0-100 should raise ValueError."""
        with pytest.raises(
            ValueError, match="Top 10% revenue contribution must be 0-100"
        ):
            Lens1Metrics(
                total_customers=10,
                one_time_buyers=3,
                one_time_buyer_pct=Decimal("30.00"),
                total_revenue=Decimal("1000.00"),
                top_10pct_revenue_contribution=Decimal("150.0"),
                top_20pct_revenue_contribution=Decimal("70.0"),
                avg_orders_per_customer=Decimal("2.00"),
                median_customer_value=Decimal("50.00"),
                rfm_distribution={},
            )


class TestAnalyzeSinglePeriod:
    """Test analyze_single_period function."""

    def test_empty_input_returns_zero_metrics(self):
        """Empty RFM metrics should return zero-valued Lens 1 metrics."""
        result = analyze_single_period([])
        assert result.total_customers == 0
        assert result.one_time_buyers == 0
        assert result.one_time_buyer_pct == Decimal("0")
        assert result.total_revenue == Decimal("0")

    def test_single_customer_analysis(self):
        """Analyze single customer."""
        metrics = [
            RFMMetrics(
                "C1",
                10,
                5,
                Decimal("50.00"),
                datetime(2023, 1, 1),
                datetime(2023, 12, 31),
                Decimal("250.00"),
            )
        ]
        result = analyze_single_period(metrics)

        assert result.total_customers == 1
        assert result.one_time_buyers == 0  # Has 5 orders
        assert result.one_time_buyer_pct == Decimal("0.00")
        assert result.total_revenue == Decimal("250.00")
        assert result.avg_orders_per_customer == Decimal("5.00")
        assert result.median_customer_value == Decimal("250.00")

    def test_multiple_customers_with_one_time_buyers(self):
        """Analyze multiple customers including one-time buyers."""
        metrics = [
            RFMMetrics(
                "C1",
                10,
                5,
                Decimal("50.00"),
                datetime(2023, 1, 1),
                datetime(2023, 12, 31),
                Decimal("250.00"),
            ),
            RFMMetrics(
                "C2",
                30,
                1,
                Decimal("100.00"),
                datetime(2023, 1, 1),
                datetime(2023, 12, 31),
                Decimal("100.00"),
            ),
            RFMMetrics(
                "C3",
                5,
                10,
                Decimal("75.00"),
                datetime(2023, 1, 1),
                datetime(2023, 12, 31),
                Decimal("750.00"),
            ),
        ]
        result = analyze_single_period(metrics)

        assert result.total_customers == 3
        assert result.one_time_buyers == 1  # C2
        assert result.one_time_buyer_pct == Decimal("33.33")  # 1/3 * 100
        assert result.total_revenue == Decimal("1100.00")  # 250 + 100 + 750
        assert result.avg_orders_per_customer == Decimal("5.33")  # (5+1+10)/3

    def test_median_customer_value_even_count(self):
        """Median calculation with even number of customers."""
        metrics = [
            RFMMetrics(
                "C1",
                10,
                1,
                Decimal("100.00"),
                datetime(2023, 1, 1),
                datetime(2023, 12, 31),
                Decimal("100.00"),
            ),
            RFMMetrics(
                "C2",
                10,
                1,
                Decimal("200.00"),
                datetime(2023, 1, 1),
                datetime(2023, 12, 31),
                Decimal("200.00"),
            ),
            RFMMetrics(
                "C3",
                10,
                1,
                Decimal("300.00"),
                datetime(2023, 1, 1),
                datetime(2023, 12, 31),
                Decimal("300.00"),
            ),
            RFMMetrics(
                "C4",
                10,
                1,
                Decimal("400.00"),
                datetime(2023, 1, 1),
                datetime(2023, 12, 31),
                Decimal("400.00"),
            ),
        ]
        result = analyze_single_period(metrics)

        # Median of [100, 200, 300, 400] = (200 + 300) / 2 = 250
        assert result.median_customer_value == Decimal("250.00")

    def test_median_customer_value_odd_count(self):
        """Median calculation with odd number of customers."""
        metrics = [
            RFMMetrics(
                "C1",
                10,
                1,
                Decimal("100.00"),
                datetime(2023, 1, 1),
                datetime(2023, 12, 31),
                Decimal("100.00"),
            ),
            RFMMetrics(
                "C2",
                10,
                1,
                Decimal("200.00"),
                datetime(2023, 1, 1),
                datetime(2023, 12, 31),
                Decimal("200.00"),
            ),
            RFMMetrics(
                "C3",
                10,
                1,
                Decimal("300.00"),
                datetime(2023, 1, 1),
                datetime(2023, 12, 31),
                Decimal("300.00"),
            ),
        ]
        result = analyze_single_period(metrics)

        # Median of [100, 200, 300] = 200
        assert result.median_customer_value == Decimal("200.00")

    def test_rfm_distribution_with_scores(self):
        """RFM distribution should be calculated when scores provided."""
        metrics = [
            RFMMetrics(
                "C1",
                10,
                5,
                Decimal("50.00"),
                datetime(2023, 1, 1),
                datetime(2023, 12, 31),
                Decimal("250.00"),
            ),
            RFMMetrics(
                "C2",
                30,
                2,
                Decimal("75.00"),
                datetime(2023, 1, 1),
                datetime(2023, 12, 31),
                Decimal("150.00"),
            ),
        ]
        scores = [
            RFMScore("C1", 5, 4, 3, "543"),
            RFMScore("C2", 3, 2, 4, "324"),
        ]
        result = analyze_single_period(metrics, scores)

        assert len(result.rfm_distribution) == 2
        assert result.rfm_distribution["543"] == 1
        assert result.rfm_distribution["324"] == 1

    def test_rfm_distribution_empty_without_scores(self):
        """RFM distribution should be empty when scores not provided."""
        metrics = [
            RFMMetrics(
                "C1",
                10,
                5,
                Decimal("50.00"),
                datetime(2023, 1, 1),
                datetime(2023, 12, 31),
                Decimal("250.00"),
            )
        ]
        result = analyze_single_period(metrics)

        assert result.rfm_distribution == {}


class TestCalculateRevenueConcentration:
    """Test calculate_revenue_concentration function."""

    def test_empty_input_returns_zero_concentration(self):
        """Empty RFM metrics should return zero concentration."""
        result = calculate_revenue_concentration([], percentiles=[10, 20])
        assert result[10] == Decimal("0")
        assert result[20] == Decimal("0")

    def test_zero_revenue_returns_zero_concentration(self):
        """Zero total revenue should return zero concentration."""
        metrics = [
            RFMMetrics(
                "C1",
                10,
                1,
                Decimal("0.00"),
                datetime(2023, 1, 1),
                datetime(2023, 12, 31),
                Decimal("0.00"),
            )
        ]
        result = calculate_revenue_concentration(metrics, percentiles=[10])
        assert result[10] == Decimal("0")

    def test_concentration_with_uniform_distribution(self):
        """Uniform revenue distribution should show linear concentration."""
        metrics = [
            RFMMetrics(
                "C1",
                10,
                1,
                Decimal("100.00"),
                datetime(2023, 1, 1),
                datetime(2023, 12, 31),
                Decimal("100.00"),
            ),
            RFMMetrics(
                "C2",
                10,
                1,
                Decimal("100.00"),
                datetime(2023, 1, 1),
                datetime(2023, 12, 31),
                Decimal("100.00"),
            ),
            RFMMetrics(
                "C3",
                10,
                1,
                Decimal("100.00"),
                datetime(2023, 1, 1),
                datetime(2023, 12, 31),
                Decimal("100.00"),
            ),
            RFMMetrics(
                "C4",
                10,
                1,
                Decimal("100.00"),
                datetime(2023, 1, 1),
                datetime(2023, 12, 31),
                Decimal("100.00"),
            ),
        ]
        result = calculate_revenue_concentration(metrics, percentiles=[25, 50])

        # With uniform distribution, top 25% (1 customer) should contribute 25%
        assert result[25] == Decimal("25.0")
        # Top 50% (2 customers) should contribute 50%
        assert result[50] == Decimal("50.0")

    def test_concentration_with_pareto_distribution(self):
        """Pareto-like distribution should show high concentration."""
        metrics = [
            RFMMetrics(
                "C1",
                10,
                1,
                Decimal("700.00"),
                datetime(2023, 1, 1),
                datetime(2023, 12, 31),
                Decimal("700.00"),
            ),
            RFMMetrics(
                "C2",
                10,
                1,
                Decimal("200.00"),
                datetime(2023, 1, 1),
                datetime(2023, 12, 31),
                Decimal("200.00"),
            ),
            RFMMetrics(
                "C3",
                10,
                1,
                Decimal("100.00"),
                datetime(2023, 1, 1),
                datetime(2023, 12, 31),
                Decimal("100.00"),
            ),
        ]
        result = calculate_revenue_concentration(metrics, percentiles=[33])

        # Top 33% (1 customer with $700) should contribute 70% of $1000 total
        assert result[33] == Decimal("70.0")

    def test_concentration_multiple_percentiles(self):
        """Calculate concentration for multiple percentiles."""
        metrics = [
            RFMMetrics(
                f"C{i}",
                10,
                1,
                Decimal("100.00"),
                datetime(2023, 1, 1),
                datetime(2023, 12, 31),
                Decimal("100.00"),
            )
            for i in range(10)
        ]
        # Make C0 a high-value customer
        metrics[0] = RFMMetrics(
            "C0",
            10,
            1,
            Decimal("500.00"),
            datetime(2023, 1, 1),
            datetime(2023, 12, 31),
            Decimal("500.00"),
        )

        result = calculate_revenue_concentration(metrics, percentiles=[10, 20, 50, 100])

        # Total revenue = 500 + 9*100 = 1400
        # Top 10% (1 customer = C0 with 500) = 500/1400 = 35.7%
        assert result[10] == Decimal("35.7")
        # Top 20% (2 customers = C0 + C1) = 600/1400 = 42.9%
        assert result[20] == Decimal("42.9")
        # Top 100% should be 100%
        assert result[100] == Decimal("100.0")

    def test_concentration_sorting(self):
        """Revenue concentration should sort by spend (not customer_id)."""
        metrics = [
            RFMMetrics(
                "C1",
                10,
                1,
                Decimal("100.00"),
                datetime(2023, 1, 1),
                datetime(2023, 12, 31),
                Decimal("100.00"),
            ),
            RFMMetrics(
                "C2",
                10,
                1,
                Decimal("500.00"),
                datetime(2023, 1, 1),
                datetime(2023, 12, 31),
                Decimal("500.00"),
            ),
            RFMMetrics(
                "C3",
                10,
                1,
                Decimal("200.00"),
                datetime(2023, 1, 1),
                datetime(2023, 12, 31),
                Decimal("200.00"),
            ),
        ]
        result = calculate_revenue_concentration(metrics, percentiles=[33])

        # Top 33% (1 customer) should be C2 with $500 = 62.5% of $800 total
        assert result[33] == Decimal("62.5")
