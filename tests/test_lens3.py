"""Unit tests for Lens 3 cohort evolution tracking."""

from datetime import datetime

import pytest

from customer_base_audit.analyses.lens3 import (
    CohortPeriodMetrics,
    Lens3Metrics,
    analyze_cohort_evolution,
    calculate_retention_curve,
)
from customer_base_audit.foundation.data_mart import PeriodAggregation


class TestCohortPeriodMetrics:
    """Test suite for CohortPeriodMetrics dataclass."""

    def test_valid_cohort_period_metrics(self):
        """Test creation of valid CohortPeriodMetrics."""
        metrics = CohortPeriodMetrics(
            period_number=0,
            active_customers=100,
            retention_rate=1.0,
            avg_orders_per_customer=1.5,
            avg_revenue_per_customer=50.0,
            avg_orders_per_cohort_member=1.5,
            avg_revenue_per_cohort_member=50.0,
            total_revenue=5000.0,
        )
        assert metrics.period_number == 0
        assert metrics.active_customers == 100
        assert metrics.retention_rate == 1.0

    def test_negative_period_number_raises_error(self):
        """Test that negative period_number raises ValueError."""
        with pytest.raises(ValueError, match="period_number must be >= 0"):
            CohortPeriodMetrics(
                period_number=-1,
                active_customers=100,
                retention_rate=1.0,
                avg_orders_per_customer=1.5,
                avg_revenue_per_customer=50.0,
                avg_orders_per_cohort_member=1.5,
                avg_revenue_per_cohort_member=50.0,
                total_revenue=5000.0,
            )

    def test_negative_active_customers_raises_error(self):
        """Test that negative active_customers raises ValueError."""
        with pytest.raises(ValueError, match="active_customers must be >= 0"):
            CohortPeriodMetrics(
                period_number=0,
                active_customers=-1,
                retention_rate=1.0,
                avg_orders_per_customer=1.5,
                avg_revenue_per_customer=50.0,
                avg_orders_per_cohort_member=1.5,
                avg_revenue_per_cohort_member=50.0,
                total_revenue=5000.0,
            )

    def test_invalid_retention_rate_raises_error(self):
        """Test that retention_rate outside [0, 1] raises ValueError."""
        with pytest.raises(ValueError, match="retention_rate must be between 0 and 1"):
            CohortPeriodMetrics(
                period_number=0,
                active_customers=100,
                retention_rate=1.5,
                avg_orders_per_customer=1.5,
                avg_revenue_per_customer=50.0,
                avg_orders_per_cohort_member=1.5,
                avg_revenue_per_cohort_member=50.0,
                total_revenue=5000.0,
            )

    def test_negative_avg_orders_raises_error(self):
        """Test that negative avg_orders_per_customer raises ValueError."""
        with pytest.raises(ValueError, match="avg_orders_per_customer must be >= 0"):
            CohortPeriodMetrics(
                period_number=0,
                active_customers=100,
                retention_rate=1.0,
                avg_orders_per_customer=-1.5,
                avg_revenue_per_customer=50.0,
                avg_orders_per_cohort_member=1.5,
                avg_revenue_per_cohort_member=50.0,
                total_revenue=5000.0,
            )

    def test_negative_avg_revenue_raises_error(self):
        """Test that negative avg_revenue_per_customer raises ValueError."""
        with pytest.raises(ValueError, match="avg_revenue_per_customer must be >= 0"):
            CohortPeriodMetrics(
                period_number=0,
                active_customers=100,
                retention_rate=1.0,
                avg_orders_per_customer=1.5,
                avg_revenue_per_customer=-50.0,
                avg_orders_per_cohort_member=1.5,
                avg_revenue_per_cohort_member=50.0,
                total_revenue=5000.0,
            )

    def test_negative_total_revenue_raises_error(self):
        """Test that negative total_revenue raises ValueError."""
        with pytest.raises(ValueError, match="total_revenue must be >= 0"):
            CohortPeriodMetrics(
                period_number=0,
                active_customers=100,
                retention_rate=1.0,
                avg_orders_per_customer=1.5,
                avg_revenue_per_customer=50.0,
                avg_orders_per_cohort_member=1.5,
                avg_revenue_per_cohort_member=50.0,
                total_revenue=-5000.0,
            )


class TestLens3Metrics:
    """Test suite for Lens3Metrics dataclass."""

    def test_valid_lens3_metrics(self):
        """Test creation of valid Lens3Metrics."""
        periods = [
            CohortPeriodMetrics(0, 100, 1.0, 1.5, 50.0, 1.5, 50.0, 5000.0),
            CohortPeriodMetrics(1, 80, 0.8, 1.2, 40.0, 1.2, 40.0, 3200.0),
        ]
        metrics = Lens3Metrics(
            cohort_name="2023-01",
            acquisition_date=datetime(2023, 1, 1),
            cohort_size=100,
            periods=periods,
        )
        assert metrics.cohort_name == "2023-01"
        assert metrics.cohort_size == 100
        assert len(metrics.periods) == 2

    def test_negative_cohort_size_raises_error(self):
        """Test that negative cohort_size raises ValueError."""
        with pytest.raises(ValueError, match="cohort_size must be >= 0"):
            Lens3Metrics(
                cohort_name="2023-01",
                acquisition_date=datetime(2023, 1, 1),
                cohort_size=-1,
                periods=[],
            )

    def test_unordered_periods_raises_error(self):
        """Test that periods not ordered by period_number raises ValueError."""
        periods = [
            CohortPeriodMetrics(1, 80, 0.8, 1.2, 40.0, 1.2, 40.0, 3200.0),
            CohortPeriodMetrics(
                0, 100, 1.0, 1.5, 50.0, 1.5, 50.0, 5000.0
            ),  # Out of order
        ]
        with pytest.raises(
            ValueError, match="periods must be ordered by period_number"
        ):
            Lens3Metrics(
                cohort_name="2023-01",
                acquisition_date=datetime(2023, 1, 1),
                cohort_size=100,
                periods=periods,
            )

    def test_non_contiguous_period_numbers_raises_error(self):
        """Test that non-contiguous period numbers raise ValueError."""
        periods = [
            CohortPeriodMetrics(0, 100, 1.0, 1.5, 50.0, 1.5, 50.0, 5000.0),
            CohortPeriodMetrics(
                2, 60, 0.6, 1.0, 35.0, 1.0, 35.0, 2100.0
            ),  # Skips period 1
        ]
        with pytest.raises(ValueError, match="period_numbers must be contiguous"):
            Lens3Metrics(
                cohort_name="2023-01",
                acquisition_date=datetime(2023, 1, 1),
                cohort_size=100,
                periods=periods,
            )


class TestAnalyzeCohortEvolution:
    """Test suite for analyze_cohort_evolution function."""

    def test_simple_cohort_evolution(self):
        """Test basic cohort evolution with 3 customers over 3 periods."""
        cohort_customers = ["C1", "C2", "C3"]
        acquisition_date = datetime(2023, 1, 1)

        period_aggregations = [
            # Period 0 (acquisition): All 3 customers active
            PeriodAggregation(
                "C1", datetime(2023, 1, 1), datetime(2023, 2, 1), 2, 100.0, 30.0, 5
            ),
            PeriodAggregation(
                "C2", datetime(2023, 1, 1), datetime(2023, 2, 1), 1, 50.0, 15.0, 2
            ),
            PeriodAggregation(
                "C3", datetime(2023, 1, 1), datetime(2023, 2, 1), 1, 75.0, 20.0, 3
            ),
            # Period 1: Only C1 and C3 active (C2 churned)
            PeriodAggregation(
                "C1", datetime(2023, 2, 1), datetime(2023, 3, 1), 1, 75.0, 20.0, 3
            ),
            PeriodAggregation(
                "C3", datetime(2023, 2, 1), datetime(2023, 3, 1), 2, 100.0, 30.0, 4
            ),
            # Period 2: Only C1 active
            PeriodAggregation(
                "C1", datetime(2023, 3, 1), datetime(2023, 4, 1), 1, 50.0, 15.0, 2
            ),
        ]

        metrics = analyze_cohort_evolution(
            cohort_name="2023-01",
            acquisition_date=acquisition_date,
            period_aggregations=period_aggregations,
            cohort_customer_ids=cohort_customers,
        )

        assert metrics.cohort_name == "2023-01"
        assert metrics.cohort_size == 3
        assert len(metrics.periods) == 3

        # Period 0: All 3 customers (100% cumulative retention)
        assert metrics.periods[0].period_number == 0
        assert metrics.periods[0].active_customers == 3
        assert metrics.periods[0].retention_rate == 1.0  # 3/3 customers seen
        assert metrics.periods[0].total_revenue == 225.0  # 100 + 50 + 75

        # Period 1: 2 customers active this period (100% cumulative retention)
        assert metrics.periods[1].period_number == 1
        assert metrics.periods[1].active_customers == 2  # C1 and C3
        assert metrics.periods[1].retention_rate == 1.0  # All 3 customers seen by now
        assert metrics.periods[1].total_revenue == 175.0  # 75 + 100

        # Period 2: 1 customer active this period (100% cumulative retention)
        assert metrics.periods[2].period_number == 2
        assert metrics.periods[2].active_customers == 1  # Only C1
        assert metrics.periods[2].retention_rate == 1.0  # All 3 customers seen by now
        assert metrics.periods[2].total_revenue == 50.0

    def test_cohort_with_single_period(self):
        """Test cohort with only acquisition period."""
        cohort_customers = ["C1", "C2"]
        acquisition_date = datetime(2023, 1, 1)

        period_aggregations = [
            PeriodAggregation(
                "C1", datetime(2023, 1, 1), datetime(2023, 2, 1), 1, 100.0, 30.0, 5
            ),
            PeriodAggregation(
                "C2", datetime(2023, 1, 1), datetime(2023, 2, 1), 1, 50.0, 15.0, 2
            ),
        ]

        metrics = analyze_cohort_evolution(
            cohort_name="2023-01",
            acquisition_date=acquisition_date,
            period_aggregations=period_aggregations,
            cohort_customer_ids=cohort_customers,
        )

        assert metrics.cohort_size == 2
        assert len(metrics.periods) == 1
        assert metrics.periods[0].period_number == 0
        assert metrics.periods[0].active_customers == 2
        assert metrics.periods[0].retention_rate == 1.0

    def test_cohort_with_100_percent_retention(self):
        """Test cohort where all customers remain active."""
        cohort_customers = ["C1", "C2"]
        acquisition_date = datetime(2023, 1, 1)

        period_aggregations = [
            # Period 0
            PeriodAggregation(
                "C1", datetime(2023, 1, 1), datetime(2023, 2, 1), 1, 100.0, 30.0, 5
            ),
            PeriodAggregation(
                "C2", datetime(2023, 1, 1), datetime(2023, 2, 1), 1, 50.0, 15.0, 2
            ),
            # Period 1
            PeriodAggregation(
                "C1", datetime(2023, 2, 1), datetime(2023, 3, 1), 1, 75.0, 20.0, 3
            ),
            PeriodAggregation(
                "C2", datetime(2023, 2, 1), datetime(2023, 3, 1), 1, 60.0, 18.0, 3
            ),
        ]

        metrics = analyze_cohort_evolution(
            cohort_name="2023-01",
            acquisition_date=acquisition_date,
            period_aggregations=period_aggregations,
            cohort_customer_ids=cohort_customers,
        )

        assert len(metrics.periods) == 2
        assert metrics.periods[0].retention_rate == 1.0
        assert metrics.periods[1].retention_rate == 1.0

    def test_cohort_with_100_percent_churn(self):
        """Test cohort where all customers churn after acquisition."""
        cohort_customers = ["C1", "C2"]
        acquisition_date = datetime(2023, 1, 1)

        period_aggregations = [
            # Period 0: All customers active
            PeriodAggregation(
                "C1", datetime(2023, 1, 1), datetime(2023, 2, 1), 1, 100.0, 30.0, 5
            ),
            PeriodAggregation(
                "C2", datetime(2023, 1, 1), datetime(2023, 2, 1), 1, 50.0, 15.0, 2
            ),
            # Period 1: No customers active (all churned)
        ]

        metrics = analyze_cohort_evolution(
            cohort_name="2023-01",
            acquisition_date=acquisition_date,
            period_aggregations=period_aggregations,
            cohort_customer_ids=cohort_customers,
        )

        # Should only have period 0 since no activity in period 1
        assert len(metrics.periods) == 1
        assert metrics.periods[0].retention_rate == 1.0

    def test_empty_cohort_customer_ids_raises_error(self):
        """Test that empty cohort_customer_ids raises ValueError."""
        with pytest.raises(ValueError, match="cohort_customer_ids cannot be empty"):
            analyze_cohort_evolution(
                cohort_name="2023-01",
                acquisition_date=datetime(2023, 1, 1),
                period_aggregations=[],
                cohort_customer_ids=[],
            )

    def test_no_matching_period_aggregations_raises_error(self):
        """Test that no matching period aggregations raises ValueError."""
        cohort_customers = ["C1", "C2"]
        period_aggregations = [
            PeriodAggregation(
                "C99", datetime(2023, 1, 1), datetime(2023, 2, 1), 1, 100.0, 30.0, 5
            ),
        ]

        with pytest.raises(ValueError, match="No period aggregations found"):
            analyze_cohort_evolution(
                cohort_name="2023-01",
                acquisition_date=datetime(2023, 1, 1),
                period_aggregations=period_aggregations,
                cohort_customer_ids=cohort_customers,
            )

    def test_acquisition_date_after_all_periods_raises_error(self):
        """Test that acquisition_date after all periods raises ValueError."""
        cohort_customers = ["C1"]
        period_aggregations = [
            PeriodAggregation(
                "C1", datetime(2023, 1, 1), datetime(2023, 2, 1), 1, 100.0, 30.0, 5
            ),
        ]

        with pytest.raises(
            ValueError, match="No periods found on or after acquisition_date"
        ):
            analyze_cohort_evolution(
                cohort_name="2023-01",
                acquisition_date=datetime(2024, 1, 1),  # After all periods
                period_aggregations=period_aggregations,
                cohort_customer_ids=cohort_customers,
            )

    def test_avg_metrics_calculations(self):
        """Test that average metrics are calculated correctly."""
        cohort_customers = ["C1", "C2"]
        acquisition_date = datetime(2023, 1, 1)

        period_aggregations = [
            # C1: 3 orders, $300 revenue
            PeriodAggregation(
                "C1", datetime(2023, 1, 1), datetime(2023, 2, 1), 3, 300.0, 90.0, 10
            ),
            # C2: 1 order, $100 revenue
            PeriodAggregation(
                "C2", datetime(2023, 1, 1), datetime(2023, 2, 1), 1, 100.0, 30.0, 5
            ),
        ]

        metrics = analyze_cohort_evolution(
            cohort_name="2023-01",
            acquisition_date=acquisition_date,
            period_aggregations=period_aggregations,
            cohort_customer_ids=cohort_customers,
        )

        # 2 active customers, 4 total orders, $400 total revenue
        assert metrics.periods[0].active_customers == 2
        assert (
            metrics.periods[0].avg_orders_per_customer == 2.0
        )  # 4 orders / 2 customers
        assert (
            metrics.periods[0].avg_revenue_per_customer == 200.0
        )  # $400 / 2 customers
        assert metrics.periods[0].total_revenue == 400.0

    def test_customer_with_multiple_periods_in_same_period(self):
        """Test handling of customers with data in the same period."""
        cohort_customers = ["C1"]
        acquisition_date = datetime(2023, 1, 1)

        # C1 has one period aggregation
        period_aggregations = [
            PeriodAggregation(
                "C1", datetime(2023, 1, 1), datetime(2023, 2, 1), 2, 100.0, 30.0, 5
            ),
        ]

        metrics = analyze_cohort_evolution(
            cohort_name="2023-01",
            acquisition_date=acquisition_date,
            period_aggregations=period_aggregations,
            cohort_customer_ids=cohort_customers,
        )

        assert metrics.periods[0].active_customers == 1
        assert metrics.periods[0].total_revenue == 100.0

    def test_filters_out_non_cohort_customers(self):
        """Test that period aggregations for non-cohort customers are filtered out."""
        cohort_customers = ["C1", "C2"]
        acquisition_date = datetime(2023, 1, 1)

        period_aggregations = [
            PeriodAggregation(
                "C1", datetime(2023, 1, 1), datetime(2023, 2, 1), 1, 100.0, 30.0, 5
            ),
            PeriodAggregation(
                "C2", datetime(2023, 1, 1), datetime(2023, 2, 1), 1, 50.0, 15.0, 2
            ),
            # C99 is not in cohort
            PeriodAggregation(
                "C99", datetime(2023, 1, 1), datetime(2023, 2, 1), 5, 500.0, 150.0, 20
            ),
        ]

        metrics = analyze_cohort_evolution(
            cohort_name="2023-01",
            acquisition_date=acquisition_date,
            period_aggregations=period_aggregations,
            cohort_customer_ids=cohort_customers,
        )

        # Should only include C1 and C2
        assert metrics.periods[0].active_customers == 2
        assert metrics.periods[0].total_revenue == 150.0  # 100 + 50, not including C99

    def test_retention_is_monotonically_non_decreasing(self):
        """Test that retention curves are cumulative and never decrease."""
        cohort_customers = ["C1", "C2", "C3"]
        acquisition_date = datetime(2023, 1, 1)

        period_aggregations = [
            # Period 0: All 3 active (100% retention)
            PeriodAggregation(
                "C1", datetime(2023, 1, 1), datetime(2023, 2, 1), 1, 100.0, 30.0, 5
            ),
            PeriodAggregation(
                "C2", datetime(2023, 1, 1), datetime(2023, 2, 1), 1, 100.0, 30.0, 5
            ),
            PeriodAggregation(
                "C3", datetime(2023, 1, 1), datetime(2023, 2, 1), 1, 100.0, 30.0, 5
            ),
            # Period 1: Only C1 active (retention stays 100% - C2, C3 already counted)
            PeriodAggregation(
                "C1", datetime(2023, 2, 1), datetime(2023, 3, 1), 1, 100.0, 30.0, 5
            ),
            # Period 2: C1 and C2 return (retention stays 100%)
            PeriodAggregation(
                "C1", datetime(2023, 3, 1), datetime(2023, 4, 1), 1, 100.0, 30.0, 5
            ),
            PeriodAggregation(
                "C2", datetime(2023, 3, 1), datetime(2023, 4, 1), 1, 100.0, 30.0, 5
            ),
            # Period 3: Only C3 active (retention stays 100% - all customers seen)
            PeriodAggregation(
                "C3", datetime(2023, 4, 1), datetime(2023, 5, 1), 1, 100.0, 30.0, 5
            ),
        ]

        metrics = analyze_cohort_evolution(
            cohort_name="2023-01",
            acquisition_date=acquisition_date,
            period_aggregations=period_aggregations,
            cohort_customer_ids=cohort_customers,
        )

        # All customers active at some point, so retention should be 100% throughout
        retention_rates = [p.retention_rate for p in metrics.periods]
        assert all(r == 1.0 for r in retention_rates), (
            f"Retention should be 100% throughout, got {retention_rates}"
        )

        # Verify retention is monotonically non-decreasing
        for i in range(1, len(retention_rates)):
            assert retention_rates[i] >= retention_rates[i - 1], (
                f"Retention decreased from {retention_rates[i - 1]} to {retention_rates[i]} at period {i}"
            )

    def test_duplicate_customer_ids_raises_error(self):
        """Test that duplicate customer IDs in cohort_customer_ids raise ValueError."""
        cohort_customers = ["C1", "C2", "C1"]  # Duplicate C1
        acquisition_date = datetime(2023, 1, 1)

        period_aggregations = [
            PeriodAggregation(
                "C1", datetime(2023, 1, 1), datetime(2023, 2, 1), 1, 100.0, 30.0, 5
            ),
        ]

        with pytest.raises(ValueError, match="contains 1 duplicate"):
            analyze_cohort_evolution(
                cohort_name="2023-01",
                acquisition_date=acquisition_date,
                period_aggregations=period_aggregations,
                cohort_customer_ids=cohort_customers,
            )


class TestCalculateRetentionCurve:
    """Test suite for calculate_retention_curve function."""

    def test_retention_curve_extraction(self):
        """Test extracting retention curve from Lens3Metrics."""
        periods = [
            CohortPeriodMetrics(0, 100, 1.0, 1.5, 50.0, 1.5, 50.0, 5000.0),
            CohortPeriodMetrics(1, 80, 0.8, 1.2, 40.0, 1.2, 40.0, 3200.0),
            CohortPeriodMetrics(2, 60, 0.6, 1.0, 35.0, 1.0, 35.0, 2100.0),
        ]
        metrics = Lens3Metrics(
            cohort_name="2023-01",
            acquisition_date=datetime(2023, 1, 1),
            cohort_size=100,
            periods=periods,
        )

        curve = calculate_retention_curve(metrics)

        assert curve == {0: 1.0, 1: 0.8, 2: 0.6}

    def test_empty_periods_returns_empty_curve(self):
        """Test that empty periods returns empty retention curve."""
        metrics = Lens3Metrics(
            cohort_name="2023-01",
            acquisition_date=datetime(2023, 1, 1),
            cohort_size=0,
            periods=[],
        )

        curve = calculate_retention_curve(metrics)

        assert curve == {}

    def test_single_period_returns_single_point_curve(self):
        """Test retention curve with single period."""
        periods = [
            CohortPeriodMetrics(0, 50, 1.0, 2.0, 60.0, 2.0, 60.0, 3000.0),
        ]
        metrics = Lens3Metrics(
            cohort_name="2023-Q1",
            acquisition_date=datetime(2023, 1, 1),
            cohort_size=50,
            periods=periods,
        )

        curve = calculate_retention_curve(metrics)

        assert curve == {0: 1.0}
