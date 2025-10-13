"""Tests for CLV Lens 5: Overall customer base health."""

from datetime import datetime, timezone
from decimal import Decimal

import pytest

from customer_base_audit.analyses.lens5 import (
    CohortRepeatBehavior,
    CohortRevenuePeriod,
    CustomerBaseHealthScore,
    Lens5Metrics,
    assess_customer_base_health,
    calculate_health_score,
)
from customer_base_audit.foundation.data_mart import PeriodAggregation


class TestCohortRevenuePeriod:
    """Tests for CohortRevenuePeriod dataclass."""

    def test_valid_cohort_revenue_period(self):
        """Test CohortRevenuePeriod with valid values."""
        crp = CohortRevenuePeriod(
            cohort_id="2024-Q1",
            period_start=datetime(2024, 1, 1, tzinfo=timezone.utc),
            total_revenue=Decimal("10000.00"),
            pct_of_period_revenue=Decimal("50.00"),
            active_customers=100,
            avg_revenue_per_customer=Decimal("100.00"),
        )
        assert crp.cohort_id == "2024-Q1"
        assert crp.total_revenue == Decimal("10000.00")
        assert crp.pct_of_period_revenue == Decimal("50.00")
        assert crp.active_customers == 100
        assert crp.avg_revenue_per_customer == Decimal("100.00")

    def test_negative_total_revenue_raises_error(self):
        """Test that negative total_revenue raises ValueError."""
        with pytest.raises(ValueError, match="total_revenue must be >= 0"):
            CohortRevenuePeriod(
                cohort_id="2024-Q1",
                period_start=datetime(2024, 1, 1, tzinfo=timezone.utc),
                total_revenue=Decimal("-100.00"),
                pct_of_period_revenue=Decimal("50.00"),
                active_customers=100,
                avg_revenue_per_customer=Decimal("100.00"),
            )

    def test_invalid_pct_of_period_revenue_raises_error(self):
        """Test that pct_of_period_revenue outside [0,100] raises ValueError."""
        with pytest.raises(ValueError, match="pct_of_period_revenue must be in"):
            CohortRevenuePeriod(
                cohort_id="2024-Q1",
                period_start=datetime(2024, 1, 1, tzinfo=timezone.utc),
                total_revenue=Decimal("10000.00"),
                pct_of_period_revenue=Decimal("150.00"),
                active_customers=100,
                avg_revenue_per_customer=Decimal("100.00"),
            )

    def test_negative_active_customers_raises_error(self):
        """Test that negative active_customers raises ValueError."""
        with pytest.raises(ValueError, match="active_customers must be >= 0"):
            CohortRevenuePeriod(
                cohort_id="2024-Q1",
                period_start=datetime(2024, 1, 1, tzinfo=timezone.utc),
                total_revenue=Decimal("10000.00"),
                pct_of_period_revenue=Decimal("50.00"),
                active_customers=-10,
                avg_revenue_per_customer=Decimal("100.00"),
            )

    def test_negative_avg_revenue_per_customer_raises_error(self):
        """Test that negative avg_revenue_per_customer raises ValueError."""
        with pytest.raises(ValueError, match="avg_revenue_per_customer must be >= 0"):
            CohortRevenuePeriod(
                cohort_id="2024-Q1",
                period_start=datetime(2024, 1, 1, tzinfo=timezone.utc),
                total_revenue=Decimal("10000.00"),
                pct_of_period_revenue=Decimal("50.00"),
                active_customers=100,
                avg_revenue_per_customer=Decimal("-10.00"),
            )


class TestCohortRepeatBehavior:
    """Tests for CohortRepeatBehavior dataclass."""

    def test_valid_repeat_behavior(self):
        """Test CohortRepeatBehavior with valid values."""
        crb = CohortRepeatBehavior(
            cohort_id="2024-Q1",
            cohort_size=100,
            one_time_buyers=40,
            repeat_buyers=60,
            repeat_rate=Decimal("60.00"),
            avg_orders_per_repeat_buyer=Decimal("3.50"),
        )
        assert crb.cohort_id == "2024-Q1"
        assert crb.cohort_size == 100
        assert crb.one_time_buyers == 40
        assert crb.repeat_buyers == 60
        assert crb.repeat_rate == Decimal("60.00")
        assert crb.avg_orders_per_repeat_buyer == Decimal("3.50")

    def test_negative_cohort_size_raises_error(self):
        """Test that negative cohort_size raises ValueError."""
        with pytest.raises(ValueError, match="cohort_size must be >= 0"):
            CohortRepeatBehavior(
                cohort_id="2024-Q1",
                cohort_size=-10,
                one_time_buyers=40,
                repeat_buyers=60,
                repeat_rate=Decimal("60.00"),
                avg_orders_per_repeat_buyer=Decimal("3.50"),
            )

    def test_buyers_sum_not_equal_cohort_size_raises_error(self):
        """Test that one_time + repeat != cohort_size raises ValueError."""
        with pytest.raises(ValueError, match="must equal cohort_size"):
            CohortRepeatBehavior(
                cohort_id="2024-Q1",
                cohort_size=100,
                one_time_buyers=40,
                repeat_buyers=50,  # Should be 60
                repeat_rate=Decimal("60.00"),
                avg_orders_per_repeat_buyer=Decimal("3.50"),
            )

    def test_invalid_repeat_rate_raises_error(self):
        """Test that repeat_rate outside [0,100] raises ValueError."""
        with pytest.raises(ValueError, match="repeat_rate must be in"):
            CohortRepeatBehavior(
                cohort_id="2024-Q1",
                cohort_size=100,
                one_time_buyers=40,
                repeat_buyers=60,
                repeat_rate=Decimal("150.00"),
                avg_orders_per_repeat_buyer=Decimal("3.50"),
            )

    def test_avg_orders_less_than_2_raises_error(self):
        """Test that avg_orders_per_repeat_buyer < 2 raises ValueError."""
        with pytest.raises(
            ValueError, match="avg_orders_per_repeat_buyer must be >= 2"
        ):
            CohortRepeatBehavior(
                cohort_id="2024-Q1",
                cohort_size=100,
                one_time_buyers=40,
                repeat_buyers=60,
                repeat_rate=Decimal("60.00"),
                avg_orders_per_repeat_buyer=Decimal("1.50"),  # Must be >= 2
            )

    def test_zero_repeat_buyers_allows_zero_avg_orders(self):
        """Test that avg_orders can be 0 when repeat_buyers is 0."""
        crb = CohortRepeatBehavior(
            cohort_id="2024-Q1",
            cohort_size=100,
            one_time_buyers=100,
            repeat_buyers=0,
            repeat_rate=Decimal("0.00"),
            avg_orders_per_repeat_buyer=Decimal("0.00"),
        )
        assert crb.repeat_buyers == 0
        assert crb.avg_orders_per_repeat_buyer == Decimal("0.00")


class TestCustomerBaseHealthScore:
    """Tests for CustomerBaseHealthScore dataclass."""

    def test_valid_health_score(self):
        """Test CustomerBaseHealthScore with valid values."""
        cbhs = CustomerBaseHealthScore(
            total_customers=1000,
            total_active_customers=750,
            overall_retention_rate=Decimal("75.00"),
            cohort_quality_trend="stable",
            revenue_predictability_pct=Decimal("65.00"),
            acquisition_dependence_pct=Decimal("25.00"),
            health_score=Decimal("70.50"),
            health_grade="C",
        )
        assert cbhs.total_customers == 1000
        assert cbhs.total_active_customers == 750
        assert cbhs.overall_retention_rate == Decimal("75.00")
        assert cbhs.cohort_quality_trend == "stable"
        assert cbhs.health_score == Decimal("70.50")
        assert cbhs.health_grade == "C"

    def test_negative_total_customers_raises_error(self):
        """Test that negative total_customers raises ValueError."""
        with pytest.raises(ValueError, match="total_customers must be >= 0"):
            CustomerBaseHealthScore(
                total_customers=-10,
                total_active_customers=750,
                overall_retention_rate=Decimal("75.00"),
                cohort_quality_trend="stable",
                revenue_predictability_pct=Decimal("65.00"),
                acquisition_dependence_pct=Decimal("25.00"),
                health_score=Decimal("70.50"),
                health_grade="C",
            )

    def test_active_exceeds_total_raises_error(self):
        """Test that total_active_customers > total_customers raises ValueError."""
        with pytest.raises(ValueError, match="cannot exceed total_customers"):
            CustomerBaseHealthScore(
                total_customers=1000,
                total_active_customers=1500,
                overall_retention_rate=Decimal("75.00"),
                cohort_quality_trend="stable",
                revenue_predictability_pct=Decimal("65.00"),
                acquisition_dependence_pct=Decimal("25.00"),
                health_score=Decimal("70.50"),
                health_grade="C",
            )

    def test_invalid_cohort_quality_trend_raises_error(self):
        """Test that invalid cohort_quality_trend raises ValueError."""
        with pytest.raises(ValueError, match="must be 'improving', 'stable', or"):
            CustomerBaseHealthScore(
                total_customers=1000,
                total_active_customers=750,
                overall_retention_rate=Decimal("75.00"),
                cohort_quality_trend="unknown",
                revenue_predictability_pct=Decimal("65.00"),
                acquisition_dependence_pct=Decimal("25.00"),
                health_score=Decimal("70.50"),
                health_grade="C",
            )

    def test_invalid_health_grade_raises_error(self):
        """Test that invalid health_grade raises ValueError."""
        with pytest.raises(ValueError, match="must be 'A', 'B', 'C', 'D', or 'F'"):
            CustomerBaseHealthScore(
                total_customers=1000,
                total_active_customers=750,
                overall_retention_rate=Decimal("75.00"),
                cohort_quality_trend="stable",
                revenue_predictability_pct=Decimal("65.00"),
                acquisition_dependence_pct=Decimal("25.00"),
                health_score=Decimal("70.50"),
                health_grade="X",
            )


class TestCalculateHealthScore:
    """Tests for calculate_health_score() helper function."""

    def test_perfect_health_score(self):
        """Test health score with perfect metrics."""
        score, grade = calculate_health_score(
            Decimal("100.00"), "improving", Decimal("100.00"), Decimal("0.00")
        )
        # Score = 100*0.3 + 80*0.3 + 100*0.2 + 100*0.2 = 30 + 24 + 20 + 20 = 94
        assert score == Decimal("94.00")
        assert grade == "A"

    def test_stable_health_score(self):
        """Test health score with stable metrics."""
        score, grade = calculate_health_score(
            Decimal("75.00"), "stable", Decimal("60.00"), Decimal("30.00")
        )
        # Score = 75*0.3 + 50*0.3 + 60*0.2 + 70*0.2 = 22.5 + 15 + 12 + 14 = 63.5
        assert score == Decimal("63.50")
        assert grade == "D"

    def test_declining_health_score(self):
        """Test health score with declining trend."""
        score, grade = calculate_health_score(
            Decimal("50.00"), "declining", Decimal("40.00"), Decimal("50.00")
        )
        # Score = 50*0.3 + 20*0.3 + 40*0.2 + 50*0.2 = 15 + 6 + 8 + 10 = 39
        assert score == Decimal("39.00")
        assert grade == "F"

    def test_improving_health_score(self):
        """Test health score with improving trend."""
        score, grade = calculate_health_score(
            Decimal("85.00"), "improving", Decimal("75.00"), Decimal("20.00")
        )
        # Score = 85*0.3 + 80*0.3 + 75*0.2 + 80*0.2 = 25.5 + 24 + 15 + 16 = 80.5
        assert score == Decimal("80.50")
        assert grade == "B"

    def test_grade_boundaries(self):
        """Test grade boundary calculations."""
        # Test A grade (90+)
        score, grade = calculate_health_score(
            Decimal("100.00"), "improving", Decimal("90.00"), Decimal("5.00")
        )
        # Score = 100*0.3 + 80*0.3 + 90*0.2 + 95*0.2 = 30 + 24 + 18 + 19 = 91
        assert score >= Decimal("90.00")
        assert grade == "A"

        # Test D grade (60-69)
        score, grade = calculate_health_score(
            Decimal("70.00"), "stable", Decimal("70.00"), Decimal("30.00")
        )
        # Score = 70*0.3 + 50*0.3 + 70*0.2 + 70*0.2 = 21 + 15 + 14 + 14 = 64
        assert Decimal("60.00") <= score < Decimal("70.00")
        assert grade == "D"


class TestLens5Metrics:
    """Tests for Lens5Metrics dataclass."""

    def test_valid_lens5_metrics(self):
        """Test Lens5Metrics with valid values."""
        crp = CohortRevenuePeriod(
            cohort_id="2024-Q1",
            period_start=datetime(2024, 1, 1, tzinfo=timezone.utc),
            total_revenue=Decimal("10000.00"),
            pct_of_period_revenue=Decimal("50.00"),
            active_customers=100,
            avg_revenue_per_customer=Decimal("100.00"),
        )
        crb = CohortRepeatBehavior(
            cohort_id="2024-Q1",
            cohort_size=100,
            one_time_buyers=40,
            repeat_buyers=60,
            repeat_rate=Decimal("60.00"),
            avg_orders_per_repeat_buyer=Decimal("3.50"),
        )
        cbhs = CustomerBaseHealthScore(
            total_customers=1000,
            total_active_customers=750,
            overall_retention_rate=Decimal("75.00"),
            cohort_quality_trend="stable",
            revenue_predictability_pct=Decimal("65.00"),
            acquisition_dependence_pct=Decimal("25.00"),
            health_score=Decimal("70.50"),
            health_grade="C",
        )

        metrics = Lens5Metrics(
            cohort_revenue_contributions=[crp],
            cohort_repeat_behavior=[crb],
            health_score=cbhs,
            analysis_start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
            analysis_end_date=datetime(2024, 12, 31, tzinfo=timezone.utc),
        )

        assert len(metrics.cohort_revenue_contributions) == 1
        assert len(metrics.cohort_repeat_behavior) == 1
        assert metrics.health_score.health_grade == "C"

    def test_invalid_date_range_raises_error(self):
        """Test that start >= end raises ValueError."""
        cbhs = CustomerBaseHealthScore(
            total_customers=1000,
            total_active_customers=750,
            overall_retention_rate=Decimal("75.00"),
            cohort_quality_trend="stable",
            revenue_predictability_pct=Decimal("65.00"),
            acquisition_dependence_pct=Decimal("25.00"),
            health_score=Decimal("70.50"),
            health_grade="C",
        )

        with pytest.raises(ValueError, match="must be before"):
            Lens5Metrics(
                cohort_revenue_contributions=[],
                cohort_repeat_behavior=[],
                health_score=cbhs,
                analysis_start_date=datetime(2024, 12, 31, tzinfo=timezone.utc),
                analysis_end_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
            )

    def test_unsorted_revenue_contributions_raise_error(self):
        """Test that unsorted cohort_revenue_contributions raise ValueError."""
        crp1 = CohortRevenuePeriod(
            cohort_id="2024-Q1",
            period_start=datetime(2024, 2, 1, tzinfo=timezone.utc),  # Later period
            total_revenue=Decimal("10000.00"),
            pct_of_period_revenue=Decimal("50.00"),
            active_customers=100,
            avg_revenue_per_customer=Decimal("100.00"),
        )
        crp2 = CohortRevenuePeriod(
            cohort_id="2024-Q1",
            period_start=datetime(2024, 1, 1, tzinfo=timezone.utc),  # Earlier period
            total_revenue=Decimal("10000.00"),
            pct_of_period_revenue=Decimal("50.00"),
            active_customers=100,
            avg_revenue_per_customer=Decimal("100.00"),
        )
        cbhs = CustomerBaseHealthScore(
            total_customers=1000,
            total_active_customers=750,
            overall_retention_rate=Decimal("75.00"),
            cohort_quality_trend="stable",
            revenue_predictability_pct=Decimal("65.00"),
            acquisition_dependence_pct=Decimal("25.00"),
            health_score=Decimal("70.50"),
            health_grade="C",
        )

        with pytest.raises(ValueError, match="must be sorted by period_start"):
            Lens5Metrics(
                cohort_revenue_contributions=[crp1, crp2],  # Wrong order
                cohort_repeat_behavior=[],
                health_score=cbhs,
                analysis_start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                analysis_end_date=datetime(2024, 12, 31, tzinfo=timezone.utc),
            )


class TestAssessCustomerBaseHealth:
    """Tests for assess_customer_base_health() main function."""

    def test_empty_period_aggregations_raises_error(self):
        """Test that empty period_aggregations raises ValueError."""
        with pytest.raises(ValueError, match="period_aggregations cannot be empty"):
            assess_customer_base_health(
                period_aggregations=[],
                cohort_assignments={"C1": "2024-Q1"},
                analysis_start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                analysis_end_date=datetime(2024, 12, 31, tzinfo=timezone.utc),
            )

    def test_empty_cohort_assignments_raises_error(self):
        """Test that empty cohort_assignments raises ValueError."""
        period = PeriodAggregation(
            "C1",
            datetime(2024, 1, 1, tzinfo=timezone.utc),
            datetime(2024, 2, 1, tzinfo=timezone.utc),
            2,
            100.0,
            20.0,
            5,
        )
        with pytest.raises(ValueError, match="cohort_assignments cannot be empty"):
            assess_customer_base_health(
                period_aggregations=[period],
                cohort_assignments={},
                analysis_start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                analysis_end_date=datetime(2024, 12, 31, tzinfo=timezone.utc),
            )

    def test_invalid_date_range_raises_error(self):
        """Test that start >= end raises ValueError."""
        period = PeriodAggregation(
            "C1",
            datetime(2024, 1, 1, tzinfo=timezone.utc),
            datetime(2024, 2, 1, tzinfo=timezone.utc),
            2,
            100.0,
            20.0,
            5,
        )
        with pytest.raises(ValueError, match="must be before"):
            assess_customer_base_health(
                period_aggregations=[period],
                cohort_assignments={"C1": "2024-Q1"},
                analysis_start_date=datetime(2024, 12, 31, tzinfo=timezone.utc),
                analysis_end_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
            )

    def test_no_periods_in_window_raises_error(self):
        """Test that no periods in analysis window raises ValueError."""
        period = PeriodAggregation(
            "C1",
            datetime(2023, 1, 1, tzinfo=timezone.utc),  # Before window
            datetime(2023, 2, 1, tzinfo=timezone.utc),
            2,
            100.0,
            20.0,
            5,
        )
        with pytest.raises(ValueError, match="No periods found in analysis window"):
            assess_customer_base_health(
                period_aggregations=[period],
                cohort_assignments={"C1": "2024-Q1"},
                analysis_start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                analysis_end_date=datetime(2024, 12, 31, tzinfo=timezone.utc),
            )

    def test_basic_health_assessment(self):
        """Test basic health assessment with minimal data."""
        periods = [
            PeriodAggregation(
                "C1",
                datetime(2024, 1, 1, tzinfo=timezone.utc),
                datetime(2024, 2, 1, tzinfo=timezone.utc),
                2,
                100.0,
                20.0,
                5,
            ),
            PeriodAggregation(
                "C2",
                datetime(2024, 1, 1, tzinfo=timezone.utc),
                datetime(2024, 2, 1, tzinfo=timezone.utc),
                3,
                150.0,
                30.0,
                7,
            ),
        ]
        cohort_assignments = {"C1": "2024-Q1", "C2": "2024-Q1"}

        metrics = assess_customer_base_health(
            period_aggregations=periods,
            cohort_assignments=cohort_assignments,
            analysis_start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
            analysis_end_date=datetime(2024, 12, 31, tzinfo=timezone.utc),
        )

        # Verify results structure
        assert isinstance(metrics, Lens5Metrics)
        assert len(metrics.cohort_revenue_contributions) > 0
        assert len(metrics.cohort_repeat_behavior) > 0
        assert isinstance(metrics.health_score, CustomerBaseHealthScore)
        assert metrics.health_score.total_customers == 2
        assert metrics.health_score.total_active_customers == 2
        assert Decimal("0") <= metrics.health_score.health_score <= Decimal("100")
        assert metrics.health_score.health_grade in ("A", "B", "C", "D", "F")
