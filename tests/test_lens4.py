"""Unit tests for Lens 4: Comparing and Contrasting Cohort Performance."""

from datetime import datetime
from decimal import Decimal

import pytest

from customer_base_audit.analyses.lens4 import (
    CohortDecomposition,
    Lens4Metrics,
    calculate_cohort_decomposition,
    compare_cohorts,
)
from customer_base_audit.foundation.data_mart import PeriodAggregation


class TestCohortDecomposition:
    """Test suite for CohortDecomposition dataclass."""

    def test_valid_cohort_decomposition(self):
        """Test creation of valid CohortDecomposition."""
        decomp = CohortDecomposition(
            cohort_id="2024-Q1",
            period_number=0,
            cohort_size=100,
            active_customers=80,
            pct_active=Decimal("80.00"),
            total_orders=120,
            aof=Decimal("1.50"),
            total_revenue=Decimal("6000.00"),
            aov=Decimal("50.00"),
            margin=Decimal("100.00"),
            revenue=Decimal("6000.00"),
        )
        assert decomp.cohort_id == "2024-Q1"
        assert decomp.period_number == 0
        assert decomp.cohort_size == 100
        assert decomp.active_customers == 80

    def test_negative_period_number_raises_error(self):
        """Test that negative period_number raises ValueError."""
        with pytest.raises(ValueError, match="period_number must be >= 0"):
            CohortDecomposition(
                cohort_id="2024-Q1",
                period_number=-1,
                cohort_size=100,
                active_customers=80,
                pct_active=Decimal("80.00"),
                total_orders=120,
                aof=Decimal("1.50"),
                total_revenue=Decimal("6000.00"),
                aov=Decimal("50.00"),
                margin=Decimal("100.00"),
                revenue=Decimal("6000.00"),
            )

    def test_negative_cohort_size_raises_error(self):
        """Test that negative cohort_size raises ValueError."""
        with pytest.raises(ValueError, match="cohort_size must be >= 0"):
            CohortDecomposition(
                cohort_id="2024-Q1",
                period_number=0,
                cohort_size=-1,
                active_customers=80,
                pct_active=Decimal("80.00"),
                total_orders=120,
                aof=Decimal("1.50"),
                total_revenue=Decimal("6000.00"),
                aov=Decimal("50.00"),
                margin=Decimal("100.00"),
                revenue=Decimal("6000.00"),
            )

    def test_active_customers_exceeds_cohort_size_raises_error(self):
        """Test that active_customers > cohort_size raises ValueError."""
        with pytest.raises(
            ValueError, match="active_customers .* cannot exceed cohort_size"
        ):
            CohortDecomposition(
                cohort_id="2024-Q1",
                period_number=0,
                cohort_size=100,
                active_customers=150,
                pct_active=Decimal("150.00"),
                total_orders=120,
                aof=Decimal("1.50"),
                total_revenue=Decimal("6000.00"),
                aov=Decimal("50.00"),
                margin=Decimal("100.00"),
                revenue=Decimal("6000.00"),
            )

    def test_invalid_pct_active_raises_error(self):
        """Test that pct_active outside [0, 100] raises ValueError."""
        with pytest.raises(ValueError, match="pct_active must be in"):
            CohortDecomposition(
                cohort_id="2024-Q1",
                period_number=0,
                cohort_size=100,
                active_customers=80,
                pct_active=Decimal("150.00"),
                total_orders=120,
                aof=Decimal("1.50"),
                total_revenue=Decimal("6000.00"),
                aov=Decimal("50.00"),
                margin=Decimal("100.00"),
                revenue=Decimal("6000.00"),
            )

    def test_negative_total_orders_raises_error(self):
        """Test that negative total_orders raises ValueError."""
        with pytest.raises(ValueError, match="total_orders must be >= 0"):
            CohortDecomposition(
                cohort_id="2024-Q1",
                period_number=0,
                cohort_size=100,
                active_customers=80,
                pct_active=Decimal("80.00"),
                total_orders=-1,
                aof=Decimal("1.50"),
                total_revenue=Decimal("6000.00"),
                aov=Decimal("50.00"),
                margin=Decimal("100.00"),
                revenue=Decimal("6000.00"),
            )

    def test_negative_aof_raises_error(self):
        """Test that negative aof raises ValueError."""
        with pytest.raises(ValueError, match="aof must be >= 0"):
            CohortDecomposition(
                cohort_id="2024-Q1",
                period_number=0,
                cohort_size=100,
                active_customers=80,
                pct_active=Decimal("80.00"),
                total_orders=120,
                aof=Decimal("-1.50"),
                total_revenue=Decimal("6000.00"),
                aov=Decimal("50.00"),
                margin=Decimal("100.00"),
                revenue=Decimal("6000.00"),
            )

    def test_negative_revenue_raises_error(self):
        """Test that negative revenue raises ValueError."""
        with pytest.raises(ValueError, match="revenue must be >= 0"):
            CohortDecomposition(
                cohort_id="2024-Q1",
                period_number=0,
                cohort_size=100,
                active_customers=80,
                pct_active=Decimal("80.00"),
                total_orders=120,
                aof=Decimal("1.50"),
                total_revenue=Decimal("6000.00"),
                aov=Decimal("50.00"),
                margin=Decimal("100.00"),
                revenue=Decimal("-100.00"),
            )


class TestCalculateCohortDecomposition:
    """Test suite for calculate_cohort_decomposition helper function."""

    def test_calculate_decomposition_with_active_customers(self):
        """Test decomposition calculation with active customers."""
        period_data = [
            PeriodAggregation(
                "cust1",
                datetime(2024, 1, 1),
                datetime(2024, 2, 1),
                2,
                100.0,
                25.0,
                5,
            ),
            PeriodAggregation(
                "cust2",
                datetime(2024, 1, 1),
                datetime(2024, 2, 1),
                3,
                150.0,
                30.0,
                7,
            ),
        ]

        decomp = calculate_cohort_decomposition(
            cohort_id="2024-Q1",
            period_number=0,
            cohort_size=10,
            period_data=period_data,
            include_margin=True,
        )

        assert decomp.cohort_id == "2024-Q1"
        assert decomp.period_number == 0
        assert decomp.cohort_size == 10
        assert decomp.active_customers == 2
        assert decomp.pct_active == Decimal("20.00")
        assert decomp.total_orders == 5
        assert decomp.aof == Decimal("2.50")
        assert decomp.total_revenue == Decimal("250.00")
        assert decomp.aov == Decimal("50.00")
        assert decomp.margin == Decimal("22.00")  # (25+30)/250 = 22%
        assert decomp.revenue == Decimal("55.00")  # 10 * 0.20 * 2.50 * 50.00 * 0.22

    def test_calculate_decomposition_without_margin(self):
        """Test decomposition calculation without margin."""
        period_data = [
            PeriodAggregation(
                "cust1",
                datetime(2024, 1, 1),
                datetime(2024, 2, 1),
                2,
                100.0,
                25.0,
                5,
            ),
        ]

        decomp = calculate_cohort_decomposition(
            cohort_id="2024-Q1",
            period_number=0,
            cohort_size=5,
            period_data=period_data,
            include_margin=False,
        )

        assert decomp.margin == Decimal("100.00")  # Revenue-only analysis

    def test_calculate_decomposition_zero_customers(self):
        """Test decomposition with zero active customers."""
        decomp = calculate_cohort_decomposition(
            cohort_id="2024-Q1",
            period_number=0,
            cohort_size=10,
            period_data=[],
            include_margin=False,
        )

        assert decomp.active_customers == 0
        assert decomp.pct_active == Decimal("0.00")
        assert decomp.total_orders == 0
        assert decomp.aof == Decimal("0.00")
        assert decomp.total_revenue == Decimal("0.00")
        assert decomp.aov == Decimal("0.00")
        assert decomp.revenue == Decimal("0.00")

    def test_calculate_decomposition_zero_orders(self):
        """Test decomposition with active customers but zero orders."""
        # Edge case: customer appears but has 0 orders
        period_data = [
            PeriodAggregation(
                "cust1",
                datetime(2024, 1, 1),
                datetime(2024, 2, 1),
                0,
                0.0,
                0.0,
                0,
            ),
        ]

        decomp = calculate_cohort_decomposition(
            cohort_id="2024-Q1",
            period_number=0,
            cohort_size=5,
            period_data=period_data,
            include_margin=False,
        )

        assert decomp.active_customers == 1
        assert decomp.total_orders == 0
        assert decomp.aof == Decimal("0.00")
        assert decomp.aov == Decimal("0.00")


class TestCompareCohorts:
    """Test suite for compare_cohorts main function."""

    def test_compare_cohorts_empty_input(self):
        """Test compare_cohorts with empty period aggregations."""
        metrics = compare_cohorts(
            period_aggregations=[],
            cohort_assignments={},
            alignment_type="left-aligned",
        )

        assert isinstance(metrics, Lens4Metrics)
        assert len(metrics.cohort_decompositions) == 0
        assert len(metrics.time_to_second_purchase) == 0
        assert len(metrics.cohort_comparisons) == 0

    def test_compare_cohorts_invalid_alignment_type(self):
        """Test compare_cohorts with invalid alignment_type."""
        with pytest.raises(ValueError, match="alignment_type must be"):
            compare_cohorts(
                period_aggregations=[],
                cohort_assignments={},
                alignment_type="invalid",
            )

    def test_compare_cohorts_time_aligned_not_implemented(self):
        """Test that time-aligned mode raises NotImplementedError."""
        with pytest.raises(
            NotImplementedError, match="Time-aligned comparison not yet implemented"
        ):
            compare_cohorts(
                period_aggregations=[],
                cohort_assignments={},
                alignment_type="time-aligned",
            )

    def test_compare_cohorts_left_aligned_single_cohort(self):
        """Test left-aligned comparison with single cohort."""
        period_data = [
            PeriodAggregation(
                "cust1",
                datetime(2024, 1, 1),
                datetime(2024, 2, 1),
                2,
                100.0,
                20.0,
                5,
            ),
            PeriodAggregation(
                "cust1",
                datetime(2024, 2, 1),
                datetime(2024, 3, 1),
                1,
                50.0,
                10.0,
                2,
            ),
            PeriodAggregation(
                "cust2",
                datetime(2024, 1, 1),
                datetime(2024, 2, 1),
                1,
                75.0,
                15.0,
                3,
            ),
        ]

        cohort_assignments = {
            "cust1": "2024-Q1",
            "cust2": "2024-Q1",
        }

        metrics = compare_cohorts(
            period_aggregations=period_data,
            cohort_assignments=cohort_assignments,
            alignment_type="left-aligned",
            include_margin=True,
        )

        assert isinstance(metrics, Lens4Metrics)
        assert len(metrics.cohort_decompositions) > 0
        assert metrics.alignment_type == "left-aligned"

        # Verify decompositions exist for the cohort
        cohort_decomps = [
            d for d in metrics.cohort_decompositions if d.cohort_id == "2024-Q1"
        ]
        assert len(cohort_decomps) > 0

    def test_compare_cohorts_left_aligned_multiple_cohorts(self):
        """Test left-aligned comparison with multiple cohorts."""
        period_data = [
            # Cohort 2024-Q1: Period 0 (2024-01)
            PeriodAggregation(
                "cust1",
                datetime(2024, 1, 1),
                datetime(2024, 2, 1),
                2,
                100.0,
                20.0,
                5,
            ),
            # Cohort 2024-Q1: Period 1 (2024-02)
            PeriodAggregation(
                "cust1",
                datetime(2024, 2, 1),
                datetime(2024, 3, 1),
                1,
                50.0,
                10.0,
                2,
            ),
            # Cohort 2024-Q2: Period 0 (2024-04)
            PeriodAggregation(
                "cust2",
                datetime(2024, 4, 1),
                datetime(2024, 5, 1),
                3,
                150.0,
                30.0,
                7,
            ),
        ]

        cohort_assignments = {
            "cust1": "2024-Q1",
            "cust2": "2024-Q2",
        }

        metrics = compare_cohorts(
            period_aggregations=period_data,
            cohort_assignments=cohort_assignments,
            alignment_type="left-aligned",
            include_margin=True,
        )

        assert isinstance(metrics, Lens4Metrics)
        assert len(metrics.cohort_decompositions) >= 2

        # Verify both cohorts have decompositions
        q1_decomps = [
            d for d in metrics.cohort_decompositions if d.cohort_id == "2024-Q1"
        ]
        q2_decomps = [
            d for d in metrics.cohort_decompositions if d.cohort_id == "2024-Q2"
        ]
        assert len(q1_decomps) > 0
        assert len(q2_decomps) > 0

    def test_compare_cohorts_customers_without_cohort_assignment(self):
        """Test that customers without cohort assignment are skipped."""
        period_data = [
            PeriodAggregation(
                "cust1",
                datetime(2024, 1, 1),
                datetime(2024, 2, 1),
                2,
                100.0,
                20.0,
                5,
            ),
            PeriodAggregation(
                "cust2",
                datetime(2024, 1, 1),
                datetime(2024, 2, 1),
                1,
                50.0,
                10.0,
                3,
            ),
        ]

        cohort_assignments = {
            "cust1": "2024-Q1",
            # cust2 has no cohort assignment
        }

        metrics = compare_cohorts(
            period_aggregations=period_data,
            cohort_assignments=cohort_assignments,
            alignment_type="left-aligned",
        )

        # Should only process cust1
        assert len(metrics.cohort_decompositions) > 0
        # Verify only cust1's cohort appears
        cohorts = {d.cohort_id for d in metrics.cohort_decompositions}
        assert "2024-Q1" in cohorts


class TestLens4Metrics:
    """Test suite for Lens4Metrics dataclass."""

    def test_valid_lens4_metrics(self):
        """Test creation of valid Lens4Metrics."""
        decomp = CohortDecomposition(
            cohort_id="2024-Q1",
            period_number=0,
            cohort_size=100,
            active_customers=80,
            pct_active=Decimal("80.00"),
            total_orders=120,
            aof=Decimal("1.50"),
            total_revenue=Decimal("6000.00"),
            aov=Decimal("50.00"),
            margin=Decimal("100.00"),
            revenue=Decimal("6000.00"),
        )

        metrics = Lens4Metrics(
            alignment_type="left-aligned",
            cohort_decompositions=[decomp],
            time_to_second_purchase=[],
            cohort_comparisons=[],
        )

        assert metrics.alignment_type == "left-aligned"
        assert len(metrics.cohort_decompositions) == 1
        assert metrics.cohort_decompositions[0].cohort_id == "2024-Q1"

    def test_invalid_alignment_type_raises_error(self):
        """Test that invalid alignment_type raises ValueError."""
        with pytest.raises(ValueError, match="alignment_type must be"):
            Lens4Metrics(
                alignment_type="invalid",
                cohort_decompositions=[],
                time_to_second_purchase=[],
                cohort_comparisons=[],
            )

    def test_lens4_metrics_sorting_by_cohort_and_period(self):
        """Test that cohort_decompositions must be sorted by cohort_id and period_number."""
        decomp1 = CohortDecomposition(
            cohort_id="2024-Q1",
            period_number=0,
            cohort_size=100,
            active_customers=80,
            pct_active=Decimal("80.00"),
            total_orders=120,
            aof=Decimal("1.50"),
            total_revenue=Decimal("6000.00"),
            aov=Decimal("50.00"),
            margin=Decimal("100.00"),
            revenue=Decimal("6000.00"),
        )

        decomp2 = CohortDecomposition(
            cohort_id="2024-Q2",
            period_number=1,
            cohort_size=100,
            active_customers=80,
            pct_active=Decimal("80.00"),
            total_orders=120,
            aof=Decimal("1.50"),
            total_revenue=Decimal("6000.00"),
            aov=Decimal("50.00"),
            margin=Decimal("100.00"),
            revenue=Decimal("6000.00"),
        )

        metrics = Lens4Metrics(
            alignment_type="left-aligned",
            cohort_decompositions=[decomp1, decomp2],
            time_to_second_purchase=[],
            cohort_comparisons=[],
        )

        # Verify they are sorted correctly by cohort_id
        assert metrics.cohort_decompositions[0].cohort_id == "2024-Q1"
        assert metrics.cohort_decompositions[1].cohort_id == "2024-Q2"


class TestDeterminismAndReconciliation:
    """Test suite for determinism and revenue reconciliation."""

    def test_compare_cohorts_deterministic(self):
        """Test that compare_cohorts produces deterministic results."""
        period_data = [
            PeriodAggregation(
                "cust1",
                datetime(2024, 1, 1),
                datetime(2024, 2, 1),
                2,
                100.0,
                20.0,
                5,
            ),
            PeriodAggregation(
                "cust2",
                datetime(2024, 1, 1),
                datetime(2024, 2, 1),
                3,
                150.0,
                30.0,
                7,
            ),
        ]

        cohort_assignments = {
            "cust1": "2024-Q1",
            "cust2": "2024-Q1",
        }

        result1 = compare_cohorts(
            period_data, cohort_assignments, alignment_type="left-aligned"
        )
        result2 = compare_cohorts(
            period_data, cohort_assignments, alignment_type="left-aligned"
        )

        # Results should be identical
        assert len(result1.cohort_decompositions) == len(result2.cohort_decompositions)
        for d1, d2 in zip(result1.cohort_decompositions, result2.cohort_decompositions):
            assert d1.cohort_id == d2.cohort_id
            assert d1.period_number == d2.period_number
            assert d1.cohort_size == d2.cohort_size
            assert d1.active_customers == d2.active_customers
            assert d1.pct_active == d2.pct_active
            assert d1.total_orders == d2.total_orders
            assert d1.aof == d2.aof
            assert d1.total_revenue == d2.total_revenue
            assert d1.aov == d2.aov
            assert d1.margin == d2.margin
            assert d1.revenue == d2.revenue

    def test_revenue_reconciliation_with_heterogeneous_customers(self):
        """Revenue decomposition should approximately match total_revenue with heterogeneous customers."""
        # Create heterogeneous customer data: one low-freq/high-spend, one high-freq/low-spend
        period_data = [
            PeriodAggregation(
                "c1",
                datetime(2024, 1, 1),
                datetime(2024, 2, 1),
                1,  # Low frequency
                100.0,  # High spend
                20.0,
                1,
            ),
            PeriodAggregation(
                "c2",
                datetime(2024, 1, 1),
                datetime(2024, 2, 1),
                10,  # High frequency
                50.0,  # Low spend
                10.0,
                15,
            ),
        ]

        cohort_assignments = {
            "c1": "2024-Q1",
            "c2": "2024-Q1",
        }

        metrics = compare_cohorts(
            period_data,
            cohort_assignments,
            alignment_type="left-aligned",
            include_margin=True,
        )

        decomp = metrics.cohort_decompositions[0]

        # Total revenue should be exact sum
        assert decomp.total_revenue == Decimal("150.00")

        # Decomposed revenue uses averages, so will differ due to heterogeneity
        # Error can be significant (< 100%) for extreme heterogeneity
        if decomp.total_revenue > 0:
            error_pct = (
                abs(decomp.revenue - decomp.total_revenue) / decomp.total_revenue * 100
            )
            # This test demonstrates the reconciliation issue with heterogeneous customers
            # Error of ~80% is expected when customers have vastly different behavior
            assert error_pct < Decimal(
                "100.00"
            )  # Within 100% for extreme heterogeneity

    def test_calculate_decomposition_zero_revenue_with_margin(self):
        """Test that margin is 0.00 when include_margin=True but total_revenue=0."""
        period_data = [
            PeriodAggregation(
                "cust1",
                datetime(2024, 1, 1),
                datetime(2024, 2, 1),
                0,
                0.0,
                0.0,
                0,
            ),
        ]

        decomp = calculate_cohort_decomposition(
            cohort_id="2024-Q1",
            period_number=0,
            cohort_size=5,
            period_data=period_data,
            include_margin=True,
        )

        # With zero revenue, margin should be 0.00, not 100.00
        assert decomp.margin == Decimal("0.00")
        assert decomp.total_revenue == Decimal("0.00")
        assert decomp.revenue == Decimal("0.00")
