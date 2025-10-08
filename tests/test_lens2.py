"""Tests for Lens 2: Period-to-Period Comparison Analysis."""

from datetime import datetime
from decimal import Decimal

import pytest

from customer_base_audit.analyses.lens2 import (
    CustomerMigration,
    Lens2Metrics,
    analyze_period_comparison,
)
from customer_base_audit.analyses.lens1 import Lens1Metrics
from customer_base_audit.foundation.rfm import RFMMetrics


class TestCustomerMigration:
    """Test CustomerMigration dataclass validation."""

    def test_valid_customer_migration(self):
        """Valid customer migration should be created successfully."""
        migration = CustomerMigration(
            retained=frozenset(["C1", "C2"]),
            churned=frozenset(["C3"]),
            new=frozenset(["C4", "C5"]),
            reactivated=frozenset(["C5"]),
        )
        assert len(migration.retained) == 2
        assert len(migration.churned) == 1
        assert len(migration.new) == 2
        assert len(migration.reactivated) == 1

    def test_retained_and_churned_overlap_raises_error(self):
        """Overlap between retained and churned should raise ValueError."""
        with pytest.raises(
            ValueError, match="cannot be both retained and churned"
        ):
            CustomerMigration(
                retained=frozenset(["C1", "C2"]),
                churned=frozenset(["C2", "C3"]),
                new=frozenset(["C4"]),
                reactivated=frozenset(),
            )

    def test_retained_and_new_overlap_raises_error(self):
        """Overlap between retained and new should raise ValueError."""
        with pytest.raises(
            ValueError, match="cannot be both retained and new"
        ):
            CustomerMigration(
                retained=frozenset(["C1", "C2"]),
                churned=frozenset(["C3"]),
                new=frozenset(["C2", "C4"]),
                reactivated=frozenset(),
            )

    def test_churned_and_new_overlap_raises_error(self):
        """Overlap between churned and new should raise ValueError."""
        with pytest.raises(
            ValueError, match="cannot be both churned and new"
        ):
            CustomerMigration(
                retained=frozenset(["C1"]),
                churned=frozenset(["C2"]),
                new=frozenset(["C2", "C4"]),
                reactivated=frozenset(),
            )

    def test_reactivated_not_subset_of_new_raises_error(self):
        """Reactivated not being a subset of new should raise ValueError."""
        with pytest.raises(
            ValueError, match="Reactivated customers must be a subset of new"
        ):
            CustomerMigration(
                retained=frozenset(["C1"]),
                churned=frozenset(["C2"]),
                new=frozenset(["C4"]),
                reactivated=frozenset(["C5"]),  # C5 not in new
            )

    def test_empty_migration(self):
        """Empty migration (no customers) should be valid."""
        migration = CustomerMigration(
            retained=frozenset(),
            churned=frozenset(),
            new=frozenset(),
            reactivated=frozenset(),
        )
        assert len(migration.retained) == 0
        assert len(migration.churned) == 0


class TestLens2Metrics:
    """Test Lens2Metrics dataclass validation."""

    def create_valid_lens1_metrics(
        self, customers: int = 100, revenue: Decimal = Decimal("10000")
    ) -> Lens1Metrics:
        """Helper to create valid Lens1Metrics."""
        return Lens1Metrics(
            total_customers=customers,
            one_time_buyers=30,
            one_time_buyer_pct=Decimal("30.00"),
            total_revenue=revenue,
            top_10pct_revenue_contribution=Decimal("45.0"),
            top_20pct_revenue_contribution=Decimal("65.0"),
            avg_orders_per_customer=Decimal("3.50"),
            median_customer_value=Decimal("75.00"),
            rfm_distribution={"555": 5},
        )

    def test_valid_lens2_metrics(self):
        """Valid Lens 2 metrics should be created successfully."""
        lens1_p1 = self.create_valid_lens1_metrics(customers=100)
        lens1_p2 = self.create_valid_lens1_metrics(customers=120)
        migration = CustomerMigration(
            retained=frozenset(["C1", "C2"]),
            churned=frozenset(["C3"]),
            new=frozenset(["C4"]),
            reactivated=frozenset(),
        )

        metrics = Lens2Metrics(
            period1_metrics=lens1_p1,
            period2_metrics=lens1_p2,
            migration=migration,
            retention_rate=Decimal("75.00"),
            churn_rate=Decimal("25.00"),
            reactivation_rate=Decimal("10.00"),
            customer_count_change=20,
            revenue_change_pct=Decimal("15.5"),
            avg_order_value_change_pct=Decimal("5.2"),
        )
        assert metrics.retention_rate == Decimal("75.00")
        assert metrics.customer_count_change == 20

    def test_invalid_retention_rate_raises_error(self):
        """Retention rate outside 0-100 should raise ValueError."""
        lens1 = self.create_valid_lens1_metrics()
        migration = CustomerMigration(
            retained=frozenset(), churned=frozenset(), new=frozenset(), reactivated=frozenset()
        )

        with pytest.raises(ValueError, match="Retention rate must be 0-100"):
            Lens2Metrics(
                period1_metrics=lens1,
                period2_metrics=lens1,
                migration=migration,
                retention_rate=Decimal("150.00"),
                churn_rate=Decimal("-50.00"),
                reactivation_rate=Decimal("0"),
                customer_count_change=0,
                revenue_change_pct=Decimal("0"),
                avg_order_value_change_pct=Decimal("0"),
            )

    def test_retention_and_churn_must_sum_to_100(self):
        """Retention + churn must equal 100% (within tolerance)."""
        lens1 = self.create_valid_lens1_metrics()
        migration = CustomerMigration(
            retained=frozenset(), churned=frozenset(), new=frozenset(), reactivated=frozenset()
        )

        with pytest.raises(
            ValueError, match="Retention rate .* \\+ churn rate .* must equal 100"
        ):
            Lens2Metrics(
                period1_metrics=lens1,
                period2_metrics=lens1,
                migration=migration,
                retention_rate=Decimal("60.00"),
                churn_rate=Decimal("30.00"),  # Sum = 90, not 100
                reactivation_rate=Decimal("0"),
                customer_count_change=0,
                revenue_change_pct=Decimal("0"),
                avg_order_value_change_pct=Decimal("0"),
            )


class TestAnalyzePeriodComparison:
    """Test analyze_period_comparison function."""

    def create_rfm(
        self, customer_id: str, frequency: int, spend: Decimal, date: datetime
    ) -> RFMMetrics:
        """Helper to create RFMMetrics."""
        return RFMMetrics(
            customer_id=customer_id,
            recency_days=10,
            frequency=frequency,
            monetary=spend / frequency,
            observation_start=datetime(2023, 1, 1),
            observation_end=date,
            total_spend=spend,
        )

    def test_basic_period_comparison(self):
        """Basic two-period comparison should calculate metrics correctly."""
        period1 = [
            self.create_rfm("C1", 5, Decimal("250"), datetime(2023, 6, 30)),
            self.create_rfm("C2", 2, Decimal("200"), datetime(2023, 6, 30)),
        ]
        period2 = [
            self.create_rfm("C1", 3, Decimal("180"), datetime(2023, 12, 31)),
            self.create_rfm("C3", 1, Decimal("150"), datetime(2023, 12, 31)),
        ]

        lens2 = analyze_period_comparison(period1, period2)

        # Check migration
        assert "C1" in lens2.migration.retained
        assert "C2" in lens2.migration.churned
        assert "C3" in lens2.migration.new
        assert len(lens2.migration.retained) == 1
        assert len(lens2.migration.churned) == 1
        assert len(lens2.migration.new) == 1

        # Check rates
        assert lens2.retention_rate == Decimal("50.00")  # 1/2
        assert lens2.churn_rate == Decimal("50.00")  # 1/2
        assert lens2.reactivation_rate == Decimal("0")  # No history provided

        # Check counts
        assert lens2.customer_count_change == 0  # 2 - 2

    def test_100_percent_retention(self):
        """All customers retained should yield 100% retention, 0% churn."""
        period1 = [
            self.create_rfm("C1", 5, Decimal("250"), datetime(2023, 6, 30)),
            self.create_rfm("C2", 2, Decimal("200"), datetime(2023, 6, 30)),
        ]
        period2 = [
            self.create_rfm("C1", 3, Decimal("180"), datetime(2023, 12, 31)),
            self.create_rfm("C2", 4, Decimal("220"), datetime(2023, 12, 31)),
        ]

        lens2 = analyze_period_comparison(period1, period2)

        assert len(lens2.migration.retained) == 2
        assert len(lens2.migration.churned) == 0
        assert lens2.retention_rate == Decimal("100.00")
        assert lens2.churn_rate == Decimal("0.00")

    def test_100_percent_churn(self):
        """No customers retained should yield 0% retention, 100% churn."""
        period1 = [
            self.create_rfm("C1", 5, Decimal("250"), datetime(2023, 6, 30)),
            self.create_rfm("C2", 2, Decimal("200"), datetime(2023, 6, 30)),
        ]
        period2 = [
            self.create_rfm("C3", 1, Decimal("150"), datetime(2023, 12, 31)),
            self.create_rfm("C4", 2, Decimal("100"), datetime(2023, 12, 31)),
        ]

        lens2 = analyze_period_comparison(period1, period2)

        assert len(lens2.migration.retained) == 0
        assert len(lens2.migration.churned) == 2
        assert len(lens2.migration.new) == 2
        assert lens2.retention_rate == Decimal("0.00")
        assert lens2.churn_rate == Decimal("100.00")

    def test_reactivated_customers_with_history(self):
        """Providing customer history should identify reactivated customers."""
        period1 = [
            self.create_rfm("C1", 5, Decimal("250"), datetime(2023, 6, 30)),
        ]
        period2 = [
            self.create_rfm("C1", 3, Decimal("180"), datetime(2023, 12, 31)),
            self.create_rfm("C2", 1, Decimal("100"), datetime(2023, 12, 31)),  # Reactivated
            self.create_rfm("C3", 1, Decimal("150"), datetime(2023, 12, 31)),  # Truly new
        ]
        all_history = ["C1", "C2"]  # C2 was seen before period1

        lens2 = analyze_period_comparison(period1, period2, all_customer_history=all_history)

        assert "C2" in lens2.migration.reactivated
        assert "C3" not in lens2.migration.reactivated
        assert len(lens2.migration.reactivated) == 1
        assert lens2.reactivation_rate == Decimal("33.33")  # 1/3 customers

    def test_reactivated_without_history(self):
        """Without history, all new customers are considered truly new."""
        period1 = [
            self.create_rfm("C1", 5, Decimal("250"), datetime(2023, 6, 30)),
        ]
        period2 = [
            self.create_rfm("C1", 3, Decimal("180"), datetime(2023, 12, 31)),
            self.create_rfm("C2", 1, Decimal("100"), datetime(2023, 12, 31)),
        ]

        lens2 = analyze_period_comparison(period1, period2)

        assert len(lens2.migration.reactivated) == 0
        assert lens2.reactivation_rate == Decimal("0")

    def test_revenue_change_calculation(self):
        """Revenue change percentage should be calculated correctly."""
        period1 = [
            self.create_rfm("C1", 10, Decimal("1000"), datetime(2023, 6, 30)),
        ]
        period2 = [
            self.create_rfm("C1", 10, Decimal("1200"), datetime(2023, 12, 31)),
        ]

        lens2 = analyze_period_comparison(period1, period2)

        # Revenue increased from 1000 to 1200 = +20%
        assert lens2.revenue_change_pct == Decimal("20.0")

    def test_revenue_decrease(self):
        """Revenue decrease should yield negative change percentage."""
        period1 = [
            self.create_rfm("C1", 10, Decimal("1000"), datetime(2023, 6, 30)),
        ]
        period2 = [
            self.create_rfm("C1", 10, Decimal("800"), datetime(2023, 12, 31)),
        ]

        lens2 = analyze_period_comparison(period1, period2)

        # Revenue decreased from 1000 to 800 = -20%
        assert lens2.revenue_change_pct == Decimal("-20.0")

    def test_zero_revenue_in_period1(self):
        """Zero revenue in period 1 should be handled gracefully."""
        period1 = [
            self.create_rfm("C1", 1, Decimal("0"), datetime(2023, 6, 30)),
        ]
        period2 = [
            self.create_rfm("C1", 10, Decimal("1000"), datetime(2023, 12, 31)),
        ]

        lens2 = analyze_period_comparison(period1, period2)

        # From 0 to 1000 = 100% increase (by convention)
        assert lens2.revenue_change_pct == Decimal("100.0")

    def test_avg_order_value_change(self):
        """Average order value change should be calculated correctly."""
        period1 = [
            self.create_rfm("C1", 10, Decimal("1000"), datetime(2023, 6, 30)),
        ]
        period2 = [
            self.create_rfm("C1", 10, Decimal("1200"), datetime(2023, 12, 31)),
        ]

        lens2 = analyze_period_comparison(period1, period2)

        # AOV P1 = 1000/10 = 100
        # AOV P2 = 1200/10 = 120
        # Change = (120-100)/100 = +20%
        assert lens2.avg_order_value_change_pct == Decimal("20.0")

    def test_customer_count_change_increase(self):
        """Customer count increase should be positive."""
        period1 = [
            self.create_rfm("C1", 5, Decimal("250"), datetime(2023, 6, 30)),
        ]
        period2 = [
            self.create_rfm("C1", 3, Decimal("180"), datetime(2023, 12, 31)),
            self.create_rfm("C2", 1, Decimal("100"), datetime(2023, 12, 31)),
            self.create_rfm("C3", 2, Decimal("150"), datetime(2023, 12, 31)),
        ]

        lens2 = analyze_period_comparison(period1, period2)

        assert lens2.customer_count_change == 2  # 3 - 1

    def test_customer_count_change_decrease(self):
        """Customer count decrease should be negative."""
        period1 = [
            self.create_rfm("C1", 5, Decimal("250"), datetime(2023, 6, 30)),
            self.create_rfm("C2", 2, Decimal("200"), datetime(2023, 6, 30)),
            self.create_rfm("C3", 1, Decimal("100"), datetime(2023, 6, 30)),
        ]
        period2 = [
            self.create_rfm("C1", 3, Decimal("180"), datetime(2023, 12, 31)),
        ]

        lens2 = analyze_period_comparison(period1, period2)

        assert lens2.customer_count_change == -2  # 1 - 3

    def test_empty_period1(self):
        """Empty period 1 should be handled gracefully."""
        period1: list[RFMMetrics] = []
        period2 = [
            self.create_rfm("C1", 3, Decimal("180"), datetime(2023, 12, 31)),
        ]

        lens2 = analyze_period_comparison(period1, period2)

        assert lens2.retention_rate == Decimal("0")
        assert lens2.churn_rate == Decimal("0")
        assert len(lens2.migration.new) == 1

    def test_empty_period2(self):
        """Empty period 2 should show 100% churn."""
        period1 = [
            self.create_rfm("C1", 5, Decimal("250"), datetime(2023, 6, 30)),
        ]
        period2: list[RFMMetrics] = []

        lens2 = analyze_period_comparison(period1, period2)

        assert lens2.churn_rate == Decimal("100.00")
        assert lens2.retention_rate == Decimal("0.00")
        assert len(lens2.migration.churned) == 1

    def test_both_periods_empty(self):
        """Both periods empty should be handled gracefully."""
        period1: list[RFMMetrics] = []
        period2: list[RFMMetrics] = []

        lens2 = analyze_period_comparison(period1, period2)

        assert lens2.retention_rate == Decimal("0")
        assert lens2.churn_rate == Decimal("0")
        assert lens2.customer_count_change == 0

    def test_lens1_metrics_integration(self):
        """Lens 2 should correctly embed Lens 1 metrics for both periods."""
        period1 = [
            self.create_rfm("C1", 5, Decimal("250"), datetime(2023, 6, 30)),
            self.create_rfm("C2", 1, Decimal("100"), datetime(2023, 6, 30)),
        ]
        period2 = [
            self.create_rfm("C1", 3, Decimal("180"), datetime(2023, 12, 31)),
        ]

        lens2 = analyze_period_comparison(period1, period2)

        # Verify Lens 1 metrics are calculated
        assert lens2.period1_metrics.total_customers == 2
        assert lens2.period1_metrics.total_revenue == Decimal("350.00")
        assert lens2.period2_metrics.total_customers == 1
        assert lens2.period2_metrics.total_revenue == Decimal("180.00")

    def test_migration_matrix_reconciliation(self):
        """Migration matrix should reconcile: period1 = retained + churned."""
        period1 = [
            self.create_rfm("C1", 5, Decimal("250"), datetime(2023, 6, 30)),
            self.create_rfm("C2", 2, Decimal("200"), datetime(2023, 6, 30)),
            self.create_rfm("C3", 1, Decimal("100"), datetime(2023, 6, 30)),
        ]
        period2 = [
            self.create_rfm("C1", 3, Decimal("180"), datetime(2023, 12, 31)),
            self.create_rfm("C3", 2, Decimal("150"), datetime(2023, 12, 31)),
            self.create_rfm("C4", 1, Decimal("100"), datetime(2023, 12, 31)),
        ]

        lens2 = analyze_period_comparison(period1, period2)

        # Period 1 customers = retained + churned
        period1_count = len(period1)
        reconciled_count = len(lens2.migration.retained) + len(lens2.migration.churned)
        assert period1_count == reconciled_count

        # Period 2 customers = retained + new
        period2_count = len(period2)
        reconciled_p2_count = len(lens2.migration.retained) + len(lens2.migration.new)
        assert period2_count == reconciled_p2_count
