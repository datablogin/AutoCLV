"""Lens 2: Period-to-Period Comparison Analysis.

Tracks customer migration patterns and metric changes between two adjacent periods,
answering questions like:
- How many customers were retained vs. churned?
- How many new customers appeared?
- How many churned customers reactivated?
- What changed in revenue, AOV, and frequency between periods?

This is the second lens from "The Customer-Base Audit" framework.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from typing import Sequence

from customer_base_audit.analyses.lens1 import Lens1Metrics, analyze_single_period
from customer_base_audit.foundation.rfm import RFMMetrics


@dataclass(frozen=True)
class CustomerMigration:
    """Customer movement between two periods.

    Attributes
    ----------
    retained:
        Set of customer IDs active in both period 1 and period 2
    churned:
        Set of customer IDs active in period 1 but not in period 2
    new:
        Set of customer IDs not in period 1 but active in period 2
        (This includes both truly new customers and reactivated ones)
    reactivated:
        Set of customer IDs not in period 1, active in period 2,
        but were active in some earlier historical period
    """

    retained: frozenset[str]
    churned: frozenset[str]
    new: frozenset[str]
    reactivated: frozenset[str]

    def __post_init__(self) -> None:
        """Validate customer migration sets."""
        # Retained and churned should not overlap
        overlap_retained_churned = self.retained & self.churned
        if overlap_retained_churned:
            raise ValueError(
                f"Customer IDs cannot be both retained and churned: {overlap_retained_churned}"
            )

        # Retained and new should not overlap
        overlap_retained_new = self.retained & self.new
        if overlap_retained_new:
            raise ValueError(
                f"Customer IDs cannot be both retained and new: {overlap_retained_new}"
            )

        # Churned and new should not overlap
        overlap_churned_new = self.churned & self.new
        if overlap_churned_new:
            raise ValueError(
                f"Customer IDs cannot be both churned and new: {overlap_churned_new}"
            )

        # Reactivated must be a subset of new
        if not self.reactivated.issubset(self.new):
            raise ValueError(
                f"Reactivated customers must be a subset of new customers: "
                f"reactivated={self.reactivated}, new={self.new}"
            )


@dataclass(frozen=True)
class Lens2Metrics:
    """Lens 2: Period-to-period comparison results.

    Attributes
    ----------
    period1_metrics:
        Lens 1 metrics for the first period
    period2_metrics:
        Lens 1 metrics for the second period
    migration:
        Customer migration patterns (retained, churned, new, reactivated)
    retention_rate:
        Percentage of period 1 customers who are still active in period 2
    churn_rate:
        Percentage of period 1 customers who are not active in period 2
    reactivation_rate:
        Percentage of period 2 customers who are reactivations
    customer_count_change:
        Change in customer count from period 1 to period 2 (period2 - period1)
    revenue_change_pct:
        Percentage change in revenue from period 1 to period 2
    avg_order_value_change_pct:
        Percentage change in average order value from period 1 to period 2
    """

    period1_metrics: Lens1Metrics
    period2_metrics: Lens1Metrics
    migration: CustomerMigration
    retention_rate: Decimal
    churn_rate: Decimal
    reactivation_rate: Decimal
    customer_count_change: int
    revenue_change_pct: Decimal
    avg_order_value_change_pct: Decimal

    def __post_init__(self) -> None:
        """Validate Lens 2 metrics."""
        if not 0 <= self.retention_rate <= 100:
            raise ValueError(
                f"Retention rate must be 0-100: {self.retention_rate}"
            )
        if not 0 <= self.churn_rate <= 100:
            raise ValueError(
                f"Churn rate must be 0-100: {self.churn_rate}"
            )
        if not 0 <= self.reactivation_rate <= 100:
            raise ValueError(
                f"Reactivation rate must be 0-100: {self.reactivation_rate}"
            )

        # Retention + churn should equal 100% (within rounding tolerance)
        # Exception: when period1 is empty, both rates are 0 (which is valid)
        rate_sum = self.retention_rate + self.churn_rate
        tolerance = Decimal("0.1")
        if rate_sum > 0 and abs(rate_sum - 100) > tolerance:
            raise ValueError(
                f"Retention rate ({self.retention_rate}) + churn rate ({self.churn_rate}) "
                f"must equal 100, got {rate_sum}"
            )


def analyze_period_comparison(
    period1_rfm: Sequence[RFMMetrics],
    period2_rfm: Sequence[RFMMetrics],
    all_customer_history: Sequence[str] | None = None,
) -> Lens2Metrics:
    """Compare two adjacent periods to identify customer migration patterns.

    Parameters
    ----------
    period1_rfm:
        RFM metrics for all customers active in period 1
    period2_rfm:
        RFM metrics for all customers active in period 2
    all_customer_history:
        Optional list of all customer IDs ever seen (across all historical periods).
        If provided, enables identification of reactivated customers (those who
        were inactive in period 1 but were active in some earlier period).
        If not provided, all "new" customers in period 2 will be considered
        truly new (reactivated will be empty).

    Returns
    -------
    Lens2Metrics
        Comprehensive period-to-period comparison results

    Examples
    --------
    >>> from decimal import Decimal
    >>> from datetime import datetime
    >>> from customer_base_audit.foundation.rfm import RFMMetrics
    >>> period1 = [
    ...     RFMMetrics("C1", 10, 5, Decimal("50"), datetime(2023,1,1), datetime(2023,6,30), Decimal("250")),
    ...     RFMMetrics("C2", 30, 2, Decimal("100"), datetime(2023,1,1), datetime(2023,6,30), Decimal("200")),
    ... ]
    >>> period2 = [
    ...     RFMMetrics("C1", 5, 3, Decimal("60"), datetime(2023,7,1), datetime(2023,12,31), Decimal("180")),
    ...     RFMMetrics("C3", 2, 1, Decimal("150"), datetime(2023,7,1), datetime(2023,12,31), Decimal("150")),
    ... ]
    >>> lens2 = analyze_period_comparison(period1, period2)
    >>> len(lens2.migration.retained)
    1
    >>> len(lens2.migration.churned)
    1
    >>> len(lens2.migration.new)
    1
    >>> float(lens2.retention_rate)
    50.0
    """
    # Calculate Lens 1 metrics for each period
    lens1_period1 = analyze_single_period(period1_rfm)
    lens1_period2 = analyze_single_period(period2_rfm)

    # Extract customer sets
    period1_customers = frozenset(m.customer_id for m in period1_rfm)
    period2_customers = frozenset(m.customer_id for m in period2_rfm)

    # Calculate migration patterns
    retained = period1_customers & period2_customers
    churned = period1_customers - period2_customers
    new = period2_customers - period1_customers

    # Identify reactivated customers (if history provided)
    if all_customer_history:
        all_history_set = frozenset(all_customer_history)
        # Reactivated = customers who are new in period 2 but were seen before period 1
        reactivated = new & (all_history_set - period1_customers)
    else:
        reactivated = frozenset()

    migration = CustomerMigration(
        retained=retained,
        churned=churned,
        new=new,
        reactivated=reactivated,
    )

    # Calculate rates
    if len(period1_customers) > 0:
        retention_rate = (
            Decimal(len(retained)) / Decimal(len(period1_customers)) * 100
        ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        churn_rate = (
            Decimal(len(churned)) / Decimal(len(period1_customers)) * 100
        ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    else:
        retention_rate = Decimal("0")
        churn_rate = Decimal("0")

    if len(period2_customers) > 0:
        reactivation_rate = (
            Decimal(len(reactivated)) / Decimal(len(period2_customers)) * 100
        ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    else:
        reactivation_rate = Decimal("0")

    # Calculate customer count change
    customer_count_change = len(period2_customers) - len(period1_customers)

    # Calculate revenue change percentage
    if lens1_period1.total_revenue > 0:
        revenue_change_pct = (
            (lens1_period2.total_revenue - lens1_period1.total_revenue)
            / lens1_period1.total_revenue
            * 100
        ).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)
    else:
        # If period 1 had zero revenue, treat any period 2 revenue as 100% increase
        revenue_change_pct = Decimal("100.0") if lens1_period2.total_revenue > 0 else Decimal("0")

    # Calculate average order value change percentage
    # AOV = total_revenue / total_orders
    period1_total_orders = sum(m.frequency for m in period1_rfm)
    period2_total_orders = sum(m.frequency for m in period2_rfm)

    if period1_total_orders > 0 and period2_total_orders > 0:
        period1_aov = lens1_period1.total_revenue / Decimal(period1_total_orders)
        period2_aov = lens1_period2.total_revenue / Decimal(period2_total_orders)
        # Only calculate percentage change if period1 AOV is non-zero
        if period1_aov > 0:
            avg_order_value_change_pct = (
                (period2_aov - period1_aov) / period1_aov * 100
            ).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)
        else:
            # If period1 AOV is 0, treat any period2 AOV as 100% increase
            avg_order_value_change_pct = Decimal("100.0") if period2_aov > 0 else Decimal("0")
    else:
        avg_order_value_change_pct = Decimal("0")

    return Lens2Metrics(
        period1_metrics=lens1_period1,
        period2_metrics=lens1_period2,
        migration=migration,
        retention_rate=retention_rate,
        churn_rate=churn_rate,
        reactivation_rate=reactivation_rate,
        customer_count_change=customer_count_change,
        revenue_change_pct=revenue_change_pct,
        avg_order_value_change_pct=avg_order_value_change_pct,
    )
