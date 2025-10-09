"""Lens 3: Single cohort evolution tracking.

This module implements Lens 3 from "The Customer-Base Audit" framework,
tracking how a single cohort's behavior evolves over time from first purchase.
Key analyses include retention curves, revenue decay patterns, and purchase
frequency evolution.

Quick Start
-----------
>>> from datetime import datetime
>>> from customer_base_audit.foundation.data_mart import PeriodAggregation
>>> from customer_base_audit.analyses.lens3 import analyze_cohort_evolution
>>>
>>> # Analyze cohort acquired in January 2023
>>> period_aggregations = [...]  # Your period aggregations
>>> cohort_customer_ids = ["C1", "C2", "C3"]  # Customer IDs in this cohort
>>> metrics = analyze_cohort_evolution(
...     cohort_name="2023-01",
...     acquisition_date=datetime(2023, 1, 1),
...     period_aggregations=period_aggregations,
...     cohort_customer_ids=cohort_customer_ids
... )
>>> print(f"Cohort size: {metrics.cohort_size}")
>>> print(f"Cumulative activation at period 3: {metrics.periods[3].cumulative_activation_rate}")
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Sequence, Mapping

from customer_base_audit.foundation.data_mart import PeriodAggregation


@dataclass(frozen=True)
class CohortPeriodMetrics:
    """Metrics for a cohort in a specific period after acquisition.

    Attributes
    ----------
    period_number:
        Periods since acquisition (0 = acquisition period).
    active_customers:
        Number of customers from the cohort who made purchases this period.
    cumulative_activation_rate:
        Cumulative percentage of cohort that has made at least one purchase since
        acquisition (includes customers active in any previous period). This is NOT
        period-specific retention; it tracks ever-active customers, so it never decreases.
    avg_orders_per_customer:
        Average orders per active customer in this period (active customers only).
    avg_revenue_per_customer:
        Average revenue per active customer in this period (active customers only).
    avg_orders_per_cohort_member:
        Average orders per original cohort member (includes churned customers as zeros).
    avg_revenue_per_cohort_member:
        Average revenue per original cohort member (includes churned customers as zeros).
    total_revenue:
        Total revenue from the cohort in this period.
    """

    period_number: int
    active_customers: int
    cumulative_activation_rate: float
    avg_orders_per_customer: float
    avg_revenue_per_customer: float
    avg_orders_per_cohort_member: float
    avg_revenue_per_cohort_member: float
    total_revenue: float

    def __post_init__(self) -> None:
        """Validate cohort period metrics constraints."""
        if self.period_number < 0:
            raise ValueError(f"period_number must be >= 0, got {self.period_number}")
        if self.active_customers < 0:
            raise ValueError(
                f"active_customers must be >= 0, got {self.active_customers}"
            )
        if not 0 <= self.cumulative_activation_rate <= 1:
            raise ValueError(
                f"cumulative_activation_rate must be between 0 and 1, got {self.cumulative_activation_rate}"
            )
        if self.avg_orders_per_customer < 0:
            raise ValueError(
                f"avg_orders_per_customer must be >= 0, got {self.avg_orders_per_customer}"
            )
        if self.avg_revenue_per_customer < 0:
            raise ValueError(
                f"avg_revenue_per_customer must be >= 0, got {self.avg_revenue_per_customer}"
            )
        if self.avg_orders_per_cohort_member < 0:
            raise ValueError(
                f"avg_orders_per_cohort_member must be >= 0, got {self.avg_orders_per_cohort_member}"
            )
        if self.avg_revenue_per_cohort_member < 0:
            raise ValueError(
                f"avg_revenue_per_cohort_member must be >= 0, got {self.avg_revenue_per_cohort_member}"
            )
        if self.total_revenue < 0:
            raise ValueError(f"total_revenue must be >= 0, got {self.total_revenue}")


@dataclass(frozen=True)
class Lens3Metrics:
    """Lens 3: Single cohort evolution results.

    Attributes
    ----------
    cohort_name:
        Identifier for the cohort (e.g., "2023-01", "Q1-2023").
    acquisition_date:
        Start date of the cohort's acquisition period.
    cohort_size:
        Initial customer count in the cohort.
    periods:
        List of CohortPeriodMetrics ordered by period_number.
    """

    cohort_name: str
    acquisition_date: datetime
    cohort_size: int
    periods: Sequence[CohortPeriodMetrics]

    def __post_init__(self) -> None:
        """Validate Lens3Metrics constraints."""
        if self.cohort_size < 0:
            raise ValueError(f"cohort_size must be >= 0, got {self.cohort_size}")

        # Validate periods are ordered by period_number
        if self.periods:
            period_numbers = [p.period_number for p in self.periods]
            if period_numbers != sorted(period_numbers):
                raise ValueError("periods must be ordered by period_number")

            # Validate period numbers are contiguous starting from 0
            expected = list(range(len(self.periods)))
            if period_numbers != expected:
                raise ValueError(
                    f"period_numbers must be contiguous starting from 0. "
                    f"Expected {expected}, got {period_numbers}"
                )


def analyze_cohort_evolution(
    cohort_name: str,
    acquisition_date: datetime,
    period_aggregations: Sequence[PeriodAggregation],
    cohort_customer_ids: Sequence[str],
) -> Lens3Metrics:
    """Track how a single cohort's behavior evolves over time.

    This function analyzes the evolution of a single acquisition cohort by
    tracking retention, purchase frequency, and revenue patterns across
    periods following acquisition.

    Parameters
    ----------
    cohort_name:
        Identifier for the cohort (e.g., "2023-01", "Q1-2023").
    acquisition_date:
        Start date of the cohort's acquisition period. Used to align periods
        for cohort analysis (period 0 = acquisition period).
    period_aggregations:
        List of period-level customer aggregations containing all transaction
        data. Will be filtered to only include customers in this cohort.
    cohort_customer_ids:
        List of customer IDs belonging to this cohort. Used to determine
        cohort size and filter period aggregations.

    Returns
    -------
    Lens3Metrics
        Cohort evolution metrics including retention curves, revenue decay,
        and purchase frequency patterns.

    Raises
    ------
    ValueError
        If cohort_customer_ids is empty or if no period aggregations match
        the cohort customers.

    Notes
    -----
    Key Analyses:
    - Retention curve: % of original cohort active each period
    - Revenue decay patterns: Total and per-customer revenue over time
    - Purchase frequency evolution: Orders per active customer by period
    - Time to second purchase distribution (implicit in period 1 metrics)

    The analysis assumes period_start dates are normalized to period boundaries
    (e.g., start of month for monthly periods). Periods are numbered sequentially
    starting from 0 (acquisition period) based on chronological order.

    Examples
    --------
    >>> from datetime import datetime
    >>> from customer_base_audit.foundation.data_mart import PeriodAggregation
    >>>
    >>> # Define cohort and periods
    >>> cohort_customers = ["C1", "C2", "C3"]
    >>> acquisition_date = datetime(2023, 1, 1)
    >>>
    >>> # Period aggregations for cohort customers
    >>> periods = [
    ...     PeriodAggregation("C1", datetime(2023, 1, 1), datetime(2023, 2, 1), 2, 100.0, 30.0, 5),
    ...     PeriodAggregation("C2", datetime(2023, 1, 1), datetime(2023, 2, 1), 1, 50.0, 15.0, 2),
    ...     PeriodAggregation("C1", datetime(2023, 2, 1), datetime(2023, 3, 1), 1, 75.0, 20.0, 3),
    ... ]
    >>>
    >>> metrics = analyze_cohort_evolution(
    ...     cohort_name="2023-01",
    ...     acquisition_date=acquisition_date,
    ...     period_aggregations=periods,
    ...     cohort_customer_ids=cohort_customers
    ... )
    >>> print(f"Cohort size: {metrics.cohort_size}")
    Cohort size: 3
    >>> print(f"Period 0 cumulative activation: {metrics.periods[0].cumulative_activation_rate:.2%}")
    Period 0 cumulative activation: 100.00%
    """
    if not cohort_customer_ids:
        raise ValueError("cohort_customer_ids cannot be empty")

    # Validate no duplicate customer IDs
    cohort_customer_set = set(cohort_customer_ids)
    if len(cohort_customer_set) != len(cohort_customer_ids):
        num_duplicates = len(cohort_customer_ids) - len(cohort_customer_set)
        raise ValueError(
            f"cohort_customer_ids contains {num_duplicates} duplicate IDs. "
            "Each customer should appear exactly once."
        )
    cohort_size = len(cohort_customer_set)

    # Filter period aggregations to only include cohort customers
    cohort_periods = [
        p for p in period_aggregations if p.customer_id in cohort_customer_set
    ]

    if not cohort_periods:
        raise ValueError(
            f"No period aggregations found for cohort {cohort_name}. "
            f"Check that cohort_customer_ids match customer_id values in period_aggregations."
        )

    # Group aggregations by period_start
    periods_by_date: dict[datetime, list[PeriodAggregation]] = {}
    for period in cohort_periods:
        if period.period_start not in periods_by_date:
            periods_by_date[period.period_start] = []
        periods_by_date[period.period_start].append(period)

    # Sort periods chronologically and assign period numbers
    sorted_dates = sorted(periods_by_date.keys())

    # Find the index of the acquisition period
    # Acquisition period is the first period >= acquisition_date
    acquisition_period_idx = None
    for idx, period_date in enumerate(sorted_dates):
        if period_date >= acquisition_date:
            acquisition_period_idx = idx
            break

    if acquisition_period_idx is None:
        raise ValueError(
            f"No periods found on or after acquisition_date {acquisition_date.isoformat()}. "
            f"Earliest period: {sorted_dates[0].isoformat()}"
        )

    # Calculate metrics for each period, starting from acquisition period
    cohort_period_metrics: list[CohortPeriodMetrics] = []
    cumulative_active_customers: set[str] = (
        set()
    )  # Track customers active in any period

    for period_number, period_date in enumerate(
        sorted_dates[acquisition_period_idx:], start=0
    ):
        period_data = periods_by_date[period_date]

        # Calculate period-specific active customers
        period_active_customers = set(p.customer_id for p in period_data)
        active_customers = len(period_active_customers)

        # Update cumulative active set (for cumulative activation tracking)
        cumulative_active_customers.update(period_active_customers)
        cumulative_activation_rate = len(cumulative_active_customers) / cohort_size

        total_orders = sum(p.total_orders for p in period_data)
        total_revenue = sum(p.total_spend for p in period_data)

        # Per-active-customer metrics (only includes active customers this period)
        avg_orders_per_customer = (
            total_orders / active_customers if active_customers > 0 else 0.0
        )
        avg_revenue_per_customer = (
            total_revenue / active_customers if active_customers > 0 else 0.0
        )

        # Per-cohort-member metrics (includes all cohort members, churned as zeros)
        avg_orders_per_cohort_member = total_orders / cohort_size
        avg_revenue_per_cohort_member = total_revenue / cohort_size

        cohort_period_metrics.append(
            CohortPeriodMetrics(
                period_number=period_number,
                active_customers=active_customers,
                cumulative_activation_rate=cumulative_activation_rate,
                avg_orders_per_customer=avg_orders_per_customer,
                avg_revenue_per_customer=avg_revenue_per_customer,
                avg_orders_per_cohort_member=avg_orders_per_cohort_member,
                avg_revenue_per_cohort_member=avg_revenue_per_cohort_member,
                total_revenue=total_revenue,
            )
        )

    return Lens3Metrics(
        cohort_name=cohort_name,
        acquisition_date=acquisition_date,
        cohort_size=cohort_size,
        periods=cohort_period_metrics,
    )


def calculate_retention_curve(cohort_metrics: Lens3Metrics) -> Mapping[int, float]:
    """Extract cumulative activation rates by period number.

    Note: Despite the function name (kept for backward compatibility), this
    returns cumulative activation rates, not period-specific retention rates.

    Parameters
    ----------
    cohort_metrics:
        Lens3Metrics containing cohort evolution data.

    Returns
    -------
    Mapping[int, float]
        Dictionary mapping period_number to cumulative_activation_rate.

    Examples
    --------
    >>> from datetime import datetime
    >>> metrics = Lens3Metrics(
    ...     cohort_name="2023-01",
    ...     acquisition_date=datetime(2023, 1, 1),
    ...     cohort_size=100,
    ...     periods=[
    ...         CohortPeriodMetrics(0, 100, 1.0, 1.5, 50.0, 5000.0),
    ...         CohortPeriodMetrics(1, 80, 0.8, 1.2, 40.0, 3200.0),
    ...         CohortPeriodMetrics(2, 60, 0.6, 1.0, 35.0, 2100.0),
    ...     ]
    ... )
    >>> curve = calculate_retention_curve(metrics)
    >>> print(curve)
    {0: 1.0, 1: 0.8, 2: 0.6}
    """
    return {
        period.period_number: period.cumulative_activation_rate for period in cohort_metrics.periods
    }
