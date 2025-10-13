"""Lens 5: Overall Customer Base Health.

This lens provides an integrative view of customer base health by:
- Analyzing revenue contributions across cohorts (C3 data)
- Tracking repeat purchase behavior by cohort
- Calculating overall health score and grade
- Assessing cohort quality trends and revenue predictability

Reference: "The Customer-Base Audit" by Fader, Hardie, and Ross (Chapter 7)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from decimal import ROUND_HALF_UP, Decimal
from typing import Sequence

from customer_base_audit.foundation.data_mart import PeriodAggregation

# Module-level constants
PERCENTAGE_PRECISION = Decimal("0.01")  # 2 decimal places

# Health score weights (must sum to 1.0)
RETENTION_WEIGHT = Decimal("0.30")  # 30% weight for overall retention
QUALITY_WEIGHT = Decimal("0.30")  # 30% weight for cohort quality trend
PREDICTABILITY_WEIGHT = Decimal("0.20")  # 20% weight for revenue predictability
INDEPENDENCE_WEIGHT = Decimal("0.20")  # 20% weight for acquisition independence

# Health grade thresholds
GRADE_A_THRESHOLD = Decimal("90")
GRADE_B_THRESHOLD = Decimal("80")
GRADE_C_THRESHOLD = Decimal("70")
GRADE_D_THRESHOLD = Decimal("60")

# Validate that health score weights sum to 1.0
_WEIGHTS_SUM = (
    RETENTION_WEIGHT + QUALITY_WEIGHT + PREDICTABILITY_WEIGHT + INDEPENDENCE_WEIGHT
)
assert _WEIGHTS_SUM == Decimal("1.0"), (
    f"Health score weights must sum to 1.0, got {_WEIGHTS_SUM}"
)

logger = logging.getLogger(__name__)


def _validate_percentage(value: Decimal, field_name: str) -> None:
    """Validate that a value is a valid percentage (0-100).

    Parameters
    ----------
    value:
        The value to validate
    field_name:
        The name of the field (for error messages)

    Raises
    ------
    ValueError:
        If value is not in [0, 100]
    """
    if not (Decimal("0") <= value <= Decimal("100")):
        raise ValueError(f"{field_name} must be in [0, 100], got {value}")


@dataclass(frozen=True)
class CohortRevenuePeriod:
    """Revenue contribution from one cohort in one calendar period.

    This data powers Customer Cohort Chart (C3) visualizations.

    Attributes
    ----------
    cohort_id:
        Cohort identifier
    period_start:
        Calendar period start date
    total_revenue:
        Total revenue from this cohort in this period
    pct_of_period_revenue:
        Percentage of period's total revenue from this cohort (0-100)
    active_customers:
        Number of active customers from this cohort in this period
    avg_revenue_per_customer:
        Average revenue per active customer in this period
    """

    cohort_id: str
    period_start: datetime
    total_revenue: Decimal
    pct_of_period_revenue: Decimal
    active_customers: int
    avg_revenue_per_customer: Decimal

    def __post_init__(self) -> None:
        """Validate cohort revenue period metrics."""
        if self.total_revenue < 0:
            raise ValueError(f"total_revenue must be >= 0, got {self.total_revenue}")
        _validate_percentage(self.pct_of_period_revenue, "pct_of_period_revenue")
        if self.active_customers < 0:
            raise ValueError(
                f"active_customers must be >= 0, got {self.active_customers}"
            )
        if self.avg_revenue_per_customer < 0:
            raise ValueError(
                f"avg_revenue_per_customer must be >= 0, "
                f"got {self.avg_revenue_per_customer}"
            )


@dataclass(frozen=True)
class CohortRepeatBehavior:
    """Repeat purchase behavior for a cohort.

    Attributes
    ----------
    cohort_id:
        Cohort identifier
    cohort_size:
        Total customers in cohort who have made at least 1 purchase
    one_time_buyers:
        Customers with only 1 purchase
    repeat_buyers:
        Customers with 2+ purchases
    repeat_rate:
        Percentage of cohort with 2+ purchases (0-100)
    avg_orders_per_repeat_buyer:
        Average orders for customers with 2+ purchases.
        When cohort_size is 0, this will be 0.00 (N/A)
    """

    cohort_id: str
    cohort_size: int
    one_time_buyers: int
    repeat_buyers: int
    repeat_rate: Decimal
    avg_orders_per_repeat_buyer: Decimal

    def __post_init__(self) -> None:
        """Validate cohort repeat behavior."""
        if self.cohort_size < 0:
            raise ValueError(f"cohort_size must be >= 0, got {self.cohort_size}")
        if self.one_time_buyers < 0:
            raise ValueError(
                f"one_time_buyers must be >= 0, got {self.one_time_buyers}"
            )
        if self.repeat_buyers < 0:
            raise ValueError(f"repeat_buyers must be >= 0, got {self.repeat_buyers}")
        if self.one_time_buyers + self.repeat_buyers != self.cohort_size:
            raise ValueError(
                f"one_time_buyers ({self.one_time_buyers}) + repeat_buyers "
                f"({self.repeat_buyers}) must equal cohort_size ({self.cohort_size})"
            )
        _validate_percentage(self.repeat_rate, "repeat_rate")
        if self.repeat_buyers > 0 and self.avg_orders_per_repeat_buyer < 2:
            raise ValueError(
                f"avg_orders_per_repeat_buyer must be >= 2 (by definition), "
                f"got {self.avg_orders_per_repeat_buyer}"
            )


@dataclass(frozen=True)
class CustomerBaseHealthScore:
    """Overall customer base health assessment.

    Attributes
    ----------
    total_customers:
        Total unique customers across all cohorts
    total_active_customers:
        Currently active customers (in analysis period)
    overall_retention_rate:
        Percentage of historical customers still active (0-100)
    cohort_quality_trend:
        "improving", "stable", or "declining" based on cohort metrics
    revenue_predictability_pct:
        Percentage of next period's revenue predictable from existing cohorts (0-100)
    acquisition_dependence_pct:
        Percentage of current revenue from newest cohort (0-100)
    health_score:
        Overall health score (0-100), higher is better
    health_grade:
        Letter grade: A (90-100), B (80-89), C (70-79), D (60-69), F (<60)
    """

    total_customers: int
    total_active_customers: int
    overall_retention_rate: Decimal
    cohort_quality_trend: str
    revenue_predictability_pct: Decimal
    acquisition_dependence_pct: Decimal
    health_score: Decimal
    health_grade: str

    def __post_init__(self) -> None:
        """Validate customer base health score."""
        if self.total_customers < 0:
            raise ValueError(f"total_customers must be >= 0, got {self.total_customers}")
        if self.total_active_customers < 0:
            raise ValueError(
                f"total_active_customers must be >= 0, got {self.total_active_customers}"
            )
        if self.total_active_customers > self.total_customers:
            raise ValueError(
                f"total_active_customers ({self.total_active_customers}) cannot exceed "
                f"total_customers ({self.total_customers})"
            )
        _validate_percentage(self.overall_retention_rate, "overall_retention_rate")
        if self.cohort_quality_trend not in ("improving", "stable", "declining"):
            raise ValueError(
                f"cohort_quality_trend must be 'improving', 'stable', or 'declining', "
                f"got '{self.cohort_quality_trend}'"
            )
        _validate_percentage(self.revenue_predictability_pct, "revenue_predictability_pct")
        _validate_percentage(self.acquisition_dependence_pct, "acquisition_dependence_pct")
        _validate_percentage(self.health_score, "health_score")
        if self.health_grade not in ("A", "B", "C", "D", "F"):
            raise ValueError(
                f"health_grade must be 'A', 'B', 'C', 'D', or 'F', "
                f"got '{self.health_grade}'"
            )


def calculate_health_score(
    overall_retention: Decimal,
    cohort_quality_trend: str,
    revenue_predictability: Decimal,
    acquisition_dependence: Decimal,
) -> tuple[Decimal, str]:
    """Calculate overall health score and grade from component metrics.

    Formula:
        score = (retention × 0.30) + (quality × 0.30) +
                (predictability × 0.20) + ((100-dependence) × 0.20)

    Where:
        - retention: overall_retention (0-100)
        - quality: 80 (improving), 50 (stable), 20 (declining)
        - predictability: revenue_predictability (0-100)
        - independence: 100 - acquisition_dependence (0-100)

    Parameters
    ----------
    overall_retention:
        Overall retention rate (0-100)
    cohort_quality_trend:
        "improving", "stable", or "declining"
    revenue_predictability:
        Revenue predictability percentage (0-100)
    acquisition_dependence:
        Acquisition dependence percentage (0-100)

    Returns
    -------
    tuple[Decimal, str]:
        (health_score, health_grade) where score is 0-100 and grade is A-F

    Examples
    --------
    >>> calculate_health_score(Decimal("75.00"), "stable", Decimal("60.00"), Decimal("30.00"))
    (Decimal('67.50'), 'C')
    """
    # Convert quality trend to numeric score
    if cohort_quality_trend == "improving":
        quality_score = Decimal("80")
    elif cohort_quality_trend == "stable":
        quality_score = Decimal("50")
    else:  # declining
        quality_score = Decimal("20")

    # Calculate independence score (higher is better)
    independence_score = Decimal("100") - acquisition_dependence

    # Calculate weighted score
    health_score = (
        overall_retention * RETENTION_WEIGHT
        + quality_score * QUALITY_WEIGHT
        + revenue_predictability * PREDICTABILITY_WEIGHT
        + independence_score * INDEPENDENCE_WEIGHT
    ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

    # Determine letter grade
    if health_score >= GRADE_A_THRESHOLD:
        health_grade = "A"
    elif health_score >= GRADE_B_THRESHOLD:
        health_grade = "B"
    elif health_score >= GRADE_C_THRESHOLD:
        health_grade = "C"
    elif health_score >= GRADE_D_THRESHOLD:
        health_grade = "D"
    else:
        health_grade = "F"

    return health_score, health_grade


@dataclass(frozen=True)
class Lens5Metrics:
    """Lens 5: Overall customer base health analysis results.

    Attributes
    ----------
    cohort_revenue_contributions:
        Revenue contribution from each cohort in each period (for C3 chart).
        Sorted by period_start, then cohort_id.
    cohort_repeat_behavior:
        Repeat purchase behavior for each cohort.
        Sorted by cohort_id.
    health_score:
        Overall customer base health assessment
    analysis_start_date:
        Start of analysis period
    analysis_end_date:
        End of analysis period
    """

    cohort_revenue_contributions: Sequence[CohortRevenuePeriod]
    cohort_repeat_behavior: Sequence[CohortRepeatBehavior]
    health_score: CustomerBaseHealthScore
    analysis_start_date: datetime
    analysis_end_date: datetime

    def __post_init__(self) -> None:
        """Validate Lens 5 metrics."""
        if self.analysis_start_date >= self.analysis_end_date:
            raise ValueError(
                f"analysis_start_date ({self.analysis_start_date}) must be before "
                f"analysis_end_date ({self.analysis_end_date})"
            )

        # Validate cohort_revenue_contributions are sorted
        if self.cohort_revenue_contributions:
            for i in range(len(self.cohort_revenue_contributions) - 1):
                curr = self.cohort_revenue_contributions[i]
                next_item = self.cohort_revenue_contributions[i + 1]

                if curr.period_start > next_item.period_start:
                    raise ValueError(
                        "cohort_revenue_contributions must be sorted by period_start"
                    )
                elif curr.period_start == next_item.period_start:
                    if curr.cohort_id >= next_item.cohort_id:
                        raise ValueError(
                            "cohort_revenue_contributions must be sorted by cohort_id "
                            "within each period"
                        )


def _calculate_c3_data(
    periods: Sequence[PeriodAggregation],
    cohort_assignments: dict[str, str],
) -> list[CohortRevenuePeriod]:
    """Calculate C3 data (revenue by cohort-period)."""
    # Group by (cohort_id, period_start)
    cohort_period_revenue: dict[tuple[str, datetime], dict] = {}

    for period in periods:
        customer_id = period.customer_id
        cohort_id = cohort_assignments.get(customer_id)
        if cohort_id is None:
            continue

        key = (cohort_id, period.period_start)
        if key not in cohort_period_revenue:
            cohort_period_revenue[key] = {
                "total_revenue": Decimal("0"),
                "active_customers": set(),
            }

        cohort_period_revenue[key]["total_revenue"] += Decimal(str(period.total_spend))
        cohort_period_revenue[key]["active_customers"].add(customer_id)

    # Calculate total revenue per period for percentages
    period_totals: dict[datetime, Decimal] = {}
    for (cohort_id, period_start), data in cohort_period_revenue.items():
        period_totals[period_start] = (
            period_totals.get(period_start, Decimal("0")) + data["total_revenue"]
        )

    # Build results sorted by (period_start, cohort_id)
    results: list[CohortRevenuePeriod] = []
    for (cohort_id, period_start), data in sorted(
        cohort_period_revenue.items(), key=lambda x: (x[0][1], x[0][0])
    ):
        total_revenue = data["total_revenue"].quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )
        active_customers = len(data["active_customers"])

        period_total = period_totals[period_start]
        if period_total > 0:
            pct_of_period_revenue = (total_revenue / period_total * 100).quantize(
                PERCENTAGE_PRECISION, rounding=ROUND_HALF_UP
            )
        else:
            pct_of_period_revenue = Decimal("0.00")

        if active_customers > 0:
            avg_revenue_per_customer = (
                total_revenue / Decimal(str(active_customers))
            ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        else:
            avg_revenue_per_customer = Decimal("0.00")

        results.append(
            CohortRevenuePeriod(
                cohort_id=cohort_id,
                period_start=period_start,
                total_revenue=total_revenue,
                pct_of_period_revenue=pct_of_period_revenue,
                active_customers=active_customers,
                avg_revenue_per_customer=avg_revenue_per_customer,
            )
        )

    return results


def _calculate_repeat_behavior(
    periods: Sequence[PeriodAggregation],
    cohort_assignments: dict[str, str],
) -> list[CohortRepeatBehavior]:
    """Calculate repeat purchase behavior by cohort."""
    # Count total orders per customer
    customer_orders: dict[str, int] = {}

    for period in periods:
        customer_id = period.customer_id
        customer_orders[customer_id] = (
            customer_orders.get(customer_id, 0) + period.total_orders
        )

    # Group by cohort
    cohort_data: dict[str, dict] = {}
    for customer_id, cohort_id in cohort_assignments.items():
        if cohort_id not in cohort_data:
            cohort_data[cohort_id] = {
                "cohort_size": 0,
                "one_time_buyers": 0,
                "repeat_buyers": 0,
                "repeat_orders": [],
            }

        # Classify based on total orders
        orders = customer_orders.get(customer_id, 0)
        if orders == 1:
            cohort_data[cohort_id]["cohort_size"] += 1
            cohort_data[cohort_id]["one_time_buyers"] += 1
        elif orders >= 2:
            cohort_data[cohort_id]["cohort_size"] += 1
            cohort_data[cohort_id]["repeat_buyers"] += 1
            cohort_data[cohort_id]["repeat_orders"].append(orders)
        # orders == 0 means customer hasn't purchased yet (not counted)

    # Build results
    results: list[CohortRepeatBehavior] = []
    for cohort_id, data in sorted(cohort_data.items()):
        cohort_size = data["cohort_size"]
        one_time_buyers = data["one_time_buyers"]
        repeat_buyers = data["repeat_buyers"]

        if cohort_size > 0:
            repeat_rate = (
                Decimal(str(repeat_buyers)) / Decimal(str(cohort_size)) * 100
            ).quantize(PERCENTAGE_PRECISION, rounding=ROUND_HALF_UP)
        else:
            repeat_rate = Decimal("0.00")

        if repeat_buyers > 0:
            avg_orders_per_repeat_buyer = (
                Decimal(sum(data["repeat_orders"])) / Decimal(repeat_buyers)
            ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        else:
            avg_orders_per_repeat_buyer = Decimal("0.00")

        results.append(
            CohortRepeatBehavior(
                cohort_id=cohort_id,
                cohort_size=cohort_size,
                one_time_buyers=one_time_buyers,
                repeat_buyers=repeat_buyers,
                repeat_rate=repeat_rate,
                avg_orders_per_repeat_buyer=avg_orders_per_repeat_buyer,
            )
        )

    return results


def assess_customer_base_health(
    period_aggregations: Sequence[PeriodAggregation],
    cohort_assignments: dict[str, str],  # customer_id -> cohort_id
    analysis_start_date: datetime,
    analysis_end_date: datetime,
) -> Lens5Metrics:
    """Assess overall customer base health (Lens 5).

    Provides integrative view of customer base by analyzing revenue contributions
    across cohorts, repeat purchase behavior, and overall health indicators.

    Parameters
    ----------
    period_aggregations:
        Customer transaction aggregations by period. Must include customer_id,
        period_start, total_orders, total_spend.
    cohort_assignments:
        Mapping of customer_id to cohort_id. Get this from assign_cohorts().
    analysis_start_date:
        Start of analysis period (inclusive)
    analysis_end_date:
        End of analysis period (inclusive)

    Returns
    -------
    Lens5Metrics:
        Overall customer base health assessment including C3 data, repeat behavior,
        and health score

    Raises
    ------
    ValueError:
        If inputs are empty, date range is invalid, or required data is missing

    Notes
    -----
    Customer Cohort Chart (C3):
        The cohort_revenue_contributions data can be used to construct a C3
        stacked area chart showing revenue by cohort over time. This is the
        gold standard visualization for customer base health.

    Health Score Calculation:
        The health_score (0-100) is calculated from:
        - Overall retention rate (30% weight)
        - Cohort quality trend (30% weight): improving=80, stable=50, declining=20
        - Revenue predictability (20% weight)
        - Acquisition independence (20% weight): 100 - acquisition_dependence

    Cohort Quality Trend:
        Determined by comparing recent cohorts to historical cohorts on:
        - Repeat purchase rate
        - Average revenue per customer
    """
    # Validate inputs
    if not period_aggregations:
        raise ValueError("period_aggregations cannot be empty")
    if not cohort_assignments:
        raise ValueError("cohort_assignments cannot be empty")
    if analysis_start_date >= analysis_end_date:
        raise ValueError(
            f"analysis_start_date ({analysis_start_date}) must be before "
            f"analysis_end_date ({analysis_end_date})"
        )

    # Filter periods to analysis window
    filtered_periods = [
        p
        for p in period_aggregations
        if analysis_start_date <= p.period_start < analysis_end_date
    ]

    if not filtered_periods:
        raise ValueError(
            f"No periods found in analysis window "
            f"[{analysis_start_date}, {analysis_end_date})"
        )

    # Calculate C3 data (revenue by cohort-period)
    cohort_revenue_contributions = _calculate_c3_data(
        filtered_periods, cohort_assignments
    )

    # Calculate repeat behavior by cohort
    cohort_repeat_behavior = _calculate_repeat_behavior(
        period_aggregations, cohort_assignments
    )

    # Calculate health score (Phase 3 uses placeholders, Phase 4 will implement)
    # Placeholder: use simple defaults
    health_score_obj = CustomerBaseHealthScore(
        total_customers=len(cohort_assignments),
        total_active_customers=len(set(p.customer_id for p in filtered_periods)),
        overall_retention_rate=Decimal("70.00"),  # TODO: Phase 4
        cohort_quality_trend="stable",  # TODO: Phase 4
        revenue_predictability_pct=Decimal("60.00"),  # TODO: Phase 4
        acquisition_dependence_pct=Decimal("30.00"),  # TODO: Phase 4
        health_score=Decimal("65.00"),  # TODO: Phase 4
        health_grade="C",  # TODO: Phase 4
    )

    return Lens5Metrics(
        cohort_revenue_contributions=cohort_revenue_contributions,
        cohort_repeat_behavior=cohort_repeat_behavior,
        health_score=health_score_obj,
        analysis_start_date=analysis_start_date,
        analysis_end_date=analysis_end_date,
    )
