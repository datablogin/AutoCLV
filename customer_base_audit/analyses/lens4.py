"""Lens 4: Comparing and Contrasting Cohort Performance.

This lens compares customer behavior across acquisition cohorts to:
- Identify best and worst performing cohorts
- Detect early warning signs of cohort quality degradation
- Understand profit driver explanations for differences
- Track cohort quality trends as company scales

Reference: "The Customer-Base Audit" by Fader, Hardie, and Ross (Chapter 6)
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from typing import Sequence
import logging

from customer_base_audit.foundation.data_mart import PeriodAggregation

# Module-level constants
PERCENTAGE_PRECISION = Decimal("0.01")  # 2 decimal places
MIN_COHORT_SIZE = 10  # Minimum customers for reliable cohort analysis
EXTREME_CHANGE_THRESHOLD = Decimal("500")  # 500% = 6x change threshold (warn about data quality)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CohortDecomposition:
    """Multiplicative decomposition of cohort revenue.

    Revenue = cohort_size × pct_active × aof × aov × margin

    Attributes
    ----------
    cohort_id:
        Cohort identifier (e.g., "2023-Q1")
    period_number:
        Period number relative to cohort acquisition (left-aligned: 0, 1, 2...)
        or absolute period index (time-aligned: 0=Jan 2023, 1=Feb 2023...)
    cohort_size:
        Total number of customers acquired in the cohort
    active_customers:
        Number of active customers in this period
    pct_active:
        Percentage of cohort that is active (0-100)
    total_orders:
        Total orders from active customers in this period
    aof:
        Average Order Frequency (orders per active customer)
    total_revenue:
        Total revenue from cohort in this period
    aov:
        Average Order Value (revenue per order)
    margin:
        Average profit margin percentage (0-100), defaults to 100 for revenue-only
    revenue:
        Calculated revenue: cohort_size × (pct_active/100) × aof × aov × (margin/100)
        Should approximately equal total_revenue (may differ due to rounding)
    """
    cohort_id: str
    period_number: int
    cohort_size: int
    active_customers: int
    pct_active: Decimal
    total_orders: int
    aof: Decimal
    total_revenue: Decimal
    aov: Decimal
    margin: Decimal
    revenue: Decimal

    def __post_init__(self) -> None:
        """Validate decomposition consistency."""
        if self.period_number < 0:
            raise ValueError(f"period_number must be >= 0, got {self.period_number}")
        if self.cohort_size < 0:
            raise ValueError(f"cohort_size must be >= 0, got {self.cohort_size}")
        if self.active_customers < 0:
            raise ValueError(f"active_customers must be >= 0, got {self.active_customers}")
        if self.active_customers > self.cohort_size:
            raise ValueError(
                f"active_customers ({self.active_customers}) cannot exceed "
                f"cohort_size ({self.cohort_size})"
            )
        if not (Decimal("0") <= self.pct_active <= Decimal("100")):
            raise ValueError(f"pct_active must be in [0, 100], got {self.pct_active}")
        if not (Decimal("0") <= self.margin <= Decimal("100")):
            raise ValueError(f"margin must be in [0, 100], got {self.margin}")
        if self.total_orders < 0:
            raise ValueError(f"total_orders must be >= 0, got {self.total_orders}")
        if self.aof < 0:
            raise ValueError(f"aof must be >= 0, got {self.aof}")
        if self.aov < 0:
            raise ValueError(f"aov must be >= 0, got {self.aov}")
        if self.total_revenue < 0:
            raise ValueError(f"total_revenue must be >= 0, got {self.total_revenue}")
        if self.revenue < 0:
            raise ValueError(f"revenue must be >= 0, got {self.revenue}")


@dataclass(frozen=True)
class TimeToSecondPurchase:
    """Time to second purchase analysis for a cohort.

    Attributes
    ----------
    cohort_id:
        Cohort identifier
    customers_with_repeat:
        Number of customers who made a second purchase
    repeat_rate:
        Percentage of cohort who made a second purchase (0-100)
    median_days:
        Median days to second purchase (for those who repeated)
    mean_days:
        Mean days to second purchase (for those who repeated)
    cumulative_repeat_by_period:
        Cumulative % of cohort making second purchase by each period
        Dict mapping period_number -> cumulative_pct
    """
    cohort_id: str
    customers_with_repeat: int
    repeat_rate: Decimal
    median_days: Decimal
    mean_days: Decimal
    cumulative_repeat_by_period: dict[int, Decimal]

    def __post_init__(self) -> None:
        """Validate time to second purchase metrics."""
        if self.customers_with_repeat < 0:
            raise ValueError(
                f"customers_with_repeat must be >= 0, got {self.customers_with_repeat}"
            )
        if not (Decimal("0") <= self.repeat_rate <= Decimal("100")):
            raise ValueError(f"repeat_rate must be in [0, 100], got {self.repeat_rate}")
        if self.median_days < 0:
            raise ValueError(f"median_days must be >= 0, got {self.median_days}")
        if self.mean_days < 0:
            raise ValueError(f"mean_days must be >= 0, got {self.mean_days}")


@dataclass(frozen=True)
class CohortComparison:
    """Comparison metrics between two cohorts at equivalent lifecycle stages.

    Attributes
    ----------
    cohort_a_id:
        Identifier for first cohort
    cohort_b_id:
        Identifier for second cohort
    period_number:
        Period number for comparison (left-aligned)
    pct_active_delta:
        Change in % active (cohort_b - cohort_a)
    aof_delta:
        Change in AOF (cohort_b - cohort_a)
    aov_delta:
        Change in AOV (cohort_b - cohort_a)
    revenue_delta:
        Change in revenue per customer (cohort_b - cohort_a)
    pct_active_change_pct:
        Percentage change in % active ((b-a)/a * 100)
    aof_change_pct:
        Percentage change in AOF
    aov_change_pct:
        Percentage change in AOV
    revenue_change_pct:
        Percentage change in revenue per customer
    """
    cohort_a_id: str
    cohort_b_id: str
    period_number: int
    pct_active_delta: Decimal
    aof_delta: Decimal
    aov_delta: Decimal
    revenue_delta: Decimal
    pct_active_change_pct: Decimal
    aof_change_pct: Decimal
    aov_change_pct: Decimal
    revenue_change_pct: Decimal


@dataclass(frozen=True)
class Lens4Metrics:
    """Lens 4: Multi-cohort comparison analysis results.

    Attributes
    ----------
    cohort_decompositions:
        Revenue decomposition for each cohort-period combination.
        Sorted by cohort_id, then period_number.
    time_to_second_purchase:
        Time to second purchase analysis for each cohort (Phase 2)
    cohort_comparisons:
        Pairwise comparisons between cohorts at equivalent periods (Phase 2)
    alignment_type:
        "left-aligned" (by cohort age) or "time-aligned" (by calendar period)
    """
    cohort_decompositions: Sequence[CohortDecomposition]
    time_to_second_purchase: Sequence[TimeToSecondPurchase]
    cohort_comparisons: Sequence[CohortComparison]
    alignment_type: str

    def __post_init__(self) -> None:
        """Validate Lens 4 metrics."""
        if self.alignment_type not in ("left-aligned", "time-aligned"):
            raise ValueError(
                f"alignment_type must be 'left-aligned' or 'time-aligned', "
                f"got '{self.alignment_type}'"
            )

        # Validate cohort_decompositions are sorted by cohort_id, period_number
        if self.cohort_decompositions:
            for i in range(len(self.cohort_decompositions) - 1):
                curr = self.cohort_decompositions[i]
                next_item = self.cohort_decompositions[i + 1]

                if curr.cohort_id > next_item.cohort_id:
                    raise ValueError(
                        "cohort_decompositions must be sorted by cohort_id"
                    )
                elif curr.cohort_id == next_item.cohort_id:
                    if curr.period_number >= next_item.period_number:
                        raise ValueError(
                            "cohort_decompositions must be sorted by period_number "
                            "within each cohort"
                        )


def calculate_cohort_decomposition(
    cohort_id: str,
    period_number: int,
    cohort_size: int,
    period_data: Sequence[PeriodAggregation],
    include_margin: bool = False,
) -> CohortDecomposition:
    """Calculate multiplicative revenue decomposition for one cohort-period.

    Parameters
    ----------
    cohort_id:
        Cohort identifier
    period_number:
        Period number (0 = acquisition period for left-aligned)
    cohort_size:
        Total number of customers in the cohort
    period_data:
        PeriodAggregation records for customers in this cohort during this period
    include_margin:
        If True, calculate margin from period_data. If False, use 100% (revenue-only)

    Returns
    -------
    CohortDecomposition:
        Multiplicative decomposition of revenue

    Notes
    -----
    Handles edge cases:
    - Zero active customers → all metrics are 0
    - Zero orders → aof and aov are 0
    - Missing margin data → defaults to 100% (revenue == revenue)
    """
    # Count active customers
    active_customers = len(period_data)

    # Calculate % active
    if cohort_size > 0:
        pct_active = (Decimal(str(active_customers)) / Decimal(str(cohort_size)) * 100).quantize(
            PERCENTAGE_PRECISION, rounding=ROUND_HALF_UP
        )
    else:
        pct_active = Decimal("0.00")

    # Aggregate orders and revenue
    total_orders = sum(p.total_orders for p in period_data)
    total_revenue_raw = sum(p.total_spend for p in period_data)
    total_revenue = Decimal(str(total_revenue_raw)).quantize(
        Decimal("0.01"), rounding=ROUND_HALF_UP
    )

    # Calculate AOF (Average Order Frequency)
    if active_customers > 0:
        aof = (Decimal(str(total_orders)) / Decimal(str(active_customers))).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )
    else:
        aof = Decimal("0.00")

    # Calculate AOV (Average Order Value)
    if total_orders > 0:
        aov = (total_revenue / Decimal(str(total_orders))).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )
    else:
        aov = Decimal("0.00")

    # Calculate margin
    if include_margin:
        total_margin_raw = sum(p.total_margin for p in period_data)
        total_margin = Decimal(str(total_margin_raw)).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )
        if total_revenue > 0:
            margin = (total_margin / total_revenue * 100).quantize(
                PERCENTAGE_PRECISION, rounding=ROUND_HALF_UP
            )
        else:
            margin = Decimal("100.00")  # Default to 100% if no revenue
    else:
        margin = Decimal("100.00")  # Revenue-only analysis

    # Calculate decomposed revenue
    # revenue = cohort_size × (pct_active/100) × aof × aov × (margin/100)
    if cohort_size > 0:
        revenue = (
            Decimal(str(cohort_size))
            * (pct_active / 100)
            * aof
            * aov
            * (margin / 100)
        ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    else:
        revenue = Decimal("0.00")

    return CohortDecomposition(
        cohort_id=cohort_id,
        period_number=period_number,
        cohort_size=cohort_size,
        active_customers=active_customers,
        pct_active=pct_active,
        total_orders=total_orders,
        aof=aof,
        total_revenue=total_revenue,
        aov=aov,
        margin=margin,
        revenue=revenue,
    )


def compare_cohorts(
    period_aggregations: Sequence[PeriodAggregation],
    cohort_assignments: dict[str, str],  # customer_id -> cohort_id
    alignment_type: str = "left-aligned",
    include_margin: bool = False,
) -> Lens4Metrics:
    """Perform multi-cohort comparison analysis (Lens 4).

    Compares customer behavior across acquisition cohorts to identify trends in
    cohort quality and performance drivers. Can align cohorts by lifecycle stage
    (left-aligned) or calendar time (time-aligned).

    Parameters
    ----------
    period_aggregations:
        Customer transaction aggregations by period. Must include customer_id,
        period_start, total_orders, total_spend, and optionally total_margin.
    cohort_assignments:
        Mapping of customer_id to cohort_id. Get this from assign_cohorts().
    alignment_type:
        How to align cohorts for comparison:
        - "left-aligned": Compare cohorts at equivalent lifecycle stages
          (Period 0 = acquisition period for each cohort)
        - "time-aligned": Compare cohorts within same calendar periods
        Default: "left-aligned"
    include_margin:
        If True, calculate margin from period_aggregations. If False, use 100%
        margin (revenue-only analysis). Default: False

    Returns
    -------
    Lens4Metrics:
        Multi-cohort comparison results including revenue decomposition,
        time to second purchase, and cohort-to-cohort comparisons

    Raises
    ------
    ValueError:
        If alignment_type is invalid, inputs are empty, or required data is missing

    Examples
    --------
    >>> from datetime import datetime, timezone
    >>> from customer_base_audit.foundation.data_mart import PeriodAggregation
    >>> from customer_base_audit.foundation.cohorts import CohortDefinition, assign_cohorts
    >>> from customer_base_audit.foundation.customer_contract import CustomerIdentifier
    >>>
    >>> # Setup cohorts
    >>> cohorts = [CohortDefinition("2023-Q1", datetime(2023, 1, 1, tzinfo=timezone.utc), datetime(2023, 4, 1, tzinfo=timezone.utc))]
    >>> customers = [CustomerIdentifier("C1", datetime(2023, 1, 15, tzinfo=timezone.utc), "system")]
    >>> cohort_assignments = assign_cohorts(customers, cohorts)
    >>>
    >>> # Period aggregations
    >>> periods = [
    ...     PeriodAggregation(
    ...         customer_id="C1",
    ...         period_start=datetime(2023, 1, 1, tzinfo=timezone.utc),
    ...         period_end=datetime(2023, 2, 1, tzinfo=timezone.utc),
    ...         total_orders=3,
    ...         total_spend=150.00,
    ...         total_margin=50.00,
    ...         total_quantity=5,
    ...     ),
    ... ]
    >>>
    >>> # Perform left-aligned comparison
    >>> lens4 = compare_cohorts(periods, cohort_assignments, alignment_type="left-aligned")
    >>> lens4.alignment_type
    'left-aligned'
    >>> len(lens4.cohort_decompositions) > 0
    True

    Notes
    -----
    Left-Aligned Analysis:
        Cohorts are compared at equivalent lifecycle stages. Period 0 represents
        the acquisition period for each cohort, Period 1 is one period after
        acquisition, etc. Best for identifying changes in cohort quality.

    Time-Aligned Analysis:
        Cohorts are compared within the same calendar periods. Shows all cohort
        contributions in each period. Best for understanding current business
        performance and revenue composition.

    Revenue Decomposition:
        Revenue = Cohort Size × % Active × AOF × AOV × Margin
        This multiplicative decomposition helps identify which specific drivers
        explain differences between cohorts.
    """
    # Validate alignment_type first
    if alignment_type not in ("left-aligned", "time-aligned"):
        raise ValueError(
            f"alignment_type must be 'left-aligned' or 'time-aligned', got '{alignment_type}'"
        )

    # Phase 1: Only left-aligned is implemented
    if alignment_type == "time-aligned":
        raise NotImplementedError("Time-aligned comparison not yet implemented (Phase 2)")

    # Handle empty inputs gracefully
    if not period_aggregations or not cohort_assignments:
        return Lens4Metrics(
            alignment_type=alignment_type,
            cohort_decompositions=[],
            time_to_second_purchase=[],
            cohort_comparisons=[],
        )

    # Group period aggregations by customer
    periods_by_customer: dict[str, list[PeriodAggregation]] = {}
    for period in period_aggregations:
        periods_by_customer.setdefault(period.customer_id, []).append(period)

    # Sort each customer's periods by period_start
    for periods in periods_by_customer.values():
        periods.sort(key=lambda p: p.period_start)

    # Determine cohort sizes
    cohort_sizes: dict[str, int] = {}
    for customer_id, cohort_id in cohort_assignments.items():
        cohort_sizes[cohort_id] = cohort_sizes.get(cohort_id, 0) + 1

    # Calculate decompositions per cohort-period (left-aligned)
    # Left-aligned: period 0 = acquisition period, 1 = next period, etc.
    cohort_decompositions: list[CohortDecomposition] = []
    cohort_period_data: dict[tuple[str, int], list[PeriodAggregation]] = {}

    # Group periods by cohort and relative period number
    for customer_id, cohort_id in cohort_assignments.items():
        customer_periods = periods_by_customer.get(customer_id, [])
        if not customer_periods:
            continue

        # First period is period 0 (acquisition)
        for period_idx, period in enumerate(customer_periods):
            key = (cohort_id, period_idx)
            cohort_period_data.setdefault(key, []).append(period)

    # Calculate decomposition for each cohort-period
    for (cohort_id, period_num), periods in sorted(cohort_period_data.items()):
        cohort_size = cohort_sizes[cohort_id]
        decomp = calculate_cohort_decomposition(
            cohort_id=cohort_id,
            period_number=period_num,
            cohort_size=cohort_size,
            period_data=periods,
            include_margin=include_margin,
        )
        cohort_decompositions.append(decomp)

    # Placeholder for time-to-second-purchase and cohort comparisons (Phase 2)
    time_to_second_purchase: list[TimeToSecondPurchase] = []
    cohort_comparisons: list[CohortComparison] = []

    return Lens4Metrics(
        cohort_decompositions=cohort_decompositions,
        time_to_second_purchase=time_to_second_purchase,
        cohort_comparisons=cohort_comparisons,
        alignment_type=alignment_type,
    )
