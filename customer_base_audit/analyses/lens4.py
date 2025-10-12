"""Lens 4: Comparing and Contrasting Cohort Performance.

This lens compares customer behavior across acquisition cohorts to:
- Identify best and worst performing cohorts
- Detect early warning signs of cohort quality degradation
- Understand profit driver explanations for differences
- Track cohort quality trends as company scales

Reference: "The Customer-Base Audit" by Fader, Hardie, and Ross (Chapter 6)
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from typing import Sequence
import logging

from customer_base_audit.foundation.data_mart import PeriodAggregation

# Module-level constants
PERCENTAGE_PRECISION = Decimal("0.01")  # 2 decimal places

# Phase 2 constants (cohort quality warnings, time-to-second-purchase analysis)
MIN_COHORT_SIZE = 10  # Minimum customers for reliable cohort analysis
EXTREME_CHANGE_THRESHOLD = Decimal(
    "500"
)  # 500% = 6x change threshold (warn about data quality)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CohortDecomposition:
    """Multiplicative decomposition of cohort revenue.

    Revenue = cohort_size × pct_active × aof × aov × margin

    IMPORTANT: Revenue Reconciliation
    ---------------------------------
    The decomposed `revenue` field will typically NOT equal `total_revenue` due to
    customer heterogeneity. This is expected and mathematically correct.

    - **total_revenue**: Actual revenue from transactions (use for reporting)
    - **revenue**: Decomposed revenue using cohort averages (use for trend analysis)

    Example: If cohort has 2 customers:
    - Customer A: 1 order × $10 = $10
    - Customer B: 10 orders × $1000 = $10,000
    - total_revenue = $10,010 (actual)
    - revenue = 2 × 100% active × 5.5 AOF × $505 AOV = $5,555 (decomposed)

    Discrepancy: 45% difference due to heterogeneity.
    Expect 10-30% discrepancy in real-world data.

    See tests/test_lens4.py::test_revenue_reconciliation_with_heterogeneous_customers

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
        Total revenue from cohort in this period (ACTUAL revenue - use for reporting)
    aov:
        Average Order Value (revenue per order)
    margin:
        Average profit margin percentage (0-100), defaults to 100 for revenue-only
    revenue:
        Calculated revenue: cohort_size × (pct_active/100) × aof × aov × (margin/100)
        (DECOMPOSED revenue - use for trend analysis and identifying drivers)
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
            raise ValueError(
                f"active_customers must be >= 0, got {self.active_customers}"
            )
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
    cumulative_repeat_by_period: Mapping[int, Decimal]

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
        pct_active = (
            Decimal(str(active_customers)) / Decimal(str(cohort_size)) * 100
        ).quantize(PERCENTAGE_PRECISION, rounding=ROUND_HALF_UP)
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
            margin = Decimal("0.00")  # No revenue → no margin
    else:
        margin = Decimal("100.00")  # Revenue-only analysis

    # Calculate decomposed revenue
    # revenue = cohort_size × (pct_active/100) × aof × aov × (margin/100)
    if cohort_size > 0:
        revenue = (
            Decimal(str(cohort_size)) * (pct_active / 100) * aof * aov * (margin / 100)
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


def calculate_time_to_second_purchase(
    cohort_id: str,
    cohort_size: int,
    customer_periods: dict[str, list[PeriodAggregation]],
) -> TimeToSecondPurchase:
    """Calculate time to second purchase distribution for a cohort.

    Parameters
    ----------
    cohort_id:
        Cohort identifier
    cohort_size:
        Total number of customers in cohort
    customer_periods:
        Dict mapping customer_id to their sorted periods (for this cohort only)

    Returns
    -------
    TimeToSecondPurchase:
        Time to second purchase analysis

    Warnings
    --------
    This function uses period boundaries to approximate time to second purchase,
    not actual transaction timestamps. For monthly periods, expect ±15 day
    accuracy on average. For weekly periods, expect ±3.5 day accuracy.

    Example: If a customer purchases on Jan 31 (period 1) and Feb 1 (period 2),
    this function reports 31 days when actual time was 1 day.

    For precise timing analysis, use transaction-level data with exact timestamps.

    Notes
    -----
    Approximation: We use period boundaries to estimate time to second purchase.
    Actual days = (second_period.period_start - first_period.period_start).days
    This is less precise than using exact transaction timestamps, but works with
    the data mart architecture.
    """
    # Log warning about approximation accuracy
    if customer_periods:
        # Estimate period length from first customer's first two periods
        sample_customer = next(iter(customer_periods.values()))
        if len(sample_customer) >= 2:
            period_length_days = (
                sample_customer[1].period_start - sample_customer[0].period_start
            ).days
            logger.warning(
                f"Time-to-second-purchase for cohort {cohort_id} calculated using "
                f"period boundaries. Expected accuracy: ±{period_length_days // 2} days. "
                f"Use transaction timestamps for precise timing analysis."
            )

    customers_with_repeat = 0
    days_to_second: list[int] = []

    for customer_id, periods in customer_periods.items():
        if len(periods) >= 2:
            customers_with_repeat += 1
            first_period_start = periods[0].period_start
            second_period_start = periods[1].period_start
            days = (second_period_start - first_period_start).days
            days_to_second.append(days)

    # Calculate repeat rate
    if cohort_size > 0:
        repeat_rate = (
            Decimal(str(customers_with_repeat)) / Decimal(str(cohort_size)) * 100
        ).quantize(PERCENTAGE_PRECISION, rounding=ROUND_HALF_UP)
    else:
        repeat_rate = Decimal("0.00")

    # Calculate median and mean
    if days_to_second:
        days_sorted = sorted(days_to_second)
        n = len(days_sorted)
        if n % 2 == 0:
            median_days = Decimal(
                str((days_sorted[n // 2 - 1] + days_sorted[n // 2]) / 2)
            ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        else:
            median_days = Decimal(str(days_sorted[n // 2])).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            )

        mean_days = (Decimal(str(sum(days_to_second))) / Decimal(str(n))).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )
    else:
        median_days = Decimal("0.00")
        mean_days = Decimal("0.00")

    # Calculate cumulative repeat by period
    cumulative_repeat_by_period: dict[int, Decimal] = {}
    max_periods = (
        max(len(periods) for periods in customer_periods.values())
        if customer_periods
        else 0
    )

    for period_num in range(max_periods):
        customers_repeated_by_period = sum(
            1 for periods in customer_periods.values() if len(periods) > period_num + 1
        )
        if cohort_size > 0:
            cumulative_pct = (
                Decimal(str(customers_repeated_by_period))
                / Decimal(str(cohort_size))
                * 100
            ).quantize(PERCENTAGE_PRECISION, rounding=ROUND_HALF_UP)
        else:
            cumulative_pct = Decimal("0.00")
        cumulative_repeat_by_period[period_num] = cumulative_pct

    return TimeToSecondPurchase(
        cohort_id=cohort_id,
        customers_with_repeat=customers_with_repeat,
        repeat_rate=repeat_rate,
        median_days=median_days,
        mean_days=mean_days,
        cumulative_repeat_by_period=cumulative_repeat_by_period,
    )


def compare_cohort_pair(
    cohort_a_decomp: CohortDecomposition,
    cohort_b_decomp: CohortDecomposition,
) -> CohortComparison:
    """Compare two cohorts at equivalent lifecycle stage.

    Parameters
    ----------
    cohort_a_decomp:
        Decomposition for cohort A at period N
    cohort_b_decomp:
        Decomposition for cohort B at period N (same period_number)

    Returns
    -------
    CohortComparison:
        Pairwise comparison metrics

    Raises
    ------
    ValueError:
        If cohorts are at different period numbers
    """
    if cohort_a_decomp.period_number != cohort_b_decomp.period_number:
        raise ValueError(
            f"Cannot compare cohorts at different periods: "
            f"{cohort_a_decomp.period_number} != {cohort_b_decomp.period_number}"
        )

    period_number = cohort_a_decomp.period_number

    # Calculate deltas
    pct_active_delta = cohort_b_decomp.pct_active - cohort_a_decomp.pct_active
    aof_delta = cohort_b_decomp.aof - cohort_a_decomp.aof
    aov_delta = cohort_b_decomp.aov - cohort_a_decomp.aov

    # Revenue per customer
    revenue_per_customer_a = (
        cohort_a_decomp.total_revenue / Decimal(str(cohort_a_decomp.cohort_size))
        if cohort_a_decomp.cohort_size > 0
        else Decimal("0.00")
    )
    revenue_per_customer_b = (
        cohort_b_decomp.total_revenue / Decimal(str(cohort_b_decomp.cohort_size))
        if cohort_b_decomp.cohort_size > 0
        else Decimal("0.00")
    )
    revenue_delta = revenue_per_customer_b - revenue_per_customer_a

    # Calculate percentage changes
    def calc_pct_change(old: Decimal, new: Decimal) -> Decimal:
        """Calculate percentage change with edge case handling.

        Edge cases:
        - old > 0: Normal percentage change formula
        - old == 0, new > 0: Returns 100% (cannot calculate true % change from 0)
          Note: This means 0→0.01 and 0→1000 both show 100% change.
          Consider using absolute deltas for trend analysis in these cases.
        - old == 0, new == 0: Returns 0% (no change)
        """
        if old > Decimal("0"):
            return ((new - old) / old * 100).quantize(
                PERCENTAGE_PRECISION, rounding=ROUND_HALF_UP
            )
        elif new > Decimal("0"):
            return Decimal("100.00")  # Went from 0 to positive
        else:
            return Decimal("0.00")  # Both are 0

    pct_active_change_pct = calc_pct_change(
        cohort_a_decomp.pct_active, cohort_b_decomp.pct_active
    )
    aof_change_pct = calc_pct_change(cohort_a_decomp.aof, cohort_b_decomp.aof)
    aov_change_pct = calc_pct_change(cohort_a_decomp.aov, cohort_b_decomp.aov)
    revenue_change_pct = calc_pct_change(revenue_per_customer_a, revenue_per_customer_b)

    return CohortComparison(
        cohort_a_id=cohort_a_decomp.cohort_id,
        cohort_b_id=cohort_b_decomp.cohort_id,
        period_number=period_number,
        pct_active_delta=pct_active_delta,
        aof_delta=aof_delta,
        aov_delta=aov_delta,
        revenue_delta=revenue_delta,
        pct_active_change_pct=pct_active_change_pct,
        aof_change_pct=aof_change_pct,
        aov_change_pct=aov_change_pct,
        revenue_change_pct=revenue_change_pct,
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
        time to second purchase, and cohort-to-cohort comparisons.

        Note: cohort_comparisons will be empty for time-aligned mode, as
        comparing cohorts at different lifecycle stages is not meaningful.
        Use left-aligned mode for cohort quality comparisons.

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

    # ASSUMPTION: The first period in period_aggregations for each customer is their
    # acquisition period (period 0 in left-aligned mode). If period data is incomplete
    # (e.g., customer acquired Jan 2024 but first period is Mar 2024), the left-alignment
    # will be incorrect. Validation of this assumption is deferred to Phase 2.

    # Determine cohort sizes
    cohort_sizes: dict[str, int] = {}
    for customer_id, cohort_id in cohort_assignments.items():
        cohort_sizes[cohort_id] = cohort_sizes.get(cohort_id, 0) + 1

    # Warn about small cohorts (unreliable statistics)
    for cohort_id, cohort_size in cohort_sizes.items():
        if cohort_size < MIN_COHORT_SIZE:
            logger.warning(
                f"Cohort {cohort_id} has only {cohort_size} customers "
                f"(minimum {MIN_COHORT_SIZE} recommended for reliable statistics). "
                f"Metrics may be unstable: median/mean calculations, percentage changes, "
                f"and comparisons should be interpreted with caution."
            )

    # Calculate decompositions per cohort-period
    cohort_decompositions: list[CohortDecomposition] = []
    cohort_period_data: dict[tuple[str, int], list[PeriodAggregation]] = {}

    if alignment_type == "left-aligned":
        # Left-aligned: period 0 = acquisition period, 1 = next period, etc.
        # Group periods by cohort and relative period number
        for customer_id, cohort_id in cohort_assignments.items():
            customer_periods = periods_by_customer.get(customer_id, [])
            if not customer_periods:
                continue

            # First period is period 0 (acquisition)
            for period_idx, period in enumerate(customer_periods):
                key = (cohort_id, period_idx)
                cohort_period_data.setdefault(key, []).append(period)
    else:  # time-aligned
        # Time-aligned: period 0 = first calendar period globally, 1 = second, etc.
        # Determine all unique period starts
        all_period_starts = sorted(set(p.period_start for p in period_aggregations))
        period_start_to_idx = {ps: idx for idx, ps in enumerate(all_period_starts)}

        # Group periods by cohort and absolute period index
        for customer_id, cohort_id in cohort_assignments.items():
            customer_periods = periods_by_customer.get(customer_id, [])
            for period in customer_periods:
                period_idx = period_start_to_idx[period.period_start]
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

    # Pre-group customers by cohort for efficient lookup
    customers_by_cohort: dict[str, list[str]] = {}
    for customer_id, cohort_id in cohort_assignments.items():
        customers_by_cohort.setdefault(cohort_id, []).append(customer_id)

    # Calculate time to second purchase for each cohort
    time_to_second_purchase_list: list[TimeToSecondPurchase] = []
    for cohort_id, cohort_size in sorted(cohort_sizes.items()):
        # Get customer periods for this cohort (optimized lookup)
        customer_ids = customers_by_cohort.get(cohort_id, [])
        cohort_customer_periods: dict[str, list[PeriodAggregation]] = {
            cid: periods_by_customer.get(cid, [])
            for cid in customer_ids
            if periods_by_customer.get(cid)
        }

        ttsp = calculate_time_to_second_purchase(
            cohort_id=cohort_id,
            cohort_size=cohort_size,
            customer_periods=cohort_customer_periods,
        )
        time_to_second_purchase_list.append(ttsp)

    # Calculate pairwise comparisons between cohorts at same periods
    cohort_comparisons_list: list[CohortComparison] = []
    if alignment_type == "left-aligned":
        # Group decompositions by period_number
        decomps_by_period: dict[int, list[CohortDecomposition]] = {}
        for decomp in cohort_decompositions:
            decomps_by_period.setdefault(decomp.period_number, []).append(decomp)

        # For each period, compare all cohort pairs
        for period_num, decomps in sorted(decomps_by_period.items()):
            if len(decomps) < 2:
                continue
            # Compare consecutive cohorts.
            # NOTE: Comparisons assume cohort_ids are lexicographically sortable in
            # chronological order (e.g., "2024-01", "2024-02", "2024-03").
            # If your cohort_ids don't sort this way (e.g., "2024-Q1", "2024-Q2" with
            # "Q10" < "Q2"), comparisons may be incorrect. Ensure cohort_id naming
            # convention supports lexicographic sorting or pre-sort by acquisition date.
            for i in range(len(decomps) - 1):
                comparison = compare_cohort_pair(decomps[i], decomps[i + 1])
                cohort_comparisons_list.append(comparison)
    else:  # time-aligned
        logger.info(
            "Cohort comparisons not supported for time-aligned mode. "
            "Time-aligned mode shows cohorts at different lifecycle stages within "
            "the same calendar period, making direct comparisons less meaningful. "
            "Use left-aligned mode to compare cohorts at equivalent lifecycle stages."
        )

    return Lens4Metrics(
        cohort_decompositions=cohort_decompositions,
        time_to_second_purchase=time_to_second_purchase_list,
        cohort_comparisons=cohort_comparisons_list,
        alignment_type=alignment_type,
    )
