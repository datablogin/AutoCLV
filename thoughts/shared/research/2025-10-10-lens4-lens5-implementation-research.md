# Lens 4 & 5 Implementation Research

**Date**: 2025-10-10
**Track**: B
**Issues**: #34 (Lens 4), #35 (Lens 5)
**Author**: Research synthesis from book, web sources, and codebase analysis

---

## Executive Summary

This document synthesizes research from three sources to propose implementation plans for Lens 4 (Multi-Cohort Comparison) and Lens 5 (Overall Customer Base Health):

1. **Source Material**: "The Customer-Base Audit" book (document.txt, Chapters 6-7)
2. **Web Research**: Academic papers, commercial implementations, industry examples
3. **Codebase Analysis**: Architectural patterns from existing Lens 1-3 implementations

**Key Findings**:
- Lens 4 requires multiplicative revenue decomposition and cohort comparison (left-aligned vs time-aligned)
- Lens 5 requires Customer Cohort Chart (C3) construction and overall health scoring
- Both lenses should follow established patterns: frozen dataclasses, Decimal precision, comprehensive validation
- Estimated effort: 2-3 days per lens (per Issues #34 and #35)

---

## Table of Contents

1. [Lens 4: Multi-Cohort Comparison](#lens-4-multi-cohort-comparison)
2. [Lens 5: Overall Customer Base Health](#lens-5-overall-customer-base-health)
3. [Implementation Priorities](#implementation-priorities)
4. [Architectural Patterns to Follow](#architectural-patterns-to-follow)
5. [Testing Strategy](#testing-strategy)
6. [Integration Points](#integration-points)

---

## Lens 4: Multi-Cohort Comparison

### Purpose (from Chapter 6)

> "Lens 4 is so important. Like a traditional financial audit, boring is usually good. And where there is smoke, there is often fire."

Lens 4 compares behaviors across acquisition cohorts to:
- Identify best and worst performing cohorts
- Detect early warning signs of cohort quality degradation
- Understand profit driver explanations for cohort differences
- Track cohort quality trends as company scales

### Key Analyses Required

#### 1. Multiplicative Revenue Decomposition

**Formula**: `Revenue = Cohort Size × % Active × AOF × AOV × Margin`

Where:
- **Cohort Size**: Number of customers acquired in the cohort
- **% Active**: Percentage of cohort still making purchases
- **AOF** (Average Order Frequency): Orders per active customer
- **AOV** (Average Order Value): Revenue per order
- **Margin**: Average profit margin (optional, can default to 1.0 for revenue-only)

**Purpose**: Decompose revenue differences between cohorts into specific drivers to identify root causes.

#### 2. Left-Aligned Comparison (by Cohort Age)

Compare cohorts at equivalent lifecycle stages:
- Period 0 = first purchase period for each cohort
- Period 1 = one period after first purchase
- Etc.

**Use Case**: "How did Q1 2023 cohort perform in their first 90 days vs Q1 2022 cohort in their first 90 days?"

#### 3. Time-Aligned Comparison (by Calendar Time)

Compare cohorts within the same calendar periods:
- All cohorts' contributions in Q4 2023
- All cohorts' contributions in Q1 2024
- Etc.

**Use Case**: "What revenue did each cohort contribute in Q4 2023?"

#### 4. Time to Second Purchase Analysis

Track distribution of days/periods until second purchase:
- Cumulative: % of cohort making second purchase by period N
- Incremental: % of cohort making second purchase in period N

**Early Warning Indicator**: Increasing time to second purchase in newer cohorts signals quality degradation.

#### 5. Repeat Purchase Rate by Cohort

Track what percentage of each cohort makes 2+ purchases:
- Formula: `(Customers with 2+ purchases) / Cohort Size × 100%`
- Industry benchmark: 20-40% for e-commerce

### Proposed Implementation: Lens 4

#### Module: `customer_base_audit/analyses/lens4.py`

#### Dataclasses

```python
from dataclasses import dataclass
from decimal import Decimal
from typing import Sequence

@dataclass(frozen=True)
class CohortDecomposition:
    """Multiplicative decomposition of cohort revenue.

    Revenue = cohort_size × pct_active × aof × aov × margin

    Attributes
    ----------
    cohort_id:
        Cohort identifier (e.g., "2023-Q1")
    period_number:
        Period number relative to cohort acquisition (left-aligned)
        or absolute period (time-aligned)
    cohort_size:
        Total number of customers in the cohort
    active_customers:
        Number of active customers in this period
    pct_active:
        Percentage of cohort that is active (0-100)
    total_orders:
        Total orders from active customers
    aof:
        Average Order Frequency (orders per active customer)
    total_revenue:
        Total revenue from cohort in this period
    aov:
        Average Order Value (revenue per order)
    margin:
        Average profit margin percentage (0-100), defaults to 100 for revenue-only
    revenue:
        Calculated revenue (should equal total_revenue)
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
        if self.cohort_size < 0:
            raise ValueError(f"cohort_size must be >= 0, got {self.cohort_size}")
        if self.active_customers < 0:
            raise ValueError(f"active_customers must be >= 0, got {self.active_customers}")
        if self.active_customers > self.cohort_size:
            raise ValueError(
                f"active_customers ({self.active_customers}) cannot exceed "
                f"cohort_size ({self.cohort_size})"
            )
        if not (0 <= self.pct_active <= 100):
            raise ValueError(f"pct_active must be in [0, 100], got {self.pct_active}")
        if not (0 <= self.margin <= 100):
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
        if not (0 <= self.repeat_rate <= 100):
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
        Percentage change in % active
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
        Revenue decomposition for each cohort-period combination
    time_to_second_purchase:
        Time to second purchase analysis for each cohort
    cohort_comparisons:
        Pairwise comparisons between cohorts at equivalent periods
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
```

#### Main Function

```python
from datetime import datetime
from customer_base_audit.foundation.data_mart import PeriodAggregation
from customer_base_audit.foundation.cohorts import CohortAssignment


def compare_cohorts(
    period_aggregations: Sequence[PeriodAggregation],
    cohort_assignments: Sequence[CohortAssignment],
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
        period_start, total_orders, total_spend, and optionally margin.
    cohort_assignments:
        Cohort membership for each customer, including cohort_id and
        acquisition_date.
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
    >>> from datetime import datetime
    >>> from customer_base_audit.foundation.data_mart import PeriodAggregation
    >>> from customer_base_audit.foundation.cohorts import CohortAssignment
    >>>
    >>> # Period aggregations for 2 customers over 3 periods
    >>> periods = [
    ...     PeriodAggregation(
    ...         customer_id="C1",
    ...         period_start=datetime(2023, 1, 1),
    ...         total_orders=3,
    ...         total_spend=Decimal("150.00"),
    ...     ),
    ...     PeriodAggregation(
    ...         customer_id="C1",
    ...         period_start=datetime(2023, 2, 1),
    ...         total_orders=2,
    ...         total_spend=Decimal("100.00"),
    ...     ),
    ... ]
    >>>
    >>> # Cohort assignments
    >>> cohorts = [
    ...     CohortAssignment(
    ...         customer_id="C1",
    ...         cohort_id="2023-Q1",
    ...         acquisition_date=datetime(2023, 1, 15),
    ...     ),
    ...     CohortAssignment(
    ...         customer_id="C2",
    ...         cohort_id="2023-Q2",
    ...         acquisition_date=datetime(2023, 4, 10),
    ...     ),
    ... ]
    >>>
    >>> # Perform left-aligned comparison
    >>> lens4 = compare_cohorts(periods, cohorts, alignment_type="left-aligned")
    >>> len(lens4.cohort_decompositions) > 0
    True
    >>> lens4.alignment_type
    'left-aligned'

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
    # Implementation details in customer_base_audit/analyses/lens4.py
    pass
```

#### Helper Functions

```python
def calculate_cohort_decomposition(
    cohort_id: str,
    period_number: int,
    cohort_size: int,
    period_data: Sequence[PeriodAggregation],
    include_margin: bool = False,
) -> CohortDecomposition:
    """Calculate multiplicative revenue decomposition for one cohort-period."""
    pass


def calculate_time_to_second_purchase(
    cohort_id: str,
    cohort_customers: Sequence[str],
    all_transactions: Sequence[PeriodAggregation],
) -> TimeToSecondPurchase:
    """Calculate time to second purchase distribution for a cohort."""
    pass


def compare_cohort_pair(
    cohort_a_decomp: CohortDecomposition,
    cohort_b_decomp: CohortDecomposition,
) -> CohortComparison:
    """Compare two cohorts at equivalent lifecycle stage."""
    pass
```

---

## Lens 5: Overall Customer Base Health

### Purpose (from Chapter 7)

Lens 5 provides an integrative view synthesizing all previous lenses to answer:
- **How healthy is our customer base?**
- **How dependent are we on new customer acquisition?**
- **What percentage of revenue comes from each cohort?**
- **How has customer quality changed over time?**

### Key Analyses Required

#### 1. Customer Cohort Chart (C3)

**Definition**: Stacked visualization showing revenue/orders over time, broken down by acquisition cohort.

**Structure**:
- X-axis: Time periods (months, quarters, years)
- Y-axis: Revenue or order volume
- Layers: Each cohort as a colored band (oldest at bottom)

**Insights**:
- **Acquisition Dependence**: Heavy reliance on newest cohort is a red flag
- **Retention Quality**: Shrinking bands indicate poor retention
- **Revenue Maturation**: How cohort revenue evolves over time

#### 2. Annual/Quarterly Performance by Cohort

Decompose total revenue/profit into contributions from each cohort:
- What % of current revenue comes from 2020 cohort? 2021? etc.
- Which cohorts remain most valuable?
- Is revenue too concentrated in recent cohorts?

#### 3. Repeat-Buying Rates by Cohort

Track what percentage of each cohort makes repeat purchases:
- Overall repeat rate
- Repeat rate by cohort
- Evolution of repeat behavior over cohort lifecycle

#### 4. Overall Customer Base Health Score

Aggregate metrics assessing overall base health:
- **Retention stability**: Consistent retention across cohorts?
- **Cohort quality trend**: Improving or declining?
- **Revenue predictability**: What % of next period revenue is predictable?
- **Acquisition dependence**: What % of revenue comes from newest cohort?

### Proposed Implementation: Lens 5

#### Module: `customer_base_audit/analyses/lens5.py`

#### Dataclasses

```python
from dataclasses import dataclass
from decimal import Decimal
from datetime import datetime
from typing import Sequence


@dataclass(frozen=True)
class CohortRevenuePeriod:
    """Revenue contribution from one cohort in one period.

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
        Number of active customers from this cohort
    avg_revenue_per_customer:
        Average revenue per active customer
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
        if not (0 <= self.pct_of_period_revenue <= 100):
            raise ValueError(
                f"pct_of_period_revenue must be in [0, 100], "
                f"got {self.pct_of_period_revenue}"
            )
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
        Total customers in cohort
    one_time_buyers:
        Customers with only 1 purchase
    repeat_buyers:
        Customers with 2+ purchases
    repeat_rate:
        Percentage of cohort with 2+ purchases (0-100)
    avg_orders_per_repeat_buyer:
        Average orders for customers with 2+ purchases
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
        if not (0 <= self.repeat_rate <= 100):
            raise ValueError(f"repeat_rate must be in [0, 100], got {self.repeat_rate}")
        if self.avg_orders_per_repeat_buyer < 2:
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
        Currently active customers
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
        if not (0 <= self.overall_retention_rate <= 100):
            raise ValueError(
                f"overall_retention_rate must be in [0, 100], "
                f"got {self.overall_retention_rate}"
            )
        if self.cohort_quality_trend not in ("improving", "stable", "declining"):
            raise ValueError(
                f"cohort_quality_trend must be 'improving', 'stable', or 'declining', "
                f"got '{self.cohort_quality_trend}'"
            )
        if not (0 <= self.revenue_predictability_pct <= 100):
            raise ValueError(
                f"revenue_predictability_pct must be in [0, 100], "
                f"got {self.revenue_predictability_pct}"
            )
        if not (0 <= self.acquisition_dependence_pct <= 100):
            raise ValueError(
                f"acquisition_dependence_pct must be in [0, 100], "
                f"got {self.acquisition_dependence_pct}"
            )
        if not (0 <= self.health_score <= 100):
            raise ValueError(f"health_score must be in [0, 100], got {self.health_score}")
        if self.health_grade not in ("A", "B", "C", "D", "F"):
            raise ValueError(
                f"health_grade must be 'A', 'B', 'C', 'D', or 'F', "
                f"got '{self.health_grade}'"
            )


@dataclass(frozen=True)
class Lens5Metrics:
    """Lens 5: Overall customer base health analysis results.

    Attributes
    ----------
    cohort_revenue_contributions:
        Revenue contribution from each cohort in each period (for C3 chart)
    cohort_repeat_behavior:
        Repeat purchase behavior for each cohort
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
```

#### Main Function

```python
def assess_customer_base_health(
    period_aggregations: Sequence[PeriodAggregation],
    cohort_assignments: Sequence[CohortAssignment],
    analysis_start_date: datetime,
    analysis_end_date: datetime,
) -> Lens5Metrics:
    """Assess overall customer base health (Lens 5).

    Provides integrative view of customer base by analyzing revenue contributions
    across cohorts, repeat purchase behavior, and overall health indicators.
    Answers: How healthy is our customer base? How dependent are we on acquisition?

    Parameters
    ----------
    period_aggregations:
        Customer transaction aggregations by period. Must include customer_id,
        period_start, total_orders, total_spend.
    cohort_assignments:
        Cohort membership for each customer, including cohort_id and
        acquisition_date.
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

    Examples
    --------
    >>> from datetime import datetime
    >>> from customer_base_audit.foundation.data_mart import PeriodAggregation
    >>> from customer_base_audit.foundation.cohorts import CohortAssignment
    >>>
    >>> # Period aggregations
    >>> periods = [...]
    >>>
    >>> # Cohort assignments
    >>> cohorts = [...]
    >>>
    >>> # Assess health for 2023
    >>> lens5 = assess_customer_base_health(
    ...     periods,
    ...     cohorts,
    ...     analysis_start_date=datetime(2023, 1, 1),
    ...     analysis_end_date=datetime(2023, 12, 31),
    ... )
    >>>
    >>> # Check health score
    >>> lens5.health_score.health_grade in ("A", "B", "C", "D", "F")
    True
    >>>
    >>> # Get newest cohort's revenue contribution
    >>> newest_cohort = lens5.cohort_revenue_contributions[-1]
    >>> newest_cohort.pct_of_period_revenue < 50  # Should not be too dependent
    True

    Notes
    -----
    Customer Cohort Chart (C3):
        The cohort_revenue_contributions data can be used to construct a C3
        stacked area chart showing revenue by cohort over time. This is the
        gold standard visualization for customer base health.

    Health Score Calculation:
        The health_score (0-100) is calculated from:
        - Overall retention rate (30% weight)
        - Cohort quality trend (30% weight)
        - Revenue predictability (20% weight)
        - Acquisition independence (20% weight)

    Cohort Quality Trend:
        Determined by comparing recent cohorts to historical cohorts on:
        - Repeat purchase rate
        - Average revenue per customer
        - Retention at equivalent lifecycle stages
    """
    # Implementation details in customer_base_audit/analyses/lens5.py
    pass
```

#### Helper Functions

```python
def calculate_health_score(
    overall_retention: Decimal,
    cohort_quality_trend: str,
    revenue_predictability: Decimal,
    acquisition_dependence: Decimal,
) -> tuple[Decimal, str]:
    """Calculate overall health score and grade from component metrics.

    Returns
    -------
    tuple[Decimal, str]:
        (health_score, health_grade) where score is 0-100 and grade is A-F
    """
    pass


def determine_cohort_quality_trend(
    cohort_metrics: Sequence[CohortRepeatBehavior],
) -> str:
    """Determine if cohort quality is improving, stable, or declining.

    Returns
    -------
    str:
        "improving", "stable", or "declining"
    """
    pass


def calculate_revenue_predictability(
    cohort_revenue_contributions: Sequence[CohortRevenuePeriod],
    newest_cohort_id: str,
) -> Decimal:
    """Calculate what % of next period revenue is predictable from existing cohorts.

    Assumes newest cohort represents unpredictable acquisition-driven revenue.
    """
    pass
```

---

## Implementation Priorities

### Phase 1: Lens 4 Foundation (Days 1-2)

**Goal**: Implement core Lens 4 functionality with multiplicative decomposition

**Tasks**:
1. Create `lens4.py` module with dataclasses
2. Implement `calculate_cohort_decomposition()` helper
3. Implement left-aligned comparison in `compare_cohorts()`
4. Write comprehensive unit tests for decomposition logic
5. Test with synthetic data from Texas CLV generator

**Success Criteria**:
- All dataclass validation tests pass
- Decomposition calculations match manual calculations
- Left-aligned comparison produces expected results
- Test coverage > 90%

### Phase 2: Lens 4 Extensions (Day 3)

**Goal**: Add time-aligned comparison and time-to-second-purchase analysis

**Tasks**:
1. Implement time-aligned comparison mode
2. Implement `calculate_time_to_second_purchase()` helper
3. Implement `compare_cohort_pair()` for pairwise comparisons
4. Add integration tests with multiple cohorts
5. Add performance tests for large datasets

**Success Criteria**:
- Both alignment modes work correctly
- Time-to-second-purchase distributions are accurate
- Performance acceptable for 10K+ customers
- Documentation complete with examples

### Phase 3: Lens 5 Foundation (Days 4-5)

**Goal**: Implement core Lens 5 functionality with C3 data and health scoring

**Tasks**:
1. Create `lens5.py` module with dataclasses
2. Implement C3 data calculation (cohort revenue by period)
3. Implement repeat behavior analysis
4. Implement health score calculation
5. Write comprehensive unit tests

**Success Criteria**:
- All dataclass validation tests pass
- C3 data correctly aggregates revenue by cohort-period
- Health score calculation is reasonable and well-documented
- Test coverage > 90%

### Phase 4: Lens 5 Integration (Day 6)

**Goal**: Complete Lens 5 with integration and documentation

**Tasks**:
1. Implement `determine_cohort_quality_trend()` logic
2. Implement `calculate_revenue_predictability()` logic
3. Add integration tests combining Lens 4 + Lens 5
4. Create example notebooks showing Lens 4 + 5 usage
5. Update Track B AGENTS.md with completion status

**Success Criteria**:
- Lens 4 and 5 work together seamlessly
- Example notebooks demonstrate key insights
- Documentation complete
- Issues #34 and #35 can be closed

---

## Architectural Patterns to Follow

Based on Lens 1-3 analysis, both Lens 4 and 5 must follow these patterns:

### 1. Module Structure

```python
# customer_base_audit/analyses/lens4.py
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
from datetime import datetime
from typing import Sequence
import logging

# Module-level constants
PERCENTAGE_PRECISION = Decimal("0.01")  # 2 decimal places
MIN_COHORT_SIZE = 10  # Minimum customers for reliable cohort analysis
EXTREME_CHANGE_THRESHOLD = Decimal("500")  # 500% = 6x change threshold

logger = logging.getLogger(__name__)
```

### 2. Dataclass Design

**MUST USE**:
- `@dataclass(frozen=True)` for immutability
- `Decimal` for ALL financial and percentage fields (NOT `float`)
- `__post_init__` validation with clear error messages
- NumPy-style docstrings for all dataclasses

**Example**:
```python
@dataclass(frozen=True)
class CohortDecomposition:
    """Multiplicative decomposition of cohort revenue.

    Attributes
    ----------
    cohort_id:
        Cohort identifier (e.g., "2023-Q1")
    period_number:
        Period number (0 = acquisition period)
    ...
    """
    cohort_id: str
    period_number: int
    cohort_size: int
    # ... more fields with Decimal types

    def __post_init__(self) -> None:
        """Validate decomposition consistency."""
        if self.cohort_size < 0:
            raise ValueError(f"cohort_size must be >= 0, got {self.cohort_size}")
        # ... more validations
```

### 3. Function Design

**Type Hints**:
- Use `Sequence[T]` for input parameters (not `List[T]`)
- Use `| None` for optional parameters (Python 3.10+)
- Return specific dataclass types

**Docstrings**:
- NumPy-style with Parameters, Returns, Raises, Examples, Notes
- Include runnable doctests in Examples section
- Document edge cases in Notes section

**Validation**:
- Check for empty inputs early
- Validate cross-parameter relationships
- Raise `ValueError` with context

**Example**:
```python
def compare_cohorts(
    period_aggregations: Sequence[PeriodAggregation],
    cohort_assignments: Sequence[CohortAssignment],
    alignment_type: str = "left-aligned",
) -> Lens4Metrics:
    """Perform multi-cohort comparison analysis.

    Parameters
    ----------
    period_aggregations:
        Customer transaction aggregations by period
    cohort_assignments:
        Cohort membership for each customer
    alignment_type:
        "left-aligned" or "time-aligned"

    Returns
    -------
    Lens4Metrics:
        Multi-cohort comparison results

    Raises
    ------
    ValueError:
        If alignment_type is invalid or inputs are empty

    Examples
    --------
    >>> lens4 = compare_cohorts(periods, cohorts)
    >>> lens4.alignment_type
    'left-aligned'

    Notes
    -----
    Left-aligned comparison aligns cohorts by lifecycle stage.
    Time-aligned comparison uses calendar periods.
    """
    # Validate inputs
    if not period_aggregations:
        raise ValueError("period_aggregations cannot be empty")
    if not cohort_assignments:
        raise ValueError("cohort_assignments cannot be empty")
    if alignment_type not in ("left-aligned", "time-aligned"):
        raise ValueError(f"Invalid alignment_type: {alignment_type}")

    # Implementation...
```

### 4. Calculations

**Precision**:
- Quantize all Decimal results: `.quantize(PERCENTAGE_PRECISION, rounding=ROUND_HALF_UP)`
- Use generator expressions for aggregation
- Protect against division by zero

**Example**:
```python
# Calculate percentage with proper precision
if cohort_size > 0:
    pct_active = (Decimal(str(active_customers)) / Decimal(str(cohort_size)) * 100).quantize(
        PERCENTAGE_PRECISION, rounding=ROUND_HALF_UP
    )
else:
    pct_active = Decimal("0.00")
```

### 5. Error Handling

**Logging**:
- Use Python `logging` module
- Log warnings for data quality issues
- Explain threshold choices in comments

**Example**:
```python
# Warn about extreme changes
if revenue_change_pct > EXTREME_CHANGE_THRESHOLD:
    logger.warning(
        f"Extreme revenue change detected for cohort {cohort_id}: "
        f"{revenue_change_pct}% change. This may indicate data quality issues."
    )
```

---

## Testing Strategy

### Test Organization

**File**: `tests/test_lens4.py` and `tests/test_lens5.py`

**Structure**:
```python
import pytest
from decimal import Decimal
from datetime import datetime

class TestCohortDecomposition:
    """Tests for CohortDecomposition dataclass."""

    def test_valid_decomposition(self):
        """Test CohortDecomposition with valid values."""
        # ...

    def test_negative_cohort_size_raises_error(self):
        """Test that negative cohort_size raises ValueError."""
        with pytest.raises(ValueError, match="cohort_size must be >= 0"):
            # ...

class TestCompareCohorts:
    """Tests for compare_cohorts() main function."""

    def test_left_aligned_comparison(self):
        """Test left-aligned cohort comparison."""
        # ...

    def test_empty_inputs_raise_error(self):
        """Test that empty inputs raise ValueError."""
        # ...

    @pytest.mark.slow
    def test_large_dataset_performance(self):
        """Verify acceptable performance with 10k customers."""
        # ...
```

### Required Test Coverage

**For Each Dataclass**:
1. Happy path with valid inputs
2. Each field validation (negative values, out of range, etc.)
3. Cross-field validation (e.g., active_customers <= cohort_size)
4. Immutability test (attempt to modify frozen field)

**For Main Functions**:
1. Empty input handling
2. Single cohort / single period
3. Multiple cohorts / multiple periods
4. Alignment mode differences
5. Edge cases (100% churn, 0 revenue, etc.)
6. Performance with large datasets (`@pytest.mark.slow`)
7. Logging warnings (`caplog` fixture)

**For Helper Functions**:
1. Unit tests for each calculation
2. Zero division protection
3. Precision/rounding correctness

**Target Coverage**: >90% for all modules

---

## Integration Points

### With Existing Track B Code

**Dependencies** (Lens 4 & 5 depend on):
1. `customer_base_audit.foundation.data_mart.PeriodAggregation` - transaction data aggregated by period
2. `customer_base_audit.foundation.cohorts.CohortAssignment` - cohort membership
3. `customer_base_audit.analyses.lens3.Lens3Metrics` - optionally compose with Lens 3

**Provides** (Lens 4 & 5 provide to):
1. CLI reporting (future Phase 5 work)
2. Example notebooks (Track C)
3. Integration tests (Track C)

### With CLV Models

**Lens 5 Health Score** could optionally integrate:
- BG/NBD predicted retention rates
- Gamma-Gamma predicted customer values
- CLV predictions for forward-looking health assessment

**Not required for MVP**, but nice enhancement for future work.

### With Validation Framework

**Use validation framework patterns**:
- Temporal train/test splitting for cohort analysis validation
- Performance metrics (MAE, MAPE) for cohort predictions
- Cross-validation across cohorts

**Example**: Validate that Lens 4 decomposition accurately predicts next period revenue.

---

## File Structure Summary

```
customer_base_audit/
└── analyses/
    ├── lens1.py (✅ complete)
    ├── lens2.py (✅ complete)
    ├── lens3.py (✅ complete)
    ├── lens4.py (🔄 to implement - 400-500 lines estimated)
    └── lens5.py (🔄 to implement - 400-500 lines estimated)

tests/
├── test_lens1.py (✅ complete, 481 lines)
├── test_lens2.py (✅ complete, 738 lines)
├── test_lens3.py (✅ complete, 595 lines)
├── test_lens4.py (🔄 to implement - 600-700 lines estimated)
└── test_lens5.py (🔄 to implement - 600-700 lines estimated)
```

**Estimated Total**: ~2,600-2,800 lines of new code (implementation + tests)

---

## Key Formulas Reference

### Lens 4: Multiplicative Decomposition

```
Revenue = Cohort Size × % Active × AOF × AOV × Margin

Where:
  Cohort Size = Number of customers acquired in cohort
  % Active = (Active customers / Cohort Size) × 100
  AOF = Total Orders / Active Customers
  AOV = Total Revenue / Total Orders
  Margin = Average profit margin % (or 100% for revenue-only)
```

### Lens 4: Repeat Purchase Rate

```
Repeat Rate = (Customers with 2+ purchases / Cohort Size) × 100
```

### Lens 4: Time to Second Purchase

```
Days to Second = (Second Purchase Date - First Purchase Date).days

Cumulative Rate (Period N) = (Customers with second purchase by Period N / Cohort Size) × 100
```

### Lens 5: Revenue Contribution

```
Period Revenue % from Cohort = (Cohort Revenue in Period / Total Period Revenue) × 100
```

### Lens 5: Health Score

```
Health Score = (
    Retention Weight × Overall Retention Rate +
    Quality Weight × Cohort Quality Score +
    Predictability Weight × Revenue Predictability +
    Independence Weight × (100 - Acquisition Dependence)
)

Where weights sum to 1.0:
  Retention Weight = 0.30
  Quality Weight = 0.30
  Predictability Weight = 0.20
  Independence Weight = 0.20
```

### Lens 5: Health Grade

```
A: 90-100
B: 80-89
C: 70-79
D: 60-69
F: <60
```

---

## Architectural Deviation Warning

**Lens 3 uses `float` instead of `Decimal`** - this is an architectural inconsistency.

**Lens 4 and 5 MUST use `Decimal`** for:
- All financial values (revenue, AOV, etc.)
- All percentage values (pct_active, repeat_rate, etc.)
- Quantize to 2 decimal places: `Decimal("0.01")`

Following Lens 1 and Lens 2 patterns, NOT Lens 3.

---

## References

### Source Material
1. **"The Customer-Base Audit"** by Fader, Hardie, and Ross
   - Chapter 6: Lens 4 - Comparing and Contrasting Cohort Performance
   - Chapter 7: Lens 5 - How Healthy Is Our Customer Base?

### Web Resources
2. **Academic Papers**:
   - "A Cross-Cohort Changepoint Model for Customer-Base Analysis" (Gopalakrishnan, Bradlow, Fader)
   - "RFM and CLV: Using Iso-value Curves for Customer Base Analysis" (Fader & Hardie)

3. **Commercial Tools**:
   - Theta CLV (Customer-Based Corporate Valuation)
   - Peel Insights (e-commerce cohort analytics)

4. **Industry Examples**:
   - Slack, Dropbox, Lyft S-1 filings (C3 disclosures)
   - REVOLVE, Farfetch, Chewy (e-commerce C3 examples)

### Codebase References
5. **Existing Implementations**:
   - `customer_base_audit/analyses/lens1.py` (291 lines)
   - `customer_base_audit/analyses/lens2.py` (371 lines)
   - `customer_base_audit/analyses/lens3.py` (423 lines)

6. **Foundation Dependencies**:
   - `customer_base_audit/foundation/data_mart.py` (PeriodAggregation)
   - `customer_base_audit/foundation/cohorts.py` (CohortAssignment)

---

## Next Steps

1. **Review this research document** with stakeholders
2. **Start with Lens 4 Phase 1** (multiplicative decomposition)
3. **Follow established patterns** from Lens 1-2 (NOT Lens 3)
4. **Test thoroughly** with synthetic data before production
5. **Document as you go** with examples and doctests

**Estimated Timeline**: 6 days total (3 days per lens)

**Issues to Close**: #34 (Lens 4), #35 (Lens 5)

---

**End of Research Document**
