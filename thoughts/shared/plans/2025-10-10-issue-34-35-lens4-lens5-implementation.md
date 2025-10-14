# Lens 4 & 5 Implementation Plan

## Overview

Implement Lens 4 (Multi-Cohort Comparison) and Lens 5 (Overall Customer Base Health) for the AutoCLV Track B analytics framework. These lenses complete the Five Lenses framework from "The Customer-Base Audit" book, providing cohort-to-cohort comparison and integrative customer base health assessment.

**Timeline**: 6 days (3 days per lens)
**Issues**: #34 (Lens 4), #35 (Lens 5)
**Track**: B
**Estimated LOC**: ~2,600-2,800 lines (implementation + tests)

## Current State Analysis

### What Exists Now

**Complete** (Track B Phase 1-4):
- ✅ Lens 1: Single Period Analysis (`customer_base_audit/analyses/lens1.py`, 291 lines)
- ✅ Lens 2: Period-to-Period Comparison (`customer_base_audit/analyses/lens2.py`, 371 lines)
- ✅ Lens 3: Cohort Evolution (`customer_base_audit/analyses/lens3.py`, 423 lines)
- ✅ Cohort infrastructure (`customer_base_audit/foundation/cohorts.py`, 642 lines)
- ✅ Data mart with PeriodAggregation (`customer_base_audit/foundation/data_mart.py`, 340 lines)
- ✅ BG/NBD and Gamma-Gamma models (Phase 3)
- ✅ Validation framework (Phase 4)

**Missing** (Track B Phase 5):
- ❌ Lens 4: Multi-cohort comparison functionality
- ❌ Lens 5: Customer base health assessment
- ❌ Example notebooks demonstrating Lens 4+5 usage

### Key Architectural Patterns (from Lens 1-3 analysis)

1. **Frozen dataclasses** with `__post_init__` validation
2. **Decimal precision** for all financial/percentage fields (NOT float like Lens 3)
3. **NumPy-style docstrings** with Parameters, Returns, Raises, Examples, Notes
4. **Comprehensive validation**: empty inputs, duplicates, cross-field constraints
5. **Generator expressions** for memory-efficient aggregation
6. **Python logging** for data quality warnings
7. **Test coverage > 90%** with pytest class organization

### Foundation Types Available

From `customer_base_audit/foundation/cohorts.py`:
- `CohortDefinition(cohort_id, start_date, end_date, metadata)` - frozen dataclass
- `assign_cohorts()` - assigns customers to cohorts
- `create_monthly_cohorts()`, `create_quarterly_cohorts()`, `create_yearly_cohorts()`

From `customer_base_audit/foundation/data_mart.py`:
- `PeriodAggregation(customer_id, period_start, period_end, total_orders, total_spend, total_margin, total_quantity, last_transaction_ts)` - NOT frozen (uses slots)
- Uses `float` for financial fields (inconsistent with Lens 1-2, but we must accept it)

## Desired End State

### Lens 4 Complete

**File**: `customer_base_audit/analyses/lens4.py` (~450 lines)
**Test File**: `tests/test_lens4.py` (~650 lines)

**Functionality**:
1. Multiplicative revenue decomposition: `Revenue = Cohort Size × % Active × AOF × AOV × Margin`
2. Left-aligned comparison (by cohort age): Period 0, 1, 2... relative to acquisition
3. Time-aligned comparison (by calendar time): All cohorts in Q1 2023, Q2 2023...
4. Time to second purchase distribution analysis
5. Cohort-to-cohort pairwise comparisons

**Dataclasses** (4 total):
- `CohortDecomposition` - revenue decomposition for one cohort-period
- `TimeToSecondPurchase` - repeat purchase timing for one cohort
- `CohortComparison` - pairwise comparison between two cohorts
- `Lens4Metrics` - top-level results container

**Main Function**: `compare_cohorts(period_aggregations, cohort_assignments, alignment_type, include_margin) -> Lens4Metrics`

### Lens 5 Complete

**File**: `customer_base_audit/analyses/lens5.py` (~450 lines)
**Test File**: `tests/test_lens5.py` (~650 lines)

**Functionality**:
1. Customer Cohort Chart (C3) data - revenue by cohort-period for visualization
2. Repeat purchase behavior by cohort
3. Overall customer base health score (0-100) with letter grade (A-F)
4. Cohort quality trend detection (improving/stable/declining)
5. Revenue predictability and acquisition dependence metrics

**Dataclasses** (4 total):
- `CohortRevenuePeriod` - revenue contribution from one cohort in one period
- `CohortRepeatBehavior` - repeat purchase behavior for one cohort
- `CustomerBaseHealthScore` - overall health assessment
- `Lens5Metrics` - top-level results container

**Main Function**: `assess_customer_base_health(period_aggregations, cohort_assignments, analysis_start_date, analysis_end_date) -> Lens5Metrics`

### Verification

#### Automated:
- [ ] All tests pass: `pytest tests/test_lens4.py tests/test_lens5.py -v`
- [ ] Test coverage > 90%: `pytest --cov=customer_base_audit/analyses/lens4 --cov=customer_base_audit/analyses/lens5 --cov-report=term-missing`
- [ ] Type checking passes: `mypy customer_base_audit/analyses/lens4.py customer_base_audit/analyses/lens5.py`
- [ ] Linting passes: `ruff check customer_base_audit/analyses/lens4.py customer_base_audit/analyses/lens5.py`

#### Manual:
- [ ] Lens 4 produces expected decomposition with Texas CLV synthetic data
- [ ] Lens 5 health score calculation is reasonable for test scenarios
- [ ] Example notebooks run end-to-end demonstrating key insights
- [ ] Documentation is clear and complete with runnable doctests

## What We're NOT Doing

**Out of Scope for MVP**:
1. ❌ Integration with BG/NBD/Gamma-Gamma predictions in Lens 5 health score (future enhancement)
2. ❌ Interactive visualizations for C3 charts (provide data only, visualization is Track C or external)
3. ❌ Real-time streaming cohort analysis (batch only)
4. ❌ Cohort segmentation by attributes (channel, campaign, etc.) - use CohortDefinition metadata instead
5. ❌ Statistical significance testing for cohort comparisons (future enhancement)
6. ❌ CLI commands for Lens 4+5 (separate PR after implementation)
7. ❌ Dashboard integration (Track C work)

## Implementation Approach

### Strategy

1. **Follow Lens 1-2 patterns strictly** - Use Decimal (NOT float), frozen dataclasses, comprehensive validation
2. **Build incrementally** - Dataclasses → helpers → main function → tests → integration
3. **Test with synthetic data** - Use Texas CLV generator for realistic cohort scenarios
4. **Document as we go** - Doctests in all functions, clear examples in docstrings

### Key Technical Decisions

**Decision 1: Use Decimal throughout**
- **Rationale**: Lens 1 and Lens 2 use Decimal for financial/percentage precision
- **Impact**: Lens 4 and 5 will use Decimal even though PeriodAggregation uses float
- **Trade-off**: Requires float→Decimal conversion, but maintains consistency

**Decision 2: Cohort assignments via simple dict**
- **Rationale**: `assign_cohorts()` returns `dict[str, str]` (customer_id → cohort_id)
- **Impact**: We'll build `dict[str, str]` from `CohortDefinition` list for lookups
- **Trade-off**: Simple and matches existing pattern

**Decision 3: Time-to-second-purchase uses PeriodAggregation**
- **Rationale**: We need transaction timestamps, which are in `last_transaction_ts` field
- **Impact**: Must reconstruct transaction sequence from period aggregations
- **Trade-off**: Less precise than raw transactions, but works with data mart architecture

**Decision 4: Health score uses weighted formula**
- **Rationale**: Retention (30%) + Quality (30%) + Predictability (20%) + Independence (20%) = 100%
- **Impact**: Score is interpretable and actionable for business stakeholders
- **Trade-off**: Weights are somewhat arbitrary, but based on Customer-Base Audit best practices

---

## Phase 1: Lens 4 Foundation (Days 1-2)

### Overview

Implement core Lens 4 functionality with multiplicative decomposition and left-aligned comparison. This establishes the foundation for cohort-to-cohort analysis.

### Changes Required

#### 1. Create Lens 4 Module Structure

**File**: `customer_base_audit/analyses/lens4.py`

**Changes**: Create new file with module-level imports and constants

```python
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
EXTREME_CHANGE_THRESHOLD = Decimal("500")  # 500% = 6x change threshold (warn about data quality)

logger = logging.getLogger(__name__)
```

#### 2. Implement CohortDecomposition Dataclass

**File**: `customer_base_audit/analyses/lens4.py`

**Changes**: Add CohortDecomposition frozen dataclass with full validation

```python
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
```

#### 3. Implement calculate_cohort_decomposition() Helper

**File**: `customer_base_audit/analyses/lens4.py`

**Changes**: Add helper function to calculate decomposition for one cohort-period

```python
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
```

#### 4. Implement compare_cohorts() Main Function (Left-Aligned Only)

**File**: `customer_base_audit/analyses/lens4.py`

**Changes**: Add main function with left-aligned comparison mode

```python
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
    # Validate inputs
    if not period_aggregations:
        raise ValueError("period_aggregations cannot be empty")
    if not cohort_assignments:
        raise ValueError("cohort_assignments cannot be empty")
    if alignment_type not in ("left-aligned", "time-aligned"):
        raise ValueError(
            f"alignment_type must be 'left-aligned' or 'time-aligned', got '{alignment_type}'"
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

    # Calculate decompositions per cohort-period
    cohort_decompositions: list[CohortDecomposition] = []

    if alignment_type == "left-aligned":
        # Left-aligned: period 0 = acquisition period, 1 = next period, etc.
        # Group periods by cohort and relative period number
        cohort_period_data: dict[tuple[str, int], list[PeriodAggregation]] = {}

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

    else:
        # Time-aligned: period 0 = first calendar period, 1 = second, etc.
        # (Implementation in Phase 2)
        raise NotImplementedError("Time-aligned comparison will be implemented in Phase 2")

    # Placeholder for time-to-second-purchase and cohort comparisons (Phase 2)
    time_to_second_purchase: list[TimeToSecondPurchase] = []
    cohort_comparisons: list[CohortComparison] = []

    return Lens4Metrics(
        cohort_decompositions=cohort_decompositions,
        time_to_second_purchase=time_to_second_purchase,
        cohort_comparisons=cohort_comparisons,
        alignment_type=alignment_type,
    )
```

#### 5. Implement Lens4Metrics Dataclass

**File**: `customer_base_audit/analyses/lens4.py`

**Changes**: Add Lens4Metrics frozen dataclass

```python
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
```

#### 6. Create Test Structure

**File**: `tests/test_lens4.py`

**Changes**: Create test file with dataclass validation tests

```python
"""Tests for CLV Lens 4: Multi-cohort comparison."""

from datetime import datetime, timezone
from decimal import Decimal

import pytest

from customer_base_audit.analyses.lens4 import (
    CohortDecomposition,
    Lens4Metrics,
    compare_cohorts,
    calculate_cohort_decomposition,
)
from customer_base_audit.foundation.data_mart import PeriodAggregation


class TestCohortDecomposition:
    """Tests for CohortDecomposition dataclass."""

    def test_valid_decomposition(self):
        """Test CohortDecomposition with valid values."""
        decomp = CohortDecomposition(
            cohort_id="2023-Q1",
            period_number=0,
            cohort_size=100,
            active_customers=80,
            pct_active=Decimal("80.00"),
            total_orders=160,
            aof=Decimal("2.00"),
            total_revenue=Decimal("8000.00"),
            aov=Decimal("50.00"),
            margin=Decimal("100.00"),
            revenue=Decimal("8000.00"),
        )

        assert decomp.cohort_id == "2023-Q1"
        assert decomp.period_number == 0
        assert decomp.cohort_size == 100
        assert decomp.active_customers == 80

    def test_negative_period_number_raises_error(self):
        """Test that negative period_number raises ValueError."""
        with pytest.raises(ValueError, match="period_number must be >= 0"):
            CohortDecomposition(
                cohort_id="2023-Q1",
                period_number=-1,
                cohort_size=100,
                active_customers=80,
                pct_active=Decimal("80.00"),
                total_orders=160,
                aof=Decimal("2.00"),
                total_revenue=Decimal("8000.00"),
                aov=Decimal("50.00"),
                margin=Decimal("100.00"),
                revenue=Decimal("8000.00"),
            )

    def test_active_exceeds_cohort_size_raises_error(self):
        """Test that active_customers > cohort_size raises ValueError."""
        with pytest.raises(ValueError, match="cannot exceed cohort_size"):
            CohortDecomposition(
                cohort_id="2023-Q1",
                period_number=0,
                cohort_size=100,
                active_customers=120,  # More than cohort size!
                pct_active=Decimal("120.00"),
                total_orders=160,
                aof=Decimal("2.00"),
                total_revenue=Decimal("8000.00"),
                aov=Decimal("50.00"),
                margin=Decimal("100.00"),
                revenue=Decimal("8000.00"),
            )

    # Add more validation tests...


class TestCalculateCohortDecomposition:
    """Tests for calculate_cohort_decomposition() helper."""

    def test_simple_decomposition(self):
        """Test decomposition with simple period data."""
        period_data = [
            PeriodAggregation(
                customer_id="C1",
                period_start=datetime(2023, 1, 1, tzinfo=timezone.utc),
                period_end=datetime(2023, 2, 1, tzinfo=timezone.utc),
                total_orders=2,
                total_spend=100.00,
                total_margin=30.00,
                total_quantity=3,
            ),
            PeriodAggregation(
                customer_id="C2",
                period_start=datetime(2023, 1, 1, tzinfo=timezone.utc),
                period_end=datetime(2023, 2, 1, tzinfo=timezone.utc),
                total_orders=3,
                total_spend=150.00,
                total_margin=45.00,
                total_quantity=5,
            ),
        ]

        decomp = calculate_cohort_decomposition(
            cohort_id="2023-Q1",
            period_number=0,
            cohort_size=2,
            period_data=period_data,
            include_margin=True,
        )

        assert decomp.cohort_id == "2023-Q1"
        assert decomp.period_number == 0
        assert decomp.cohort_size == 2
        assert decomp.active_customers == 2
        assert decomp.pct_active == Decimal("100.00")
        assert decomp.total_orders == 5
        assert decomp.aof == Decimal("2.50")
        assert decomp.total_revenue == Decimal("250.00")
        assert decomp.aov == Decimal("50.00")
        # Margin = 75 / 250 * 100 = 30%
        assert decomp.margin == Decimal("30.00")

    # Add more helper tests...


class TestCompareCohorts:
    """Tests for compare_cohorts() main function."""

    def test_empty_period_aggregations_raises_error(self):
        """Test that empty period_aggregations raises ValueError."""
        with pytest.raises(ValueError, match="period_aggregations cannot be empty"):
            compare_cohorts([], {})

    def test_empty_cohort_assignments_raises_error(self):
        """Test that empty cohort_assignments raises ValueError."""
        periods = [
            PeriodAggregation(
                customer_id="C1",
                period_start=datetime(2023, 1, 1, tzinfo=timezone.utc),
                period_end=datetime(2023, 2, 1, tzinfo=timezone.utc),
                total_orders=1,
                total_spend=50.00,
                total_margin=15.00,
                total_quantity=1,
            )
        ]
        with pytest.raises(ValueError, match="cohort_assignments cannot be empty"):
            compare_cohorts(periods, {})

    def test_invalid_alignment_type_raises_error(self):
        """Test that invalid alignment_type raises ValueError."""
        periods = [
            PeriodAggregation(
                customer_id="C1",
                period_start=datetime(2023, 1, 1, tzinfo=timezone.utc),
                period_end=datetime(2023, 2, 1, tzinfo=timezone.utc),
                total_orders=1,
                total_spend=50.00,
                total_margin=15.00,
                total_quantity=1,
            )
        ]
        cohort_assignments = {"C1": "2023-Q1"}

        with pytest.raises(ValueError, match="alignment_type must be"):
            compare_cohorts(periods, cohort_assignments, alignment_type="invalid")

    # Add more integration tests...
```

### Success Criteria

#### Automated Verification:
- [ ] Dataclass validation tests pass: `pytest tests/test_lens4.py::TestCohortDecomposition -v`
- [ ] Helper function tests pass: `pytest tests/test_lens4.py::TestCalculateCohortDecomposition -v`
- [ ] Main function tests pass: `pytest tests/test_lens4.py::TestCompareCohorts -v`
- [ ] Type checking passes: `mypy customer_base_audit/analyses/lens4.py`
- [ ] Linting passes: `ruff check customer_base_audit/analyses/lens4.py`

#### Manual Verification:
- [ ] Decomposition calculations match manual calculations for simple test case
- [ ] Left-aligned comparison produces expected period numbers (0, 1, 2...)
- [ ] Edge cases handled correctly (zero customers, zero orders, etc.)

**Implementation Note**: After completing Phase 1 and all automated verification passes, pause for manual confirmation before proceeding to Phase 2.

---

## Phase 2: Lens 4 Extensions (Day 3)

### Overview

Add time-aligned comparison, time-to-second-purchase analysis, and cohort-to-cohort pairwise comparisons. This completes Lens 4 functionality.

### Changes Required

#### 1. Implement TimeToSecondPurchase and CohortComparison Dataclasses

**File**: `customer_base_audit/analyses/lens4.py`

**Changes**: Add two frozen dataclasses with validation

```python
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
```

#### 2. Implement Time-Aligned Mode in compare_cohorts()

**File**: `customer_base_audit/analyses/lens4.py`

**Changes**: Add time-aligned branch in compare_cohorts()

```python
# In compare_cohorts(), replace NotImplementedError with:
else:  # time-aligned
    # Time-aligned: period 0 = first calendar period globally, 1 = second, etc.
    # Determine all unique period starts
    all_period_starts = sorted(set(p.period_start for p in period_aggregations))
    period_start_to_idx = {ps: idx for idx, ps in enumerate(all_period_starts)}

    # Group periods by cohort and absolute period index
    cohort_period_data: dict[tuple[str, int], list[PeriodAggregation]] = {}

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
```

#### 3. Implement calculate_time_to_second_purchase() Helper

**File**: `customer_base_audit/analyses/lens4.py`

**Changes**: Add helper to calculate time-to-second-purchase for one cohort

```python
def calculate_time_to_second_purchase(
    cohort_id: str,
    cohort_size: int,
    customer_periods: dict[str, list[PeriodAggregation]],  # customer_id -> sorted periods
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

    Notes
    -----
    Approximation: We use period boundaries to estimate time to second purchase.
    Actual days = (second_period.period_start - first_period.period_start).days
    This is less precise than using exact transaction timestamps, but works with
    the data mart architecture.
    """
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
        repeat_rate = (Decimal(str(customers_with_repeat)) / Decimal(str(cohort_size)) * 100).quantize(
            PERCENTAGE_PRECISION, rounding=ROUND_HALF_UP
        )
    else:
        repeat_rate = Decimal("0.00")

    # Calculate median and mean
    if days_to_second:
        days_sorted = sorted(days_to_second)
        n = len(days_sorted)
        if n % 2 == 0:
            median_days = Decimal(str((days_sorted[n//2 - 1] + days_sorted[n//2]) / 2)).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            )
        else:
            median_days = Decimal(str(days_sorted[n//2])).quantize(
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
    max_periods = max(len(periods) for periods in customer_periods.values()) if customer_periods else 0

    for period_num in range(max_periods):
        customers_repeated_by_period = sum(
            1 for periods in customer_periods.values() if len(periods) > period_num + 1
        )
        if cohort_size > 0:
            cumulative_pct = (
                Decimal(str(customers_repeated_by_period)) / Decimal(str(cohort_size)) * 100
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
```

#### 4. Implement compare_cohort_pair() Helper

**File**: `customer_base_audit/analyses/lens4.py`

**Changes**: Add helper to compare two cohorts at equivalent period

```python
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
        if old > Decimal("0"):
            return ((new - old) / old * 100).quantize(
                PERCENTAGE_PRECISION, rounding=ROUND_HALF_UP
            )
        elif new > Decimal("0"):
            return Decimal("100.00")  # Went from 0 to positive
        else:
            return Decimal("0.00")  # Both are 0

    pct_active_change_pct = calc_pct_change(cohort_a_decomp.pct_active, cohort_b_decomp.pct_active)
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
```

#### 5. Integrate Helpers into compare_cohorts()

**File**: `customer_base_audit/analyses/lens4.py`

**Changes**: Call helpers before returning Lens4Metrics

```python
# At end of compare_cohorts(), before return:

# Calculate time to second purchase for each cohort
time_to_second_purchase_list: list[TimeToSecondPurchase] = []
for cohort_id, cohort_size in sorted(cohort_sizes.items()):
    # Get customer periods for this cohort
    cohort_customer_periods: dict[str, list[PeriodAggregation]] = {}
    for customer_id, cid in cohort_assignments.items():
        if cid == cohort_id:
            customer_periods = periods_by_customer.get(customer_id, [])
            if customer_periods:
                cohort_customer_periods[customer_id] = customer_periods

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
        # Compare consecutive cohorts (assumes cohorts are chronologically sorted)
        for i in range(len(decomps) - 1):
            comparison = compare_cohort_pair(decomps[i], decomps[i + 1])
            cohort_comparisons_list.append(comparison)

return Lens4Metrics(
    cohort_decompositions=cohort_decompositions,
    time_to_second_purchase=time_to_second_purchase_list,
    cohort_comparisons=cohort_comparisons_list,
    alignment_type=alignment_type,
)
```

#### 6. Add Comprehensive Tests

**File**: `tests/test_lens4.py`

**Changes**: Add test classes for new dataclasses and functions

```python
class TestTimeToSecondPurchase:
    """Tests for TimeToSecondPurchase dataclass."""

    def test_valid_metrics(self):
        """Test TimeToSecondPurchase with valid values."""
        # ...

    def test_negative_customers_with_repeat_raises_error(self):
        """Test that negative customers_with_repeat raises ValueError."""
        # ...


class TestCohortComparison:
    """Tests for CohortComparison dataclass."""

    def test_valid_comparison(self):
        """Test CohortComparison with valid values."""
        # ...


class TestCalculateTimeToSecondPurchase:
    """Tests for calculate_time_to_second_purchase() helper."""

    def test_two_periods_per_customer(self):
        """Test time to second purchase with 2 periods per customer."""
        # ...


class TestCompareCohortPair:
    """Tests for compare_cohort_pair() helper."""

    def test_simple_comparison(self):
        """Test comparing two cohorts at same period."""
        # ...

    def test_different_periods_raises_error(self):
        """Test that comparing different periods raises ValueError."""
        # ...


class TestCompareCohorts:
    """Extended tests for compare_cohorts() main function."""

    def test_time_aligned_mode(self):
        """Test time-aligned comparison mode."""
        # ...

    def test_multiple_cohorts(self):
        """Test with multiple cohorts over multiple periods."""
        # ...

    @pytest.mark.slow
    def test_large_dataset_performance(self):
        """Verify acceptable performance with 10k customers."""
        # ...
```

### Success Criteria

#### Automated Verification:
- [ ] All Lens 4 tests pass: `pytest tests/test_lens4.py -v`
- [ ] Test coverage > 90%: `pytest --cov=customer_base_audit/analyses/lens4 --cov-report=term-missing`
- [ ] Type checking passes: `mypy customer_base_audit/analyses/lens4.py`
- [ ] Linting passes: `ruff check customer_base_audit/analyses/lens4.py`
- [ ] Performance test passes for 10K+ customers

#### Manual Verification:
- [ ] Time-aligned mode produces correct calendar period alignment
- [ ] Time-to-second-purchase distributions look reasonable
- [ ] Cohort comparisons show expected differences
- [ ] Documentation is complete with working doctests

**Implementation Note**: After completing Phase 2 and all automated verification passes, pause for manual confirmation before proceeding to Phase 3.

---

## Phase 3: Lens 5 Foundation (Days 4-5)

### Overview

Implement Lens 5 core functionality: C3 data calculation, repeat behavior analysis, and health score calculation. This provides the foundation for customer base health assessment.

### Changes Required

#### 1. Create Lens 5 Module Structure

**File**: `customer_base_audit/analyses/lens5.py`

**Changes**: Create new file with module structure (similar to lens4.py)

```python
"""Lens 5: Overall Customer Base Health.

This lens provides an integrative view of customer base health by:
- Analyzing revenue contributions across cohorts (C3 data)
- Tracking repeat purchase behavior by cohort
- Calculating overall health score and grade
- Assessing cohort quality trends and revenue predictability

Reference: "The Customer-Base Audit" by Fader, Hardie, and Ross (Chapter 7)
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from datetime import datetime
from typing import Sequence
import logging

# Module-level constants
PERCENTAGE_PRECISION = Decimal("0.01")  # 2 decimal places

# Health score weights (must sum to 1.0)
RETENTION_WEIGHT = Decimal("0.30")  # 30% weight for overall retention
QUALITY_WEIGHT = Decimal("0.30")    # 30% weight for cohort quality trend
PREDICTABILITY_WEIGHT = Decimal("0.20")  # 20% weight for revenue predictability
INDEPENDENCE_WEIGHT = Decimal("0.20")    # 20% weight for acquisition independence

# Health grade thresholds
GRADE_A_THRESHOLD = Decimal("90")
GRADE_B_THRESHOLD = Decimal("80")
GRADE_C_THRESHOLD = Decimal("70")
GRADE_D_THRESHOLD = Decimal("60")

logger = logging.getLogger(__name__)
```

#### 2. Implement CohortRevenuePeriod and CohortRepeatBehavior Dataclasses

**File**: `customer_base_audit/analyses/lens5.py`

**Changes**: Add two frozen dataclasses

```python
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
        if not (Decimal("0") <= self.pct_of_period_revenue <= Decimal("100")):
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
        if not (Decimal("0") <= self.repeat_rate <= Decimal("100")):
            raise ValueError(f"repeat_rate must be in [0, 100], got {self.repeat_rate}")
        if self.repeat_buyers > 0 and self.avg_orders_per_repeat_buyer < 2:
            raise ValueError(
                f"avg_orders_per_repeat_buyer must be >= 2 (by definition), "
                f"got {self.avg_orders_per_repeat_buyer}"
            )
```

#### 3. Implement CustomerBaseHealthScore Dataclass

**File**: `customer_base_audit/analyses/lens5.py`

**Changes**: Add health score dataclass with comprehensive validation

```python
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
        if not (Decimal("0") <= self.overall_retention_rate <= Decimal("100")):
            raise ValueError(
                f"overall_retention_rate must be in [0, 100], "
                f"got {self.overall_retention_rate}"
            )
        if self.cohort_quality_trend not in ("improving", "stable", "declining"):
            raise ValueError(
                f"cohort_quality_trend must be 'improving', 'stable', or 'declining', "
                f"got '{self.cohort_quality_trend}'"
            )
        if not (Decimal("0") <= self.revenue_predictability_pct <= Decimal("100")):
            raise ValueError(
                f"revenue_predictability_pct must be in [0, 100], "
                f"got {self.revenue_predictability_pct}"
            )
        if not (Decimal("0") <= self.acquisition_dependence_pct <= Decimal("100")):
            raise ValueError(
                f"acquisition_dependence_pct must be in [0, 100], "
                f"got {self.acquisition_dependence_pct}"
            )
        if not (Decimal("0") <= self.health_score <= Decimal("100")):
            raise ValueError(f"health_score must be in [0, 100], got {self.health_score}")
        if self.health_grade not in ("A", "B", "C", "D", "F"):
            raise ValueError(
                f"health_grade must be 'A', 'B', 'C', 'D', or 'F', "
                f"got '{self.health_grade}'"
            )
```

#### 4. Implement calculate_health_score() Helper

**File**: `customer_base_audit/analyses/lens5.py`

**Changes**: Add helper to calculate weighted health score

```python
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
        overall_retention * RETENTION_WEIGHT +
        quality_score * QUALITY_WEIGHT +
        revenue_predictability * PREDICTABILITY_WEIGHT +
        independence_score * INDEPENDENCE_WEIGHT
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
```

#### 5. Implement assess_customer_base_health() Main Function (Partial)

**File**: `customer_base_audit/analyses/lens5.py`

**Changes**: Add main function with C3 data and repeat behavior (Phase 4 will add helpers)

```python
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
        p for p in period_aggregations
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

    # Calculate health score (Phase 4 will implement helpers)
    # Placeholder: use simple defaults
    health_score_obj = CustomerBaseHealthScore(
        total_customers=len(cohort_assignments),
        total_active_customers=len(set(p.customer_id for p in filtered_periods)),
        overall_retention_rate=Decimal("70.00"),  # TODO
        cohort_quality_trend="stable",  # TODO
        revenue_predictability_pct=Decimal("60.00"),  # TODO
        acquisition_dependence_pct=Decimal("30.00"),  # TODO
        health_score=Decimal("65.00"),  # TODO
        health_grade="C",  # TODO
    )

    return Lens5Metrics(
        cohort_revenue_contributions=cohort_revenue_contributions,
        cohort_repeat_behavior=cohort_repeat_behavior,
        health_score=health_score_obj,
        analysis_start_date=analysis_start_date,
        analysis_end_date=analysis_end_date,
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

    # Build results
    results: list[CohortRevenuePeriod] = []
    for (cohort_id, period_start), data in sorted(cohort_period_revenue.items()):
        total_revenue = data["total_revenue"].quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )
        active_customers = len(data["active_customers"])

        period_total = period_totals[period_start]
        if period_total > 0:
            pct_of_period_revenue = (
                total_revenue / period_total * 100
            ).quantize(PERCENTAGE_PRECISION, rounding=ROUND_HALF_UP)
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
    # Count periods per customer
    customer_period_counts: dict[str, int] = {}
    customer_orders: dict[str, int] = {}

    for period in periods:
        customer_id = period.customer_id
        customer_period_counts[customer_id] = (
            customer_period_counts.get(customer_id, 0) + 1
        )
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

        cohort_data[cohort_id]["cohort_size"] += 1

        period_count = customer_period_counts.get(customer_id, 0)
        if period_count == 1:
            cohort_data[cohort_id]["one_time_buyers"] += 1
        elif period_count >= 2:
            cohort_data[cohort_id]["repeat_buyers"] += 1
            orders = customer_orders.get(customer_id, 0)
            cohort_data[cohort_id]["repeat_orders"].append(orders)

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
            avg_orders = sum(data["repeat_orders"]) / repeat_buyers
            avg_orders_per_repeat_buyer = Decimal(str(avg_orders)).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            )
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
```

#### 6. Implement Lens5Metrics Dataclass

**File**: `customer_base_audit/analyses/lens5.py`

**Changes**: Add top-level results dataclass

```python
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
```

#### 7. Create Test Structure

**File**: `tests/test_lens5.py`

**Changes**: Create test file with dataclass validation tests (similar to test_lens4.py structure)

```python
"""Tests for CLV Lens 5: Overall customer base health."""

from datetime import datetime, timezone
from decimal import Decimal

import pytest

from customer_base_audit.analyses.lens5 import (
    CohortRevenuePeriod,
    CohortRepeatBehavior,
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
        # ...

    def test_negative_total_revenue_raises_error(self):
        """Test that negative total_revenue raises ValueError."""
        # ...

    # More validation tests...


class TestCohortRepeatBehavior:
    """Tests for CohortRepeatBehavior dataclass."""

    def test_valid_repeat_behavior(self):
        """Test CohortRepeatBehavior with valid values."""
        # ...

    def test_buyers_sum_not_equal_cohort_size_raises_error(self):
        """Test that one_time + repeat != cohort_size raises ValueError."""
        # ...

    # More validation tests...


class TestCustomerBaseHealthScore:
    """Tests for CustomerBaseHealthScore dataclass."""

    def test_valid_health_score(self):
        """Test CustomerBaseHealthScore with valid values."""
        # ...

    def test_invalid_cohort_quality_trend_raises_error(self):
        """Test that invalid trend raises ValueError."""
        # ...

    # More validation tests...


class TestCalculateHealthScore:
    """Tests for calculate_health_score() helper."""

    def test_perfect_health_score(self):
        """Test health score with perfect metrics."""
        score, grade = calculate_health_score(
            Decimal("100"), "improving", Decimal("100"), Decimal("0")
        )
        assert score == Decimal("94.00")  # 100*0.3 + 80*0.3 + 100*0.2 + 100*0.2
        assert grade == "A"

    # More helper tests...


class TestAssessCustomerBaseHealth:
    """Tests for assess_customer_base_health() main function."""

    def test_empty_period_aggregations_raises_error(self):
        """Test that empty period_aggregations raises ValueError."""
        # ...

    def test_invalid_date_range_raises_error(self):
        """Test that start >= end raises ValueError."""
        # ...

    # More integration tests...
```

### Success Criteria

#### Automated Verification:
- [ ] All Lens 5 dataclass tests pass: `pytest tests/test_lens5.py::TestCohortRevenuePeriod tests/test_lens5.py::TestCohortRepeatBehavior tests/test_lens5.py::TestCustomerBaseHealthScore -v`
- [ ] Helper function tests pass: `pytest tests/test_lens5.py::TestCalculateHealthScore -v`
- [ ] Main function basic tests pass: `pytest tests/test_lens5.py::TestAssessCustomerBaseHealth -v`
- [ ] Type checking passes: `mypy customer_base_audit/analyses/lens5.py`
- [ ] Linting passes: `ruff check customer_base_audit/analyses/lens5.py`

#### Manual Verification:
- [ ] C3 data calculations match manual calculations
- [ ] Repeat behavior percentages look reasonable
- [ ] Health score formula produces expected values
- [ ] Dataclass validation catches all edge cases

**Implementation Note**: After completing Phase 3 and all automated verification passes, pause for manual confirmation before proceeding to Phase 4.

---

## Phase 4: Lens 5 Integration (Day 6)

### Overview

Complete Lens 5 by implementing helper functions for cohort quality trend detection and revenue predictability. Add integration tests, update documentation, and close Issues #34 and #35.

### Changes Required

#### 1. Implement determine_cohort_quality_trend() Helper

**File**: `customer_base_audit/analyses/lens5.py`

**Changes**: Add helper to detect improving/stable/declining trend

```python
def determine_cohort_quality_trend(
    cohort_metrics: Sequence[CohortRepeatBehavior],
) -> str:
    """Determine if cohort quality is improving, stable, or declining.

    Compares repeat rates of recent cohorts vs historical cohorts.

    Logic:
    - If only 1 cohort: "stable"
    - If >= 2 cohorts: compare newest vs oldest repeat rates
      - If newest >= oldest * 1.1: "improving" (10% better)
      - If newest <= oldest * 0.9: "declining" (10% worse)
      - Otherwise: "stable"

    Parameters
    ----------
    cohort_metrics:
        Repeat behavior for all cohorts, sorted by cohort_id (chronological)

    Returns
    -------
    str:
        "improving", "stable", or "declining"

    Examples
    --------
    >>> metrics = [
    ...     CohortRepeatBehavior("2023-Q1", 100, 40, 60, Decimal("60.00"), Decimal("3.0")),
    ...     CohortRepeatBehavior("2023-Q2", 100, 35, 65, Decimal("65.00"), Decimal("3.2")),
    ...     CohortRepeatBehavior("2023-Q3", 100, 30, 70, Decimal("70.00"), Decimal("3.5")),
    ... ]
    >>> determine_cohort_quality_trend(metrics)
    'improving'
    """
    if len(cohort_metrics) < 2:
        return "stable"

    # Compare oldest vs newest
    oldest = cohort_metrics[0]
    newest = cohort_metrics[-1]

    oldest_repeat_rate = oldest.repeat_rate
    newest_repeat_rate = newest.repeat_rate

    # Avoid division by zero
    if oldest_repeat_rate == 0:
        if newest_repeat_rate > 0:
            return "improving"
        else:
            return "stable"

    # Calculate ratio
    ratio = newest_repeat_rate / oldest_repeat_rate

    if ratio >= Decimal("1.1"):
        return "improving"
    elif ratio <= Decimal("0.9"):
        return "declining"
    else:
        return "stable"
```

#### 2. Implement calculate_revenue_predictability() Helper

**File**: `customer_base_audit/analyses/lens5.py`

**Changes**: Add helper to calculate predictable revenue percentage

```python
def calculate_revenue_predictability(
    cohort_revenue_contributions: Sequence[CohortRevenuePeriod],
    newest_cohort_id: str,
) -> Decimal:
    """Calculate what % of revenue is predictable from existing cohorts.

    Assumes newest cohort represents unpredictable acquisition-driven revenue.
    All other cohorts are considered predictable based on historical patterns.

    Formula:
        predictability = (total_revenue - newest_cohort_revenue) / total_revenue * 100

    Parameters
    ----------
    cohort_revenue_contributions:
        Revenue contributions from all cohorts
    newest_cohort_id:
        ID of the newest cohort (considered unpredictable)

    Returns
    -------
    Decimal:
        Revenue predictability percentage (0-100)

    Examples
    --------
    >>> contributions = [
    ...     CohortRevenuePeriod("2023-Q1", datetime(2023, 10, 1, tzinfo=timezone.utc), Decimal("5000"), Decimal("50"), 40, Decimal("125")),
    ...     CohortRevenuePeriod("2023-Q2", datetime(2023, 10, 1, tzinfo=timezone.utc), Decimal("3000"), Decimal("30"), 30, Decimal("100")),
    ...     CohortRevenuePeriod("2023-Q3", datetime(2023, 10, 1, tzinfo=timezone.utc), Decimal("2000"), Decimal("20"), 20, Decimal("100")),
    ... ]
    >>> calculate_revenue_predictability(contributions, "2023-Q3")
    Decimal('80.00')
    """
    # Sum total revenue
    total_revenue = sum(c.total_revenue for c in cohort_revenue_contributions)

    # Sum revenue from newest cohort
    newest_revenue = sum(
        c.total_revenue for c in cohort_revenue_contributions
        if c.cohort_id == newest_cohort_id
    )

    # Calculate predictability
    if total_revenue > 0:
        predictable_revenue = total_revenue - newest_revenue
        predictability = (
            predictable_revenue / total_revenue * 100
        ).quantize(PERCENTAGE_PRECISION, rounding=ROUND_HALF_UP)
    else:
        predictability = Decimal("0.00")

    return predictability
```

#### 3. Integrate Helpers into assess_customer_base_health()

**File**: `customer_base_audit/analyses/lens5.py`

**Changes**: Replace placeholder health score calculation with real implementation

```python
# In assess_customer_base_health(), replace placeholder health_score_obj with:

# Calculate overall retention rate
total_customers = len(cohort_assignments)
active_customers_in_window = len(set(p.customer_id for p in filtered_periods))

if total_customers > 0:
    overall_retention_rate = (
        Decimal(str(active_customers_in_window)) / Decimal(str(total_customers)) * 100
    ).quantize(PERCENTAGE_PRECISION, rounding=ROUND_HALF_UP)
else:
    overall_retention_rate = Decimal("0.00")

# Determine cohort quality trend
cohort_quality_trend = determine_cohort_quality_trend(cohort_repeat_behavior)

# Calculate revenue predictability
if cohort_revenue_contributions:
    newest_cohort_id = max(
        cohort_revenue_contributions, key=lambda c: c.cohort_id
    ).cohort_id
    revenue_predictability_pct = calculate_revenue_predictability(
        cohort_revenue_contributions, newest_cohort_id
    )

    # Calculate acquisition dependence (newest cohort's % of total revenue)
    total_revenue = sum(c.total_revenue for c in cohort_revenue_contributions)
    newest_revenue = sum(
        c.total_revenue for c in cohort_revenue_contributions
        if c.cohort_id == newest_cohort_id
    )
    if total_revenue > 0:
        acquisition_dependence_pct = (
            newest_revenue / total_revenue * 100
        ).quantize(PERCENTAGE_PRECISION, rounding=ROUND_HALF_UP)
    else:
        acquisition_dependence_pct = Decimal("0.00")
else:
    revenue_predictability_pct = Decimal("0.00")
    acquisition_dependence_pct = Decimal("0.00")

# Calculate health score and grade
health_score, health_grade = calculate_health_score(
    overall_retention=overall_retention_rate,
    cohort_quality_trend=cohort_quality_trend,
    revenue_predictability=revenue_predictability_pct,
    acquisition_dependence=acquisition_dependence_pct,
)

health_score_obj = CustomerBaseHealthScore(
    total_customers=total_customers,
    total_active_customers=active_customers_in_window,
    overall_retention_rate=overall_retention_rate,
    cohort_quality_trend=cohort_quality_trend,
    revenue_predictability_pct=revenue_predictability_pct,
    acquisition_dependence_pct=acquisition_dependence_pct,
    health_score=health_score,
    health_grade=health_grade,
)
```

#### 4. Add Comprehensive Integration Tests

**File**: `tests/test_lens4.py` and `tests/test_lens5.py`

**Changes**: Add integration tests combining both lenses

```python
# In tests/test_lens4.py:

class TestLens4Integration:
    """Integration tests for Lens 4 with realistic scenarios."""

    def test_multiple_cohorts_multiple_periods(self):
        """Test with 3 cohorts over 5 periods each."""
        # Setup realistic test data with Texas CLV synthetic generator
        # ...
        lens4 = compare_cohorts(periods, cohort_assignments)

        # Verify decompositions
        assert len(lens4.cohort_decompositions) > 0

        # Verify time to second purchase
        assert len(lens4.time_to_second_purchase) == 3  # 3 cohorts

        # Verify cohort comparisons
        assert len(lens4.cohort_comparisons) > 0

    @pytest.mark.slow
    def test_performance_10k_customers(self):
        """Verify acceptable performance with 10k customers."""
        import time

        # Generate 10k customers with Texas CLV synthetic generator
        # ...

        start = time.time()
        lens4 = compare_cohorts(periods, cohort_assignments)
        duration = time.time() - start

        assert duration < 5.0, f"Performance test took {duration:.2f}s (target: <5s)"


# In tests/test_lens5.py:

class TestLens5Integration:
    """Integration tests for Lens 5 with realistic scenarios."""

    def test_complete_health_assessment(self):
        """Test complete health assessment with realistic data."""
        # Setup realistic test data
        # ...
        lens5 = assess_customer_base_health(
            periods, cohort_assignments,
            datetime(2023, 1, 1, tzinfo=timezone.utc),
            datetime(2023, 12, 31, tzinfo=timezone.utc),
        )

        # Verify C3 data
        assert len(lens5.cohort_revenue_contributions) > 0

        # Verify repeat behavior
        assert len(lens5.cohort_repeat_behavior) > 0

        # Verify health score
        assert Decimal("0") <= lens5.health_score.health_score <= Decimal("100")
        assert lens5.health_score.health_grade in ("A", "B", "C", "D", "F")

    def test_lens4_and_lens5_together(self):
        """Test using Lens 4 and Lens 5 together for comprehensive analysis."""
        # Generate test data
        # ...

        # Run both lenses
        lens4 = compare_cohorts(periods, cohort_assignments)
        lens5 = assess_customer_base_health(
            periods, cohort_assignments,
            analysis_start_date, analysis_end_date,
        )

        # Verify they work together
        # Should have same cohorts
        lens4_cohorts = set(d.cohort_id for d in lens4.cohort_decompositions)
        lens5_cohorts = set(r.cohort_id for r in lens5.cohort_repeat_behavior)
        assert lens4_cohorts == lens5_cohorts
```

#### 5. Update AGENTS.md

**File**: `AGENTS.md`

**Changes**: Update Track B status to mark Phase 5 complete

```markdown
**Status**: ✅ PHASE 1-5 COMPLETE
- ✅ Cohorts implementation complete (Phase 1)
- ✅ Lens 3 implementation complete (Phase 2)
- ✅ Model preparation complete (Phase 3)
- ✅ BG/NBD model complete (Phase 3, PR #69)
- ✅ Gamma-Gamma model complete (Phase 3, PR #65)
- ✅ CLV calculator complete (Phase 3, PR #71)
- ✅ Model diagnostics complete (Phase 4, PR #74)
- ✅ Validation framework complete (Phase 4, PR #77)
- ✅ Lens 4: Multi-Cohort Comparison (Phase 5, Issue #34)
- ✅ Lens 5: Overall Customer Base Health (Phase 5, Issue #35)
```

### Success Criteria

#### Automated Verification:
- [ ] All Lens 4 tests pass: `pytest tests/test_lens4.py -v`
- [ ] All Lens 5 tests pass: `pytest tests/test_lens5.py -v`
- [ ] Integration tests pass: `pytest tests/test_lens4.py::TestLens4Integration tests/test_lens5.py::TestLens5Integration -v`
- [ ] Coverage > 90%: `pytest --cov=customer_base_audit/analyses/lens4 --cov=customer_base_audit/analyses/lens5 --cov-report=term-missing`
- [ ] Type checking passes: `mypy customer_base_audit/analyses/lens4.py customer_base_audit/analyses/lens5.py`
- [ ] Linting passes: `ruff check customer_base_audit/analyses/`
- [ ] Performance tests pass: `pytest -m slow tests/test_lens4.py tests/test_lens5.py`

#### Manual Verification:
- [ ] Health score calculation produces reasonable values for test scenarios
- [ ] Cohort quality trend detection works correctly (improving/stable/declining)
- [ ] Lens 4 and Lens 5 work together seamlessly
- [ ] Documentation is complete with working doctests
- [ ] Example usage is clear in docstrings

**Implementation Note**: After completing Phase 4 and all verification passes, Issues #34 and #35 can be closed.

---

## Testing Strategy

### Unit Testing

**Dataclass Validation**:
- Test each field validation (negative values, out of range, etc.)
- Test cross-field validation (e.g., active_customers <= cohort_size)
- Test immutability (attempt to modify frozen fields)
- Test boundary conditions (0, 100, edge cases)

**Helper Functions**:
- Test each helper with simple inputs
- Test edge cases (empty data, zero values, single item)
- Test precision/rounding correctness
- Test error handling (invalid inputs, missing data)

**Main Functions**:
- Test empty input handling
- Test single cohort / single period
- Test multiple cohorts / multiple periods
- Test both alignment modes (left-aligned, time-aligned)
- Test with/without margin data

### Integration Testing

**Cross-Lens Integration**:
- Test Lens 4 and Lens 5 together with same data
- Verify consistent cohort identification
- Verify no data inconsistencies

**Synthetic Data Testing**:
- Use Texas CLV generator for realistic scenarios
- Test with 100, 1000, 10000 customers
- Verify performance is acceptable

**Edge Case Scenarios**:
- All customers churn (100% churn)
- No customers repeat (100% one-time buyers)
- Single cohort only
- Many cohorts (10+ cohorts)
- Sparse data (few transactions per customer)

### Performance Testing

**Benchmarks**:
- 100 customers: < 0.1s
- 1,000 customers: < 0.5s
- 10,000 customers: < 5s

**Test with `@pytest.mark.slow`**:
- Mark performance tests as slow
- Run separately in CI/CD: `pytest -m slow`

### Test Organization

```
tests/
├── test_lens4.py
│   ├── TestCohortDecomposition (dataclass validation)
│   ├── TestTimeToSecondPurchase (dataclass validation)
│   ├── TestCohortComparison (dataclass validation)
│   ├── TestLens4Metrics (dataclass validation)
│   ├── TestCalculateCohortDecomposition (helper tests)
│   ├── TestCalculateTimeToSecondPurchase (helper tests)
│   ├── TestCompareCohortPair (helper tests)
│   ├── TestCompareCohorts (main function tests)
│   └── TestLens4Integration (integration tests)
│
└── test_lens5.py
    ├── TestCohortRevenuePeriod (dataclass validation)
    ├── TestCohortRepeatBehavior (dataclass validation)
    ├── TestCustomerBaseHealthScore (dataclass validation)
    ├── TestLens5Metrics (dataclass validation)
    ├── TestCalculateHealthScore (helper tests)
    ├── TestDetermineCohortQualityTrend (helper tests)
    ├── TestCalculateRevenuePredictability (helper tests)
    ├── TestAssessCustomerBaseHealth (main function tests)
    └── TestLens5Integration (integration tests)
```

---

## Performance Considerations

### Memory Efficiency

**Use generator expressions** for aggregation:
```python
total_orders = sum(p.total_orders for p in period_data)  # Good
total_orders = sum([p.total_orders for p in period_data])  # Bad (creates list)
```

**Group data efficiently**:
- Use `dict` for O(1) lookups instead of repeated list scans
- Sort once, then iterate (don't sort repeatedly)

### Time Complexity

**Target Complexity**:
- `O(n * m)` where n = customers, m = periods
- Acceptable for n=10,000, m=100 (1M operations)

**Optimization Strategies**:
- Pre-group period_aggregations by customer_id once
- Cache cohort sizes (don't recalculate)
- Avoid nested loops over large datasets

### Decimal Performance

**Quantize consistently**:
- Define `PERCENTAGE_PRECISION = Decimal("0.01")` once
- Reuse for all percentage calculations

**Convert float→Decimal once**:
```python
# Good: convert once
total_revenue = Decimal(str(period.total_spend))

# Bad: convert in every calculation
result = Decimal(str(period.total_spend)) / Decimal(str(count))
```

---

## Migration Notes

**No migration required** - these are new modules with no breaking changes to existing code.

**Integration Points**:
1. Lens 4 and 5 consume same inputs as Lens 3 (PeriodAggregation, cohort assignments)
2. Can be used independently or together
3. No changes to existing Lens 1-3 modules

---

## References

### Research
- Research document: `thoughts/shared/research/2025-10-10-lens4-lens5-implementation-research.md`
- Book: "The Customer-Base Audit" by Fader, Hardie, and Ross (Chapters 6-7)

### Issues
- Issue #34: Lens 4 Multi-Cohort Comparison (Track B)
- Issue #35: Lens 5 Overall Customer Base Health (Track B, depends on #34)

### Related Code
- Foundation: `customer_base_audit/foundation/cohorts.py` (CohortDefinition, assign_cohorts)
- Foundation: `customer_base_audit/foundation/data_mart.py` (PeriodAggregation)
- Examples: `customer_base_audit/analyses/lens1.py` (architectural patterns)
- Examples: `customer_base_audit/analyses/lens2.py` (Decimal usage, validation)
- Tests: `tests/test_lens1.py`, `tests/test_lens2.py`, `tests/test_lens3.py`

### Track B Documentation
- Track B status: `AGENTS.md` (lines 52-92)
- Track B architecture: `thoughts/shared/research/2025-10-10-track-b-lens-architecture.md`

---

**End of Implementation Plan**
