# Pandas Integration Adapter Layer for Track A Components - Implementation Plan

## Overview

Implement pandas DataFrame adapters for Track A components (RFM, Lens 1, Lens 2) to reduce friction in pandas-based workflows while preserving the existing dataclass-based core API. This is an additive feature that provides convenience functions for users who prefer DataFrame interfaces for data science workflows, BI tool integration, and Jupyter notebooks.

## Current State Analysis

Track A components use frozen dataclasses with comprehensive validation:
- `RFMMetrics` - Recency, Frequency, Monetary customer metrics
- `Lens1Metrics` - Single period customer base analysis
- `Lens2Metrics` - Period-to-period comparison analysis
- `PeriodAggregation` - Input data structure for RFM calculation

**Current Pain Point**: Users must manually convert between dataclasses and DataFrames:
```python
# Current workaround (manual conversion)
rfm_dict_list = [asdict(m) for m in rfm_metrics]
rfm_df = pd.DataFrame(rfm_dict_list)
```

**Existing Patterns**:
- Track B models (`model_prep.py`) use manual dict-to-DataFrame conversions
- Pattern: `rows = [dict] → pd.DataFrame(rows) → sort → return`
- No centralized conversion utilities exist
- Pandas v2.1.0+ already a core dependency

### Key Discoveries:
- `customer_base_audit/models/model_prep.py:188-217` - BG/NBD conversion pattern
- `customer_base_audit/models/model_prep.py:287-311` - Gamma-Gamma conversion pattern
- `customer_base_audit/foundation/rfm.py:337-344` - RFMScore uses pandas internally
- `tests/test_rfm.py`, `tests/test_lens1.py`, `tests/test_lens2.py` - Testing patterns to follow

## Desired End State

### User Experience
Users can seamlessly work with DataFrames:
```python
from customer_base_audit.pandas import (
    calculate_rfm_df,
    analyze_single_period_df,
    analyze_period_comparison_df,
    rfm_to_dataframe,
    dataframe_to_rfm
)

# One-line DataFrame workflow
rfm_df = calculate_rfm_df(periods_df, observation_end=datetime(2023, 12, 31))
lens1_df = analyze_single_period_df(rfm_df)

# Export to BI tools
rfm_df.to_csv('rfm_scores.csv')
lens1_df.to_csv('lens1_metrics.csv')
```

### Module Structure
```
customer_base_audit/pandas/
├── __init__.py          # Public API exports
├── rfm.py              # RFM DataFrame adapters
├── lens1.py            # Lens 1 DataFrame adapters
├── lens2.py            # Lens 2 DataFrame adapters
└── _utils.py           # Shared conversion utilities
```

### Verification
- All existing dataclass APIs unchanged (100% backward compatible)
- New pandas adapters pass all tests
- Documentation includes DataFrame workflow examples
- Performance overhead < 5% vs direct dataclass operations

## What We're NOT Doing

- ❌ **Replacing dataclass API** - Core API stays as-is for type safety
- ❌ **RFMScore adapters** - Already uses pandas internally, not in Issue #78 scope
- ❌ **Sklearn compatibility** - Not appropriate for descriptive analytics (Track A)
- ❌ **NumPy array support** - Loses semantic information (customer IDs, column names)
- ❌ **Spark/Dask native support** - Use pandas UDFs instead (future work)
- ❌ **Modifying Track B models** - Separate codebase, different ownership

## Implementation Approach

Follow proven patterns from `model_prep.py`:
1. **Export (dataclass → DataFrame)**: Build list of dicts, create DataFrame, sort by customer_id
2. **Import (DataFrame → dataclass)**: Iterate rows, validate, construct dataclass instances
3. **Type conversions**: `Decimal(str(float))` for import, `float(Decimal)` for export
4. **Datetime handling**: pandas datetime64 ↔ Python datetime via `pd.to_datetime().dt.to_pydatetime()`
5. **Testing**: Follow Track A patterns (class-based, helper methods, round-trip tests)

---

## Phase 1: Core Utilities and RFM Adapters

### Overview
Create foundational conversion utilities and implement RFM DataFrame adapters. This establishes the pattern for Lens adapters in subsequent phases.

### Changes Required:

#### 1. Shared Utilities Module
**File**: `customer_base_audit/pandas/_utils.py` (NEW)

```python
"""Shared utilities for pandas conversion operations."""
from decimal import Decimal
from datetime import datetime
import pandas as pd


def decimal_to_float(value: Decimal) -> float:
    """Convert Decimal to float for pandas compatibility."""
    return float(value)


def float_to_decimal(value: float) -> Decimal:
    """Convert float to Decimal, avoiding precision issues."""
    return Decimal(str(value))


def datetime_to_pandas(dt: datetime) -> pd.Timestamp:
    """Convert Python datetime to pandas Timestamp."""
    return pd.Timestamp(dt)


def pandas_to_datetime(timestamp: pd.Timestamp) -> datetime:
    """Convert pandas Timestamp to Python datetime."""
    return timestamp.to_pydatetime()
```

#### 2. RFM DataFrame Adapters
**File**: `customer_base_audit/pandas/rfm.py` (NEW)

```python
"""Pandas DataFrame adapters for RFM calculations."""
from typing import List, Sequence
from datetime import datetime
import pandas as pd

from customer_base_audit.foundation.rfm import RFMMetrics, calculate_rfm
from customer_base_audit.foundation.data_mart import PeriodAggregation
from ._utils import decimal_to_float, float_to_decimal


def rfm_to_dataframe(rfm_metrics: Sequence[RFMMetrics]) -> pd.DataFrame:
    """Convert RFM metrics to pandas DataFrame.

    Args:
        rfm_metrics: Sequence of RFMMetrics objects

    Returns:
        DataFrame with columns: customer_id, recency_days, frequency,
        monetary, total_spend, observation_start, observation_end

    Example:
        >>> rfm_metrics = calculate_rfm(periods, datetime(2023, 12, 31))
        >>> rfm_df = rfm_to_dataframe(rfm_metrics)
        >>> rfm_df.head()
    """
    if not rfm_metrics:
        return pd.DataFrame(columns=[
            'customer_id', 'recency_days', 'frequency', 'monetary',
            'total_spend', 'observation_start', 'observation_end'
        ])

    rows = [
        {
            'customer_id': m.customer_id,
            'recency_days': m.recency_days,
            'frequency': m.frequency,
            'monetary': decimal_to_float(m.monetary),
            'total_spend': decimal_to_float(m.total_spend),
            'observation_start': m.observation_start,
            'observation_end': m.observation_end,
        }
        for m in rfm_metrics
    ]

    df = pd.DataFrame(rows)
    df = df.sort_values('customer_id').reset_index(drop=True)
    return df


def dataframe_to_rfm(rfm_df: pd.DataFrame) -> List[RFMMetrics]:
    """Convert pandas DataFrame to RFM metrics.

    Args:
        rfm_df: DataFrame with RFM columns

    Returns:
        List of validated RFMMetrics objects

    Raises:
        ValueError: If DataFrame missing required columns or has invalid data

    Example:
        >>> rfm_metrics = dataframe_to_rfm(rfm_df)
        >>> lens1 = analyze_single_period(rfm_metrics)
    """
    required_cols = [
        'customer_id', 'recency_days', 'frequency', 'monetary',
        'total_spend', 'observation_start', 'observation_end'
    ]

    missing_cols = set(required_cols) - set(rfm_df.columns)
    if missing_cols:
        raise ValueError(f"DataFrame missing required columns: {missing_cols}")

    if rfm_df.empty:
        return []

    rfm_metrics = []
    for _, row in rfm_df.iterrows():
        rfm_metrics.append(
            RFMMetrics(
                customer_id=str(row['customer_id']),
                recency_days=int(row['recency_days']),
                frequency=int(row['frequency']),
                monetary=float_to_decimal(row['monetary']),
                observation_start=pd.to_datetime(row['observation_start']).to_pydatetime(),
                observation_end=pd.to_datetime(row['observation_end']).to_pydatetime(),
                total_spend=float_to_decimal(row['total_spend']),
            )
        )

    return rfm_metrics


def dataframe_to_period_aggregations(
    periods_df: pd.DataFrame,
    customer_id_col: str = 'customer_id',
    period_start_col: str = 'period_start',
    period_end_col: str = 'period_end',
    total_orders_col: str = 'total_orders',
    total_spend_col: str = 'total_spend',
    total_margin_col: str = 'total_margin',
    total_quantity_col: str = 'total_quantity',
) -> List[PeriodAggregation]:
    """Convert pandas DataFrame to PeriodAggregation list.

    Args:
        periods_df: DataFrame with period aggregation data
        *_col: Column name mappings for flexibility

    Returns:
        List of PeriodAggregation objects

    Raises:
        ValueError: If DataFrame missing required columns

    Example:
        >>> periods_df = pd.read_csv('customer_periods.csv')
        >>> periods = dataframe_to_period_aggregations(periods_df)
        >>> rfm = calculate_rfm(periods, datetime(2023, 12, 31))
    """
    required_mapping = {
        'customer_id': customer_id_col,
        'period_start': period_start_col,
        'period_end': period_end_col,
        'total_orders': total_orders_col,
        'total_spend': total_spend_col,
        'total_margin': total_margin_col,
        'total_quantity': total_quantity_col,
    }

    missing_cols = set(required_mapping.values()) - set(periods_df.columns)
    if missing_cols:
        raise ValueError(f"DataFrame missing required columns: {missing_cols}")

    if periods_df.empty:
        return []

    periods = []
    for _, row in periods_df.iterrows():
        periods.append(
            PeriodAggregation(
                customer_id=str(row[customer_id_col]),
                period_start=pd.to_datetime(row[period_start_col]).to_pydatetime(),
                period_end=pd.to_datetime(row[period_end_col]).to_pydatetime(),
                total_orders=int(row[total_orders_col]),
                total_spend=float(row[total_spend_col]),
                total_margin=float(row[total_margin_col]),
                total_quantity=int(row[total_quantity_col]),
            )
        )

    return periods


def calculate_rfm_df(
    periods_df: pd.DataFrame,
    observation_end: datetime,
    customer_id_col: str = 'customer_id',
    period_start_col: str = 'period_start',
    period_end_col: str = 'period_end',
    total_orders_col: str = 'total_orders',
    total_spend_col: str = 'total_spend',
    total_margin_col: str = 'total_margin',
    total_quantity_col: str = 'total_quantity',
) -> pd.DataFrame:
    """Calculate RFM metrics from a pandas DataFrame.

    Convenience function that combines conversion and calculation.

    Args:
        periods_df: DataFrame with period aggregations
        observation_end: End date for recency calculation
        *_col: Column name mappings for flexibility

    Returns:
        DataFrame with RFM metrics

    Example:
        >>> periods_df = pd.read_parquet('periods.parquet')
        >>> rfm_df = calculate_rfm_df(periods_df, datetime(2023, 12, 31))
        >>> high_value = rfm_df[rfm_df['monetary'] > 100]
    """
    # Convert DataFrame → List[PeriodAggregation]
    periods = dataframe_to_period_aggregations(
        periods_df,
        customer_id_col=customer_id_col,
        period_start_col=period_start_col,
        period_end_col=period_end_col,
        total_orders_col=total_orders_col,
        total_spend_col=total_spend_col,
        total_margin_col=total_margin_col,
        total_quantity_col=total_quantity_col,
    )

    # Calculate RFM using core API
    rfm_metrics = calculate_rfm(periods, observation_end)

    # Convert List[RFMMetrics] → DataFrame
    return rfm_to_dataframe(rfm_metrics)
```

#### 3. Public API Module
**File**: `customer_base_audit/pandas/__init__.py` (NEW)

```python
"""Pandas DataFrame adapters for customer base audit components."""

from .rfm import (
    rfm_to_dataframe,
    dataframe_to_rfm,
    dataframe_to_period_aggregations,
    calculate_rfm_df,
)

__all__ = [
    'rfm_to_dataframe',
    'dataframe_to_rfm',
    'dataframe_to_period_aggregations',
    'calculate_rfm_df',
]
```

#### 4. Test Suite
**File**: `tests/test_pandas_rfm.py` (NEW)

```python
"""Tests for RFM pandas adapters."""
import pytest
from datetime import datetime
from decimal import Decimal
import pandas as pd

from customer_base_audit.foundation.rfm import RFMMetrics, calculate_rfm
from customer_base_audit.foundation.data_mart import PeriodAggregation
from customer_base_audit.pandas import (
    rfm_to_dataframe,
    dataframe_to_rfm,
    dataframe_to_period_aggregations,
    calculate_rfm_df,
)


class TestRFMToDataFrame:
    """Test rfm_to_dataframe conversion."""

    def test_single_rfm_metric(self):
        """Single RFM metric converts to DataFrame correctly."""
        metrics = [
            RFMMetrics(
                customer_id="C1",
                recency_days=10,
                frequency=5,
                monetary=Decimal("50.00"),
                observation_start=datetime(2023, 1, 1),
                observation_end=datetime(2023, 12, 31),
                total_spend=Decimal("250.00"),
            )
        ]

        df = rfm_to_dataframe(metrics)

        assert len(df) == 1
        assert df.iloc[0]['customer_id'] == "C1"
        assert df.iloc[0]['recency_days'] == 10
        assert df.iloc[0]['frequency'] == 5
        assert df.iloc[0]['monetary'] == 50.0  # Decimal converted to float
        assert df.iloc[0]['total_spend'] == 250.0

    def test_empty_input_returns_empty_dataframe(self):
        """Empty metrics list returns empty DataFrame with correct columns."""
        df = rfm_to_dataframe([])

        assert df.empty
        assert list(df.columns) == [
            'customer_id', 'recency_days', 'frequency', 'monetary',
            'total_spend', 'observation_start', 'observation_end'
        ]

    def test_sorted_by_customer_id(self):
        """Output DataFrame is sorted by customer_id."""
        metrics = [
            RFMMetrics("C3", 10, 5, Decimal("50"), datetime(2023,1,1), datetime(2023,12,31), Decimal("250")),
            RFMMetrics("C1", 20, 3, Decimal("40"), datetime(2023,1,1), datetime(2023,12,31), Decimal("120")),
            RFMMetrics("C2", 15, 4, Decimal("45"), datetime(2023,1,1), datetime(2023,12,31), Decimal("180")),
        ]

        df = rfm_to_dataframe(metrics)

        assert list(df['customer_id']) == ["C1", "C2", "C3"]


class TestDataFrameToRFM:
    """Test dataframe_to_rfm conversion."""

    def test_round_trip_conversion(self):
        """Round-trip conversion preserves data exactly."""
        original = [
            RFMMetrics(
                "C1", 10, 5, Decimal("50.00"),
                datetime(2023, 1, 1), datetime(2023, 12, 31),
                Decimal("250.00")
            )
        ]

        df = rfm_to_dataframe(original)
        restored = dataframe_to_rfm(df)

        assert len(restored) == 1
        assert restored[0].customer_id == original[0].customer_id
        assert restored[0].recency_days == original[0].recency_days
        assert restored[0].monetary == original[0].monetary

    def test_missing_columns_raises_error(self):
        """DataFrame missing required columns raises ValueError."""
        df = pd.DataFrame({
            'customer_id': ['C1'],
            'recency_days': [10],
            # Missing other columns
        })

        with pytest.raises(ValueError, match="missing required columns"):
            dataframe_to_rfm(df)

    def test_empty_dataframe_returns_empty_list(self):
        """Empty DataFrame returns empty list."""
        df = pd.DataFrame(columns=[
            'customer_id', 'recency_days', 'frequency', 'monetary',
            'total_spend', 'observation_start', 'observation_end'
        ])

        result = dataframe_to_rfm(df)

        assert result == []


class TestCalculateRFMDF:
    """Test calculate_rfm_df convenience function."""

    def test_end_to_end_dataframe_workflow(self):
        """End-to-end DataFrame workflow produces correct results."""
        periods_df = pd.DataFrame({
            'customer_id': ['C1', 'C1', 'C2'],
            'period_start': [
                datetime(2023, 1, 1),
                datetime(2023, 2, 1),
                datetime(2023, 1, 1),
            ],
            'period_end': [
                datetime(2023, 2, 1),
                datetime(2023, 3, 1),
                datetime(2023, 2, 1),
            ],
            'total_orders': [2, 1, 3],
            'total_spend': [100.0, 50.0, 200.0],
            'total_margin': [30.0, 15.0, 60.0],
            'total_quantity': [5, 2, 8],
        })

        rfm_df = calculate_rfm_df(periods_df, datetime(2023, 4, 15))

        assert len(rfm_df) == 2  # C1 and C2
        assert 'customer_id' in rfm_df.columns
        assert 'frequency' in rfm_df.columns
        assert 'monetary' in rfm_df.columns

    @pytest.mark.slow
    def test_performance_10k_customers(self):
        """Performance test with 10k customers."""
        import time

        # Create 10k customer periods
        periods_df = pd.DataFrame({
            'customer_id': [f"C{i}" for i in range(10000)],
            'period_start': [datetime(2023, 1, 1)] * 10000,
            'period_end': [datetime(2023, 2, 1)] * 10000,
            'total_orders': [5] * 10000,
            'total_spend': [250.0] * 10000,
            'total_margin': [75.0] * 10000,
            'total_quantity': [10] * 10000,
        })

        start = time.time()
        rfm_df = calculate_rfm_df(periods_df, datetime(2023, 12, 31))
        duration = time.time() - start

        assert duration < 2.0, f"Took {duration:.2f}s (expected < 2.0s)"
        assert len(rfm_df) == 10000
```

### Success Criteria:

#### Automated Verification:
- [ ] All unit tests pass: `make test`
- [ ] Type checking passes: `mypy customer_base_audit/pandas/`
- [ ] Linting passes: `make lint`
- [ ] Test coverage for pandas module >= 90%: `pytest --cov=customer_base_audit/pandas tests/test_pandas_rfm.py`
- [ ] Performance test passes (10k customers < 2s): `pytest -m slow tests/test_pandas_rfm.py`

#### Manual Verification:
- [ ] Round-trip conversion preserves Decimal precision (manually verify test output)
- [ ] DataFrame column order matches documentation
- [ ] Error messages are clear and helpful for invalid DataFrames
- [ ] Import works from top-level: `from customer_base_audit.pandas import calculate_rfm_df`

**Implementation Note**: After completing this phase and all automated verification passes, pause here for manual confirmation that the manual testing was successful before proceeding to Phase 2.

---

## Phase 2: Lens 1 DataFrame Adapters

### Overview
Implement Lens 1 analysis DataFrame adapters, building on the RFM conversion utilities from Phase 1.

### Changes Required:

#### 1. Lens 1 Adapters Module
**File**: `customer_base_audit/pandas/lens1.py` (NEW)

```python
"""Pandas DataFrame adapters for Lens 1 analysis."""
from typing import Sequence, Optional
import pandas as pd

from customer_base_audit.analyses.lens1 import Lens1Metrics, analyze_single_period
from customer_base_audit.foundation.rfm import RFMMetrics, RFMScore
from .rfm import rfm_to_dataframe, dataframe_to_rfm
from ._utils import decimal_to_float


def lens1_to_dataframe(lens1: Lens1Metrics) -> pd.DataFrame:
    """Convert Lens1Metrics to single-row DataFrame.

    Args:
        lens1: Lens1Metrics object

    Returns:
        Single-row DataFrame with Lens 1 metrics

    Example:
        >>> lens1 = analyze_single_period(rfm_metrics)
        >>> lens1_df = lens1_to_dataframe(lens1)
        >>> print(lens1_df['one_time_buyer_pct'].iloc[0])
    """
    return pd.DataFrame([{
        'total_customers': lens1.total_customers,
        'one_time_buyers': lens1.one_time_buyers,
        'one_time_buyer_pct': decimal_to_float(lens1.one_time_buyer_pct),
        'total_revenue': decimal_to_float(lens1.total_revenue),
        'top_10pct_revenue_contribution': decimal_to_float(lens1.top_10pct_revenue_contribution),
        'top_20pct_revenue_contribution': decimal_to_float(lens1.top_20pct_revenue_contribution),
        'avg_orders_per_customer': decimal_to_float(lens1.avg_orders_per_customer),
        'median_customer_value': decimal_to_float(lens1.median_customer_value),
        'rfm_distribution': str(lens1.rfm_distribution),  # Dict as string for CSV export
    }])


def analyze_single_period_df(
    rfm_df: pd.DataFrame,
    rfm_scores_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Perform Lens 1 analysis on DataFrame.

    Convenience function combining conversion and analysis.

    Args:
        rfm_df: DataFrame with RFM metrics
        rfm_scores_df: Optional DataFrame with RFM scores

    Returns:
        Single-row DataFrame with Lens 1 metrics

    Example:
        >>> rfm_df = calculate_rfm_df(periods_df, datetime(2023, 12, 31))
        >>> lens1_df = analyze_single_period_df(rfm_df)
        >>> lens1_df.to_csv('lens1_metrics.csv', index=False)
    """
    # Convert DataFrame → List[RFMMetrics]
    rfm_metrics = dataframe_to_rfm(rfm_df)

    # Convert RFM scores if provided
    rfm_scores = None
    if rfm_scores_df is not None:
        # Note: RFMScore conversion not in scope for Issue #78
        # For now, rfm_scores must be None or passed as list
        raise NotImplementedError("RFMScore DataFrame conversion not yet supported")

    # Use core API
    lens1 = analyze_single_period(rfm_metrics, rfm_scores=rfm_scores)

    # Convert Lens1Metrics → DataFrame
    return lens1_to_dataframe(lens1)
```

#### 2. Update Public API
**File**: `customer_base_audit/pandas/__init__.py`

```python
"""Pandas DataFrame adapters for customer base audit components."""

from .rfm import (
    rfm_to_dataframe,
    dataframe_to_rfm,
    dataframe_to_period_aggregations,
    calculate_rfm_df,
)
from .lens1 import (
    lens1_to_dataframe,
    analyze_single_period_df,
)

__all__ = [
    # RFM adapters
    'rfm_to_dataframe',
    'dataframe_to_rfm',
    'dataframe_to_period_aggregations',
    'calculate_rfm_df',
    # Lens 1 adapters
    'lens1_to_dataframe',
    'analyze_single_period_df',
]
```

#### 3. Test Suite
**File**: `tests/test_pandas_lens1.py` (NEW)

```python
"""Tests for Lens 1 pandas adapters."""
import pytest
from datetime import datetime
from decimal import Decimal
import pandas as pd

from customer_base_audit.foundation.rfm import RFMMetrics
from customer_base_audit.analyses.lens1 import analyze_single_period
from customer_base_audit.pandas import (
    lens1_to_dataframe,
    analyze_single_period_df,
    rfm_to_dataframe,
)


class TestLens1ToDataFrame:
    """Test lens1_to_dataframe conversion."""

    def test_converts_all_fields(self):
        """All Lens1Metrics fields are converted to DataFrame columns."""
        metrics = [
            RFMMetrics("C1", 10, 5, Decimal("50.00"), datetime(2023,1,1), datetime(2023,12,31), Decimal("250.00")),
            RFMMetrics("C2", 20, 3, Decimal("40.00"), datetime(2023,1,1), datetime(2023,12,31), Decimal("120.00")),
        ]
        lens1 = analyze_single_period(metrics)

        df = lens1_to_dataframe(lens1)

        assert len(df) == 1  # Single row
        assert df.iloc[0]['total_customers'] == 2
        assert df.iloc[0]['one_time_buyers'] == 0
        assert 'one_time_buyer_pct' in df.columns
        assert 'total_revenue' in df.columns
        assert 'top_10pct_revenue_contribution' in df.columns


class TestAnalyzeSinglePeriodDF:
    """Test analyze_single_period_df convenience function."""

    def test_end_to_end_lens1_workflow(self):
        """End-to-end Lens 1 DataFrame workflow."""
        metrics = [
            RFMMetrics("C1", 10, 1, Decimal("100.00"), datetime(2023,1,1), datetime(2023,12,31), Decimal("100.00")),
            RFMMetrics("C2", 20, 5, Decimal("50.00"), datetime(2023,1,1), datetime(2023,12,31), Decimal("250.00")),
        ]
        rfm_df = rfm_to_dataframe(metrics)

        lens1_df = analyze_single_period_df(rfm_df)

        assert len(lens1_df) == 1
        assert lens1_df.iloc[0]['total_customers'] == 2
        assert lens1_df.iloc[0]['one_time_buyers'] == 1  # C1 has frequency=1

    def test_empty_dataframe_returns_zero_metrics(self):
        """Empty RFM DataFrame returns zero-valued metrics."""
        rfm_df = pd.DataFrame(columns=[
            'customer_id', 'recency_days', 'frequency', 'monetary',
            'total_spend', 'observation_start', 'observation_end'
        ])

        lens1_df = analyze_single_period_df(rfm_df)

        assert len(lens1_df) == 1
        assert lens1_df.iloc[0]['total_customers'] == 0
        assert lens1_df.iloc[0]['total_revenue'] == 0.0
```

### Success Criteria:

#### Automated Verification:
- [ ] All unit tests pass: `make test`
- [ ] Type checking passes: `mypy customer_base_audit/pandas/lens1.py`
- [ ] Linting passes: `make lint`
- [ ] Test coverage for lens1 module >= 90%: `pytest --cov=customer_base_audit/pandas/lens1 tests/test_pandas_lens1.py`

#### Manual Verification:
- [ ] Lens 1 DataFrame exports cleanly to CSV
- [ ] All Decimal fields correctly converted to float
- [ ] rfm_distribution dict representation is readable
- [ ] Error handling provides clear messages

**Implementation Note**: After completing this phase and all automated verification passes, pause here for manual confirmation that the manual testing was successful before proceeding to Phase 3.

---

## Phase 3: Lens 2 DataFrame Adapters with Nested Structures

### Overview
Implement Lens 2 period comparison adapters with special handling for nested structures (CustomerMigration, Lens1Metrics). Returns a dictionary of DataFrames for maximum flexibility.

### Changes Required:

#### 1. Lens 2 Adapters Module
**File**: `customer_base_audit/pandas/lens2.py` (NEW)

```python
"""Pandas DataFrame adapters for Lens 2 analysis."""
from typing import Dict
import pandas as pd

from customer_base_audit.analyses.lens2 import Lens2Metrics, analyze_period_comparison
from customer_base_audit.foundation.rfm import RFMMetrics
from .rfm import dataframe_to_rfm
from .lens1 import lens1_to_dataframe
from ._utils import decimal_to_float


def lens2_to_dataframes(lens2: Lens2Metrics) -> Dict[str, pd.DataFrame]:
    """Convert Lens2Metrics to multiple DataFrames.

    Args:
        lens2: Lens2Metrics object with nested structures

    Returns:
        Dictionary with keys:
        - 'metrics': Single-row DataFrame with Lens 2 scalar metrics
        - 'migration': Multi-row DataFrame with customer migration (customer_id, status)
        - 'period1_summary': Single-row DataFrame with period 1 Lens 1 metrics
        - 'period2_summary': Single-row DataFrame with period 2 Lens 1 metrics

    Example:
        >>> lens2 = analyze_period_comparison(period1_rfm, period2_rfm)
        >>> dfs = lens2_to_dataframes(lens2)
        >>> dfs['metrics'].to_csv('lens2_metrics.csv', index=False)
        >>> dfs['migration'].to_csv('customer_migration.csv', index=False)
    """
    # Scalar metrics (single row)
    metrics_df = pd.DataFrame([{
        'retention_rate': decimal_to_float(lens2.retention_rate),
        'churn_rate': decimal_to_float(lens2.churn_rate),
        'reactivation_rate': decimal_to_float(lens2.reactivation_rate),
        'customer_count_change': lens2.customer_count_change,
        'revenue_change_pct': decimal_to_float(lens2.revenue_change_pct),
        'avg_order_value_change_pct': decimal_to_float(lens2.avg_order_value_change_pct),
    }])

    # Customer migration (multi-row with status column)
    migration_rows = []
    for customer_id in lens2.migration.retained:
        migration_rows.append({'customer_id': customer_id, 'status': 'retained'})
    for customer_id in lens2.migration.churned:
        migration_rows.append({'customer_id': customer_id, 'status': 'churned'})
    for customer_id in lens2.migration.new:
        migration_rows.append({'customer_id': customer_id, 'status': 'new'})
    for customer_id in lens2.migration.reactivated:
        migration_rows.append({'customer_id': customer_id, 'status': 'reactivated'})

    migration_df = pd.DataFrame(migration_rows) if migration_rows else pd.DataFrame(columns=['customer_id', 'status'])
    migration_df = migration_df.sort_values('customer_id').reset_index(drop=True) if not migration_df.empty else migration_df

    # Period summaries (convert nested Lens1Metrics)
    period1_summary = lens1_to_dataframe(lens2.period1_metrics)
    period2_summary = lens1_to_dataframe(lens2.period2_metrics)

    return {
        'metrics': metrics_df,
        'migration': migration_df,
        'period1_summary': period1_summary,
        'period2_summary': period2_summary,
    }


def analyze_period_comparison_df(
    period1_rfm_df: pd.DataFrame,
    period2_rfm_df: pd.DataFrame,
) -> Dict[str, pd.DataFrame]:
    """Compare two periods using DataFrames.

    Convenience function combining conversion and analysis.

    Args:
        period1_rfm_df: DataFrame with period 1 RFM metrics
        period2_rfm_df: DataFrame with period 2 RFM metrics

    Returns:
        Dictionary of DataFrames (same as lens2_to_dataframes)

    Example:
        >>> q1_rfm = calculate_rfm_df(q1_periods, datetime(2023, 3, 31))
        >>> q2_rfm = calculate_rfm_df(q2_periods, datetime(2023, 6, 30))
        >>> comparison = analyze_period_comparison_df(q1_rfm, q2_rfm)
        >>> print(f"Retention: {comparison['metrics']['retention_rate'].iloc[0]}%")
        >>> churned = comparison['migration'][comparison['migration']['status'] == 'churned']
    """
    # Convert DataFrames → List[RFMMetrics]
    period1_rfm = dataframe_to_rfm(period1_rfm_df)
    period2_rfm = dataframe_to_rfm(period2_rfm_df)

    # Use core API
    lens2 = analyze_period_comparison(period1_rfm, period2_rfm)

    # Convert Lens2Metrics → Dict[str, DataFrame]
    return lens2_to_dataframes(lens2)
```

#### 2. Update Public API
**File**: `customer_base_audit/pandas/__init__.py`

```python
"""Pandas DataFrame adapters for customer base audit components."""

from .rfm import (
    rfm_to_dataframe,
    dataframe_to_rfm,
    dataframe_to_period_aggregations,
    calculate_rfm_df,
)
from .lens1 import (
    lens1_to_dataframe,
    analyze_single_period_df,
)
from .lens2 import (
    lens2_to_dataframes,
    analyze_period_comparison_df,
)

__all__ = [
    # RFM adapters
    'rfm_to_dataframe',
    'dataframe_to_rfm',
    'dataframe_to_period_aggregations',
    'calculate_rfm_df',
    # Lens 1 adapters
    'lens1_to_dataframe',
    'analyze_single_period_df',
    # Lens 2 adapters
    'lens2_to_dataframes',
    'analyze_period_comparison_df',
]
```

#### 3. Test Suite
**File**: `tests/test_pandas_lens2.py` (NEW)

```python
"""Tests for Lens 2 pandas adapters."""
import pytest
from datetime import datetime
from decimal import Decimal
import pandas as pd

from customer_base_audit.foundation.rfm import RFMMetrics
from customer_base_audit.analyses.lens2 import analyze_period_comparison
from customer_base_audit.pandas import (
    lens2_to_dataframes,
    analyze_period_comparison_df,
    rfm_to_dataframe,
)


class TestLens2ToDataFrames:
    """Test lens2_to_dataframes conversion."""

    def create_rfm(self, customer_id: str, frequency: int, spend: Decimal) -> RFMMetrics:
        """Helper to create RFMMetrics for testing."""
        return RFMMetrics(
            customer_id=customer_id,
            recency_days=10,
            frequency=frequency,
            monetary=spend / frequency,
            observation_start=datetime(2023, 1, 1),
            observation_end=datetime(2023, 12, 31),
            total_spend=spend,
        )

    def test_returns_four_dataframes(self):
        """Returns dict with 4 DataFrames: metrics, migration, period1_summary, period2_summary."""
        period1 = [self.create_rfm("C1", 5, Decimal("250"))]
        period2 = [self.create_rfm("C1", 3, Decimal("180"))]

        lens2 = analyze_period_comparison(period1, period2)
        dfs = lens2_to_dataframes(lens2)

        assert set(dfs.keys()) == {'metrics', 'migration', 'period1_summary', 'period2_summary'}
        assert isinstance(dfs['metrics'], pd.DataFrame)
        assert isinstance(dfs['migration'], pd.DataFrame)

    def test_migration_dataframe_structure(self):
        """Migration DataFrame has customer_id and status columns."""
        period1 = [
            self.create_rfm("C1", 5, Decimal("250")),
            self.create_rfm("C2", 3, Decimal("150")),
        ]
        period2 = [
            self.create_rfm("C1", 3, Decimal("180")),
            self.create_rfm("C3", 2, Decimal("100")),
        ]

        lens2 = analyze_period_comparison(period1, period2)
        dfs = lens2_to_dataframes(lens2)
        migration = dfs['migration']

        assert list(migration.columns) == ['customer_id', 'status']
        assert 'C1' in migration['customer_id'].values  # Retained
        assert 'C2' in migration['customer_id'].values  # Churned
        assert 'C3' in migration['customer_id'].values  # New

    def test_migration_status_values(self):
        """Migration status column contains correct values."""
        period1 = [self.create_rfm("C1", 5, Decimal("250"))]
        period2 = [self.create_rfm("C2", 3, Decimal("180"))]

        lens2 = analyze_period_comparison(period1, period2)
        dfs = lens2_to_dataframes(lens2)
        migration = dfs['migration']

        c1_status = migration[migration['customer_id'] == 'C1']['status'].iloc[0]
        c2_status = migration[migration['customer_id'] == 'C2']['status'].iloc[0]

        assert c1_status == 'churned'
        assert c2_status == 'new'


class TestAnalyzePeriodComparisonDF:
    """Test analyze_period_comparison_df convenience function."""

    def test_end_to_end_lens2_workflow(self):
        """End-to-end Lens 2 DataFrame workflow."""
        period1_rfm = rfm_to_dataframe([
            RFMMetrics("C1", 10, 5, Decimal("50"), datetime(2023,1,1), datetime(2023,6,30), Decimal("250")),
            RFMMetrics("C2", 20, 3, Decimal("50"), datetime(2023,1,1), datetime(2023,6,30), Decimal("150")),
        ])
        period2_rfm = rfm_to_dataframe([
            RFMMetrics("C1", 15, 3, Decimal("60"), datetime(2023,7,1), datetime(2023,12,31), Decimal("180")),
            RFMMetrics("C3", 10, 2, Decimal("50"), datetime(2023,7,1), datetime(2023,12,31), Decimal("100")),
        ])

        dfs = analyze_period_comparison_df(period1_rfm, period2_rfm)

        assert dfs['metrics'].iloc[0]['retention_rate'] == 50.0  # 1 of 2 retained
        assert len(dfs['migration']) == 3  # C1, C2, C3
```

### Success Criteria:

#### Automated Verification:
- [ ] All unit tests pass: `make test`
- [ ] Type checking passes: `mypy customer_base_audit/pandas/lens2.py`
- [ ] Linting passes: `make lint`
- [ ] Test coverage for lens2 module >= 90%: `pytest --cov=customer_base_audit/pandas/lens2 tests/test_pandas_lens2.py`

#### Manual Verification:
- [ ] All 4 DataFrames export cleanly to CSV
- [ ] Migration DataFrame filtering works as expected (filter by status)
- [ ] Period summaries correctly reflect nested Lens1Metrics
- [ ] Dict structure is intuitive for data analysis workflows

**Implementation Note**: After completing this phase and all automated verification passes, pause here for manual confirmation that the manual testing was successful before proceeding to Phase 4.

---

## Phase 4: Documentation and Examples

### Overview
Create comprehensive documentation and example notebooks demonstrating pandas integration workflows.

### Changes Required:

#### 1. User Guide Updates
**File**: `docs/user_guide.md`

Add new section "Working with Pandas DataFrames":

```markdown
## Working with Pandas DataFrames

For users who prefer pandas workflows, the `customer_base_audit.pandas` module provides DataFrame adapters for all Track A components.

### Quick Start

```python
import pandas as pd
from datetime import datetime
from customer_base_audit.pandas import calculate_rfm_df, analyze_single_period_df

# Load period data
periods_df = pd.read_csv('customer_periods.csv')

# Calculate RFM metrics
rfm_df = calculate_rfm_df(periods_df, observation_end=datetime(2023, 12, 31))

# Analyze single period
lens1_df = analyze_single_period_df(rfm_df)

# Export to BI tools
rfm_df.to_csv('rfm_scores.csv', index=False)
lens1_df.to_csv('lens1_metrics.csv', index=False)
```

### Period Comparison Workflow

```python
from customer_base_audit.pandas import analyze_period_comparison_df

# Calculate RFM for two periods
q1_rfm = calculate_rfm_df(q1_periods_df, datetime(2023, 3, 31))
q2_rfm = calculate_rfm_df(q2_periods_df, datetime(2023, 6, 30))

# Compare periods
comparison = analyze_period_comparison_df(q1_rfm, q2_rfm)

# Access results
print(f"Retention: {comparison['metrics']['retention_rate'].iloc[0]}%")

# Analyze churned customers
churned = comparison['migration'][comparison['migration']['status'] == 'churned']
churned_rfm = q1_rfm[q1_rfm['customer_id'].isin(churned['customer_id'])]
print(f"Avg churned customer value: ${churned_rfm['monetary'].mean():.2f}")
```

### Column Name Mapping

If your DataFrame has different column names, use the column mapping parameters:

```python
rfm_df = calculate_rfm_df(
    periods_df,
    observation_end=datetime(2023, 12, 31),
    customer_id_col='client_id',
    total_spend_col='revenue',
    total_orders_col='order_count',
)
```
```

#### 2. Example Notebook
**File**: `examples/pandas_integration_demo.ipynb` (NEW)

Create Jupyter notebook with:
- Load sample data from CSV
- RFM calculation with DataFrames
- Lens 1 analysis
- Lens 2 period comparison
- Visualization (matplotlib/seaborn)
- Export to various formats (CSV, Parquet, Excel)

#### 3. README Updates
**File**: `README.md`

Add pandas integration section:

```markdown
## Pandas Integration

Work seamlessly with pandas DataFrames:

```python
from customer_base_audit.pandas import calculate_rfm_df, analyze_single_period_df

# One-line DataFrame workflow
rfm_df = calculate_rfm_df(periods_df, datetime(2023, 12, 31))
lens1_df = analyze_single_period_df(rfm_df)

# Export for Tableau/PowerBI
rfm_df.to_csv('rfm_scores.csv')
```

See [`examples/pandas_integration_demo.ipynb`](examples/pandas_integration_demo.ipynb) for complete examples.
```

#### 4. API Reference
**File**: `docs/api_reference.md`

Add pandas module documentation:

```markdown
## customer_base_audit.pandas

Pandas DataFrame adapters for Track A components.

### RFM Adapters

#### `calculate_rfm_df(periods_df, observation_end, **col_mappings) -> pd.DataFrame`

Calculate RFM metrics from a pandas DataFrame.

**Parameters:**
- `periods_df` (pd.DataFrame): DataFrame with period aggregations
- `observation_end` (datetime): End date for recency calculation
- `customer_id_col` (str, optional): Column name for customer ID (default: 'customer_id')
- `period_start_col` (str, optional): Column name for period start (default: 'period_start')
- ... [other col mappings]

**Returns:** DataFrame with columns: customer_id, recency_days, frequency, monetary, total_spend, observation_start, observation_end

**Example:**
```python
rfm_df = calculate_rfm_df(periods_df, datetime(2023, 12, 31))
high_value = rfm_df[rfm_df['monetary'] > 100]
```

[... continue for all functions ...]
```

### Success Criteria:

#### Automated Verification:
- [ ] Example notebook executes without errors: `jupyter nbconvert --to notebook --execute examples/pandas_integration_demo.ipynb`
- [ ] All documentation links are valid: `markdown-link-check docs/user_guide.md`
- [ ] Code examples in docs are syntactically valid (manual check via copy-paste to Python REPL)

#### Manual Verification:
- [ ] User guide pandas section is clear and comprehensive
- [ ] Example notebook demonstrates real-world workflows
- [ ] README quick start is easy to follow
- [ ] API reference documentation is complete and accurate
- [ ] All code examples run successfully when copy-pasted

**Implementation Note**: After completing this phase and all verification passes, the pandas integration feature is complete and ready for release.

---

## Testing Strategy

### Unit Tests
**Locations**: `tests/test_pandas_rfm.py`, `tests/test_pandas_lens1.py`, `tests/test_pandas_lens2.py`

**Coverage**:
- Round-trip conversions (dataclass → DataFrame → dataclass)
- Empty input handling
- Type conversions (Decimal ↔ float, datetime ↔ pandas datetime64)
- Column validation (missing columns raise clear errors)
- Data validation (invalid data raises clear errors)
- Sorting guarantees (output sorted by customer_id)

### Integration Tests
**Location**: `tests/test_integration_pandas.py` (create in Phase 4)

**Scenarios**:
- Complete workflow: CSV → DataFrame → RFM → Lens 1 → CSV export
- Period comparison workflow with real-world data patterns
- Large dataset handling (10k+ customers)

### Performance Tests
**Marked with**: `@pytest.mark.slow`

**Benchmarks**:
- 10k customer RFM conversion < 2.0s
- 10k customer Lens 1 analysis < 1.0s
- 10k customer Lens 2 comparison < 3.0s
- Memory usage within 2x of dataclass operations

### Manual Testing
- Export to Tableau Desktop (verify CSV import)
- Export to PowerBI (verify Parquet import)
- Jupyter notebook workflow (verify interactive usage)
- Column name mapping with non-standard schemas

## Performance Considerations

### Conversion Overhead
- Expected overhead: < 5% vs direct dataclass operations
- Bottleneck: DataFrame row iteration in `dataframe_to_rfm()`
- Mitigation: Use vectorized operations where possible

### Memory Usage
- DataFrames store data in columnar format (more memory efficient for analytics)
- Conversion creates temporary list of dicts (overhead for small datasets)
- For large datasets (100k+ customers), pandas is more memory efficient

### Optimization Opportunities (Future Work)
- Use `df.to_dict(orient='records')` instead of `iterrows()` for import
- Vectorize Decimal/datetime conversions with `df.apply()`
- Add Dask support for distributed processing (out of scope for Issue #78)

## Migration Notes

This is an additive feature with no migration required:
- ✅ All existing dataclass APIs unchanged
- ✅ No breaking changes to Track A consumers
- ✅ Pandas module is opt-in (users can ignore if not needed)
- ✅ Zero impact on users not using pandas

## References

- Original issue: GitHub Issue #78
- Research document: `thoughts/shared/research/2025-10-10-issue-78-pandas-integration.md`
- Pandas integration proposal: `PANDAS_INTEGRATION_PROPOSAL.md`
- Existing conversion patterns: `customer_base_audit/models/model_prep.py:188-217, 287-311`
- Testing patterns: `tests/test_rfm.py`, `tests/test_lens1.py`, `tests/test_lens2.py`
- Track A scope: `.worktrees/track-a/AGENTS.md`
