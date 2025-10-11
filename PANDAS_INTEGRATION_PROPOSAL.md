# Pandas Integration Proposal for Track A

## Problem Statement

Current Track A interfaces (RFM, Lens 1, Lens 2) use dataclasses, which provide type safety but create friction for pandas-based workflows. Users need to manually convert between dataclasses and DataFrames.

## Proposed Solution: Adapter Layer (NOT Replacement)

### Principle: Multiple Interfaces, One Implementation

Keep dataclass-based core, add pandas adapters as **convenience layer**:

```python
# Core API (unchanged)
from customer_base_audit.foundation.rfm import calculate_rfm, RFMMetrics

# New pandas adapters
from customer_base_audit.pandas import (
    calculate_rfm_df,
    analyze_single_period_df,
    analyze_period_comparison_df,
    rfm_to_dataframe,
    dataframe_to_rfm
)
```

## API Design

### 1. RFM Pandas Adapter

```python
import pandas as pd
from datetime import datetime

def calculate_rfm_df(
    periods_df: pd.DataFrame,
    observation_end: datetime,
    customer_id_col: str = 'customer_id',
    period_start_col: str = 'period_start',
    period_end_col: str = 'period_end',
    total_orders_col: str = 'total_orders',
    total_spend_col: str = 'total_spend'
) -> pd.DataFrame:
    """
    Calculate RFM metrics from a pandas DataFrame.

    Args:
        periods_df: DataFrame with period aggregations
        observation_end: End date for recency calculation
        *_col: Column name mappings for flexibility

    Returns:
        DataFrame with columns:
        - customer_id
        - recency_days
        - frequency
        - monetary
        - total_spend
        - observation_start
        - observation_end

    Example:
        >>> periods_df = pd.DataFrame({
        ...     'customer_id': ['C1', 'C1', 'C2'],
        ...     'period_start': [...],
        ...     'total_orders': [2, 1, 3],
        ...     'total_spend': [100.0, 50.0, 200.0]
        ... })
        >>> rfm_df = calculate_rfm_df(periods_df, datetime(2023, 12, 31))
        >>> rfm_df.head()
    """
    # Convert DataFrame → List[PeriodAggregation]
    period_aggregations = dataframe_to_period_aggregations(
        periods_df,
        customer_id_col=customer_id_col,
        # ... other mappings
    )

    # Use core API
    rfm_metrics = calculate_rfm(period_aggregations, observation_end)

    # Convert List[RFMMetrics] → DataFrame
    return rfm_to_dataframe(rfm_metrics)


def rfm_to_dataframe(rfm_metrics: List[RFMMetrics]) -> pd.DataFrame:
    """Convert RFM metrics to pandas DataFrame."""
    return pd.DataFrame([
        {
            'customer_id': m.customer_id,
            'recency_days': m.recency_days,
            'frequency': m.frequency,
            'monetary': float(m.monetary),
            'total_spend': float(m.total_spend),
            'observation_start': m.observation_start,
            'observation_end': m.observation_end
        }
        for m in rfm_metrics
    ])


def dataframe_to_rfm(rfm_df: pd.DataFrame) -> List[RFMMetrics]:
    """Convert pandas DataFrame to RFM metrics."""
    return [
        RFMMetrics(
            customer_id=row['customer_id'],
            recency_days=int(row['recency_days']),
            frequency=int(row['frequency']),
            monetary=Decimal(str(row['monetary'])),
            observation_start=pd.to_datetime(row['observation_start']).to_pydatetime(),
            observation_end=pd.to_datetime(row['observation_end']).to_pydatetime(),
            total_spend=Decimal(str(row['total_spend']))
        )
        for _, row in rfm_df.iterrows()
    ]
```

### 2. Lens 1 Pandas Adapter

```python
def analyze_single_period_df(
    rfm_df: pd.DataFrame,
    rfm_scores_df: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Perform Lens 1 analysis on DataFrame.

    Returns:
        DataFrame with single row containing Lens 1 metrics:
        - total_customers
        - one_time_buyers
        - one_time_buyer_pct
        - total_revenue
        - top_10pct_revenue_contribution
        - top_20pct_revenue_contribution
        - avg_orders_per_customer
        - median_customer_value
    """
    # Convert DataFrame → List[RFMMetrics]
    rfm_metrics = dataframe_to_rfm(rfm_df)

    rfm_scores = None
    if rfm_scores_df is not None:
        rfm_scores = dataframe_to_rfm_scores(rfm_scores_df)

    # Use core API
    lens1 = analyze_single_period(rfm_metrics, rfm_scores=rfm_scores)

    # Convert Lens1Metrics → DataFrame
    return pd.DataFrame([{
        'total_customers': lens1.total_customers,
        'one_time_buyers': lens1.one_time_buyers,
        'one_time_buyer_pct': float(lens1.one_time_buyer_pct),
        'total_revenue': float(lens1.total_revenue),
        'top_10pct_revenue_contribution': float(lens1.top_10pct_revenue_contribution),
        'top_20pct_revenue_contribution': float(lens1.top_20pct_revenue_contribution),
        'avg_orders_per_customer': float(lens1.avg_orders_per_customer),
        'median_customer_value': float(lens1.median_customer_value)
    }])
```

### 3. Lens 2 Pandas Adapter

```python
def analyze_period_comparison_df(
    period1_rfm_df: pd.DataFrame,
    period2_rfm_df: pd.DataFrame
) -> dict[str, pd.DataFrame]:
    """
    Compare two periods using DataFrames.

    Returns:
        Dictionary with:
        - 'metrics': DataFrame with Lens 2 metrics (single row)
        - 'migration': DataFrame with customer migration details
        - 'period1_summary': DataFrame with period 1 Lens 1 metrics
        - 'period2_summary': DataFrame with period 2 Lens 1 metrics
    """
    # Convert DataFrames → List[RFMMetrics]
    period1_rfm = dataframe_to_rfm(period1_rfm_df)
    period2_rfm = dataframe_to_rfm(period2_rfm_df)

    # Use core API
    lens2 = analyze_period_comparison(period1_rfm, period2_rfm)

    # Convert to multiple DataFrames for easier analysis
    return {
        'metrics': pd.DataFrame([{
            'retention_rate': float(lens2.retention_rate),
            'churn_rate': float(lens2.churn_rate),
            'customer_count_change': lens2.customer_count_change,
            'revenue_change_pct': float(lens2.revenue_change_pct),
            'avg_order_value_change_pct': float(lens2.avg_order_value_change_pct)
        }]),
        'migration': pd.DataFrame([
            {'customer_id': cid, 'status': 'retained'}
            for cid in lens2.migration.retained
        ] + [
            {'customer_id': cid, 'status': 'churned'}
            for cid in lens2.migration.churned
        ] + [
            {'customer_id': cid, 'status': 'new'}
            for cid in lens2.migration.new
        ]),
        'period1_summary': analyze_single_period_df(period1_rfm_df),
        'period2_summary': analyze_single_period_df(period2_rfm_df)
    }
```

## Usage Examples

### Example 1: Quick RFM Analysis in Notebook

```python
import pandas as pd
from customer_base_audit.pandas import calculate_rfm_df
from datetime import datetime

# Load data
periods_df = pd.read_parquet('customer_periods.parquet')

# Calculate RFM (one line!)
rfm_df = calculate_rfm_df(periods_df, observation_end=datetime(2023, 12, 31))

# Immediate visualization
import matplotlib.pyplot as plt
rfm_df['monetary'].hist(bins=50)
plt.show()

# Immediate aggregation
high_value_customers = rfm_df[rfm_df['monetary'] > 100]
print(f"High value customers: {len(high_value_customers)}")
```

### Example 2: Period Comparison

```python
from customer_base_audit.pandas import calculate_rfm_df, analyze_period_comparison_df

# Calculate RFM for two periods
q1_rfm = calculate_rfm_df(q1_periods_df, datetime(2023, 3, 31))
q2_rfm = calculate_rfm_df(q2_periods_df, datetime(2023, 6, 30))

# Compare periods
comparison = analyze_period_comparison_df(q1_rfm, q2_rfm)

# Access results as DataFrames
print(comparison['metrics'])
print(f"Retention rate: {comparison['metrics']['retention_rate'].iloc[0]}%")

# Analyze churned customers
churned = comparison['migration'][comparison['migration']['status'] == 'churned']
churned_details = q1_rfm[q1_rfm['customer_id'].isin(churned['customer_id'])]
print(f"Average monetary value of churned customers: ${churned_details['monetary'].mean():.2f}")
```

### Example 3: Export to Tableau/PowerBI

```python
# Calculate metrics
rfm_df = calculate_rfm_df(periods_df, observation_end)
lens1_df = analyze_single_period_df(rfm_df)

# Export for BI tools
rfm_df.to_csv('rfm_scores.csv', index=False)
lens1_df.to_csv('lens1_metrics.csv', index=False)

# Or to Parquet for data lake
rfm_df.to_parquet('s3://data-lake/customer-analytics/rfm_scores.parquet')
```

## Why NOT Full Sklearn Integration?

### Track A is Descriptive Analytics, Not ML

**Sklearn is designed for**:
- Supervised learning (X, y)
- Predictive modeling
- Cross-validation, grid search
- Classification/regression metrics

**Track A does**:
- Customer segmentation (no y labels)
- Descriptive statistics (revenue, retention)
- Time-series comparison (not prediction)

### Sklearn API Doesn't Fit Our Domain

**Sklearn expects**:
```python
estimator.fit(X, y)  # We don't have y (no target variable)
estimator.transform(X)  # Returns X', but we return metrics (scalars)
estimator.predict(X)  # We're not predicting, we're describing
```

**Our API makes sense**:
```python
rfm = calculate_rfm(periods, observation_end)  # Clear what it does
lens1 = analyze_single_period(rfm)  # Returns summary metrics, not transformed X
```

### Exception: Track B Models SHOULD Use Sklearn

**BG/NBD and Gamma-Gamma** (Track B, not our scope) are ML models:
```python
# This makes sense for Track B
from customer_base_audit.sklearn import BGNBDEstimator

model = BGNBDEstimator()
model.fit(rfm_df[['frequency', 'recency', 'T']])
predictions = model.predict(rfm_df, time_periods=12)  # Predict purchases
```

## NumPy Integration: Not Recommended

**Why NumPy arrays are a poor fit**:

1. **Loss of column semantics**:
```python
# NumPy: What does this mean?
rfm_array = np.array([[30, 5, 100.0], ...])  # Which column is what?

# Pandas: Self-documenting
rfm_df = pd.DataFrame({'recency_days': [30], 'frequency': [5], 'monetary': [100.0]})
```

2. **Loss of customer IDs**:
```python
# NumPy: Can't track which customer is which
rfm_array[0]  # [30, 5, 100.0] - Who is this?

# Pandas: Customer ID preserved
rfm_df.loc[rfm_df['customer_id'] == 'C123']
```

3. **Loss of type safety**:
```python
# NumPy: Everything is float64
rfm_array = np.array([[30.0, 5.0, 100.0]])  # recency_days should be int!

# Dataclass: Type-checked
@dataclass
class RFMMetrics:
    recency_days: int  # mypy validates this
    frequency: int
    monetary: Decimal
```

**When NumPy makes sense**: Track B models (parameter vectors, gradients)

## Implementation Plan

### Phase 1: Core Adapters (2 days)
- `customer_base_audit/pandas/__init__.py`
- `customer_base_audit/pandas/rfm.py` - RFM adapters
- `customer_base_audit/pandas/lens1.py` - Lens 1 adapters
- `customer_base_audit/pandas/lens2.py` - Lens 2 adapters
- Unit tests for all adapters

### Phase 2: Documentation (1 day)
- Update user guide with pandas examples
- Add notebook: "Quick Start with Pandas"
- Add notebook: "From CSV to Insights in 5 Minutes"

### Phase 3: Integration Examples (1 day)
- Example: Jupyter notebook workflow
- Example: Export to Tableau
- Example: Integration with Databricks

## Benefits

✅ **Reduced Friction**: One-line conversion between dataclasses and DataFrames
✅ **Backward Compatible**: Existing code unchanged
✅ **Type Safety Preserved**: Core API still uses dataclasses
✅ **Enterprise Ready**: Easy integration with BI tools, data lakes
✅ **Jupyter Friendly**: Natural pandas workflow in notebooks

## Non-Goals (Explicit Exclusions)

❌ **Replace dataclass API**: Core stays as-is
❌ **Full sklearn compatibility**: Not appropriate for descriptive analytics
❌ **NumPy arrays**: Lose too much semantic information
❌ **Spark/Dask native**: Use pandas UDFs instead
❌ **Streaming API**: Batch processing only for MVP

## Success Metrics

1. **Adoption**: % of users using pandas adapters vs. core API
2. **Reduced Support**: Fewer "how do I convert to DataFrame?" questions
3. **Integration**: Evidence of Track A tools in production BI dashboards
4. **Performance**: Pandas adapter overhead < 5% of core API time

## Related Issues

- Issue #75: Parallel Processing (could be applied to pandas adapters too)
- Track B: BG/NBD and Gamma-Gamma sklearn compatibility (separate effort)

## Recommendation

**Yes, we should add pandas integration**, but as an **adapter layer, not a replacement**:

1. Keep dataclass-based core API (type safety, maintainability)
2. Add `customer_base_audit.pandas` module with convenience functions
3. Document pandas workflow in user guide
4. Do NOT force sklearn compatibility on Track A (wrong abstraction)
5. Consider sklearn for Track B models (BG/NBD, Gamma-Gamma)

**Priority**: Medium-High
- Enables easier adoption in data science workflows
- Relatively low effort (4 days implementation + docs)
- High value for enterprise integration
