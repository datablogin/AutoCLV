# AutoCLV User Guide

## Version Compatibility

**Python Version:** 3.12 or higher

**Core Dependencies:**
- pandas >= 2.1.0
- matplotlib >= 3.8.0
- click >= 8.1.7
- jinja2 >= 3.1.2
- plotly >= 5.18.0

**Development Dependencies:**
- pytest >= 8.4.0
- ruff == 0.12.12

**Tested Platforms:**
- macOS (Darwin 24.6.0)
- Linux (Ubuntu 22.04+)
- Windows 10/11

## Installation and Setup

### Installing from Source

```bash
# Clone the repository
git clone https://github.com/datablogin/AutoCLV.git
cd AutoCLV

# Install in development mode
pip install -e .

# Install development dependencies (optional)
pip install -e ".[dev]"
```

### Verifying Installation

```python
# Test the installation
from customer_base_audit.foundation.data_mart import CustomerDataMartBuilder
from customer_base_audit.analyses.lens1 import analyze_single_period
from customer_base_audit.synthetic.texas_clv_client import generate_texas_clv_client

# Generate sample data
customers, transactions, city_map = generate_texas_clv_client(total_customers=100, seed=42)
print(f"✓ Generated {len(customers)} customers and {len(transactions)} transactions")
```

If this runs without errors, your installation is successful!

### Running Tests

```bash
# Run the full test suite
pytest

# Run with coverage
pytest --cov=customer_base_audit

# Run specific test file
pytest tests/test_lens1.py
```

## Configuration Options
[To be filled in Phase 2]

## Data Preparation Requirements

AutoCLV expects transaction data in a specific format for the CustomerDataMartBuilder.

### Required Transaction Fields

Each transaction must be a dictionary (or dict-like object) with these fields:

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `order_id` | str | Unique identifier for the order | `"ORD-12345"` |
| `customer_id` | str | Unique identifier for the customer | `"CUST-001"` |
| `event_ts` or `order_ts` | datetime | Timestamp of the transaction | `datetime(2024, 1, 15, 14, 30)` |
| `quantity` | int (optional) | Quantity of items purchased | `2` |
| `unit_price` | float | Price per unit | `29.99` |
| `product_id` | str (optional) | Product identifier | `"SKU-ABC"` |

### Example Transaction Data

```python
from datetime import datetime

transactions = [
    {
        "order_id": "ORD-001",
        "customer_id": "CUST-001",
        "event_ts": datetime(2024, 1, 15, 14, 30),
        "quantity": 2,
        "unit_price": 29.99,
        "product_id": "SKU-ABC"
    },
    {
        "order_id": "ORD-002",
        "customer_id": "CUST-001",
        "event_ts": datetime(2024, 2, 10, 9, 15),
        "quantity": 1,
        "unit_price": 49.99,
        "product_id": "SKU-XYZ"
    },
    # ... more transactions
]
```

### Data Quality Requirements

**Required:**
- ✅ All `customer_id` values must be strings
- ✅ All `event_ts`/`order_ts` must be datetime objects (not strings)
- ✅ Quantities cannot be negative
- ✅ Unit prices cannot be negative

**Recommended:**
- 📊 At least 100 customers for stable RFM scoring
- 📊 At least 1 year of transaction history
- 📊 Customer acquisition dates available for cohort analysis

### Loading Your Data

```python
from customer_base_audit.foundation.data_mart import CustomerDataMartBuilder, PeriodGranularity

# If you have a pandas DataFrame
import pandas as pd
df = pd.read_csv('your_transactions.csv')
df['event_ts'] = pd.to_datetime(df['event_ts'])
transactions = df.to_dict('records')

# Build the data mart
builder = CustomerDataMartBuilder(period_granularities=[PeriodGranularity.MONTH])
mart = builder.build(transactions)

print(f"Processed {len(mart.orders)} orders")
print(f"Generated {len(mart.periods[PeriodGranularity.MONTH])} monthly aggregations")
```

### Using Synthetic Data for Testing

If you don't have real data yet, use the Texas CLV synthetic data generator:

```python
from customer_base_audit.synthetic.texas_clv_client import generate_texas_clv_client

# Generate 1000 customers across 4 cities
customers, transactions, city_map = generate_texas_clv_client(
    total_customers=1000,
    seed=42  # For reproducibility
)

print(f"Generated {len(transactions)} transactions")
print(f"Cities: {set(city_map.values())}")
```

## Running Five Lenses Audit

The Five Lenses framework from "The Customer-Base Audit" provides a comprehensive view of your customer base through five complementary analyses. Each lens answers specific questions about customer behavior and business health.

### Lens 1: Single Period Analysis

Lens 1 provides a snapshot view of your customer base within a single time period, answering fundamental questions:
- How many customers are there?
- What percentage are one-time buyers?
- How concentrated is revenue? (Pareto analysis)
- What is the distribution across RFM segments?

#### Running Lens 1 Analysis

**Data Validation First**

Before running analysis, validate your data to catch common issues:

```python
from datetime import datetime
from customer_base_audit.foundation.data_mart import CustomerDataMartBuilder, PeriodGranularity

# Quick validation checks
print(f"Transaction count: {len(transactions)}")
print(f"Date range: {min(t['event_ts'] for t in transactions)} to {max(t['event_ts'] for t in transactions)}")
print(f"Unique customers: {len(set(t['customer_id'] for t in transactions))}")
print(f"Unique orders: {len(set(t['order_id'] for t in transactions))}")

# This will raise helpful errors if data is malformed
try:
    builder = CustomerDataMartBuilder(period_granularities=[PeriodGranularity.MONTH])
    mart = builder.build(transactions)
    print(f"✓ Data validation passed: {len(mart.orders)} orders processed")
except (ValueError, TypeError, KeyError) as e:
    print(f"❌ Data validation failed: {e}")
    # Fix your data and try again
```

**Complete Example**

```python
from datetime import datetime
from dataclasses import asdict
from customer_base_audit.foundation.data_mart import CustomerDataMartBuilder, PeriodGranularity
from customer_base_audit.foundation.rfm import calculate_rfm, calculate_rfm_scores
from customer_base_audit.analyses.lens1 import analyze_single_period
from customer_base_audit.synthetic.texas_clv_client import generate_texas_clv_client

# 1. Generate synthetic data (or load your own transactions)
customers, transactions, city_map = generate_texas_clv_client(total_customers=1000, seed=42)

# 2. Build the data mart with monthly granularity
builder = CustomerDataMartBuilder(period_granularities=[PeriodGranularity.MONTH])
mart = builder.build([asdict(txn) for txn in transactions])

# 3. Get period aggregations and calculate RFM metrics
period_aggregations = mart.periods[PeriodGranularity.MONTH]
observation_end = datetime(2025, 12, 31, 23, 59, 59)
rfm_metrics = calculate_rfm(
    period_aggregations=period_aggregations,
    observation_end=observation_end
)

# 4. Calculate RFM scores (required for RFM distribution analysis)
rfm_scores = calculate_rfm_scores(rfm_metrics)

# 5. Run Lens 1 analysis (pass rfm_scores for distribution, or None to skip)
lens1_results = analyze_single_period(rfm_metrics, rfm_scores)

# 6. View results
print(f"Total Customers: {lens1_results.total_customers}")
print(f"One-Time Buyers: {lens1_results.one_time_buyers} ({lens1_results.one_time_buyer_pct}%)")
print(f"Total Revenue: ${lens1_results.total_revenue:,.2f}")
print(f"Top 10% Revenue Contribution: {lens1_results.top_10pct_revenue_contribution}%")
print(f"Top 20% Revenue Contribution: {lens1_results.top_20pct_revenue_contribution}%")
print(f"Avg Orders per Customer: {lens1_results.avg_orders_per_customer}")
print(f"Median Customer Value: ${lens1_results.median_customer_value}")
```

#### Interpreting Lens 1 Results

**One-Time Buyer Percentage**
- **< 20%**: Excellent - strong repeat purchase behavior
- **20-40%**: Good - typical for many businesses
- **40-60%**: Concerning - may indicate acquisition or retention issues
- **> 60%**: Critical - investigate customer experience and product-market fit

**Revenue Concentration (Pareto Analysis)**
- The "80/20 rule" suggests ~80% of revenue from ~20% of customers
- **Top 10% contribution < 40%**: More evenly distributed revenue (subscription businesses)
- **Top 10% contribution 40-60%**: Typical distribution
- **Top 10% contribution > 60%**: Highly concentrated - monitor top customer churn risk

**RFM Distribution**
- `555` scores: Your best customers (high recency, frequency, monetary)
- `111` scores: At-risk customers (low on all dimensions)
- Look for imbalances: Many `1XX` scores suggest recency issues (customers not returning)

#### Example Output

```
Total Customers: 343
One-Time Buyers: 10 (2.92%)
Total Revenue: $1,195,624.61
Top 10% Revenue Contribution: 23.0%
Top 20% Revenue Contribution: 41.3%
Avg Orders per Customer: 15.74
Median Customer Value: $3,233.89
```

**Interpretation:** This Texas CLV dataset shows excellent customer retention (only 2.92% one-time buyers) with relatively distributed revenue (top 10% contribute 23%). The high average orders per customer (15.74) indicates strong repeat purchase behavior, typical of a successful subscription or high-engagement retail business.

#### Performance Expectations

Lens 1 analysis performance scales linearly with customer count:

- **< 1,000 customers**: Near-instant (< 1 second)
- **1,000-10,000 customers**: 1-3 seconds
- **10,000-100,000 customers**: 3-10 seconds
- **100,000-1M customers**: 10-60 seconds
- **> 1M customers**: Consider batching by cohort or time period

**Memory usage**: Approximately 1-2 MB per 1,000 customers for RFM calculations.

**Tip**: For very large datasets, process cohorts separately and aggregate results.

#### Advanced: Cohort-Specific Lens 1 Analysis

You can analyze specific cohorts by filtering customers before running Lens 1:

```python
from customer_base_audit.foundation.cohorts import create_monthly_cohorts, assign_cohorts

# Create monthly cohorts (see cohort documentation in foundation module)
# Note: Full cohort analysis will be available in Lens 3 (Phase 2)
cohort_definitions = create_monthly_cohorts(
    customers=customers,
    start_date=datetime(2024, 4, 1),
    end_date=datetime(2024, 12, 31)
)

# Assign customers to cohorts
cohort_assignments = assign_cohorts(customers, cohort_definitions)

# Analyze a specific cohort (e.g., April 2024 acquisitions)
april_cohort_id = "2024-04"
april_customer_ids = {
    cust_id for cust_id, cohort_id in cohort_assignments.items()
    if cohort_id == april_cohort_id
}

# Filter RFM metrics for this cohort
april_rfm = [rfm for rfm in rfm_metrics if rfm.customer_id in april_customer_ids]
april_scores = [score for score in rfm_scores if score.customer_id in april_customer_ids]

# Run Lens 1 on the cohort
april_lens1 = analyze_single_period(april_rfm, april_scores)

print(f"April 2024 Cohort Analysis:")
print(f"  Customers: {april_lens1.total_customers}")
print(f"  One-time buyers: {april_lens1.one_time_buyer_pct}%")
print(f"  Top 20% revenue contribution: {april_lens1.top_20pct_revenue_contribution}%")
```

This cohort-specific analysis is particularly useful for:
- Comparing acquisition channel performance
- Identifying seasonal cohort differences
- Tracking cohort maturation over time

#### Common Patterns and What They Mean

1. **High one-time buyer % + Low top 10% concentration**
   - Many customers making small single purchases
   - Action: Improve onboarding and post-purchase engagement

2. **Low one-time buyer % + High top 10% concentration**
   - Strong repeat business driven by a few power users
   - Action: Reduce dependence by growing the middle tier

3. **Increasing median customer value over time**
   - Customers are spending more per capita
   - Action: Indicative of good product expansion or pricing

#### See Also

For more examples and edge cases:
- **`tests/test_lens1.py`**: Comprehensive test cases including edge conditions
- **`tests/test_rfm.py`**: RFM calculation examples and validation tests
- **`tests/test_customer_foundation.py`**: Data mart building examples

### Lens 2: Period-to-Period Comparison

Lens 2 compares two time periods to analyze customer migration patterns and business trends, answering questions like:
- How many customers were retained vs. churned between periods?
- What percentage of customers are new vs. reactivated?
- How did revenue and average order value change?
- Which customer segments grew or declined?

#### Running Lens 2 Analysis

**Complete Example**

```python
from datetime import datetime
from dataclasses import asdict
from customer_base_audit.foundation.data_mart import CustomerDataMartBuilder, PeriodGranularity
from customer_base_audit.foundation.rfm import calculate_rfm
from customer_base_audit.analyses.lens2 import analyze_period_comparison
from customer_base_audit.synthetic.texas_clv_client import generate_texas_clv_client

# 1. Generate synthetic data (or load your own transactions)
customers, transactions, city_map = generate_texas_clv_client(total_customers=1000, seed=42)

# 2. Build the data mart with monthly granularity
builder = CustomerDataMartBuilder(period_granularities=[PeriodGranularity.MONTH])
mart = builder.build([asdict(txn) for txn in transactions])

# 3. Get period aggregations for two consecutive periods
all_period_aggregations = mart.periods[PeriodGranularity.MONTH]

# Define two periods to compare (e.g., Q1 vs Q2 2025)
period1_start = datetime(2025, 1, 1)
period1_end = datetime(2025, 3, 31, 23, 59, 59)
period2_start = datetime(2025, 4, 1)
period2_end = datetime(2025, 6, 30, 23, 59, 59)

# Filter aggregations for each period
period1_aggs = [
    agg for agg in all_period_aggregations
    if period1_start <= agg.period_start <= period1_end
]
period2_aggs = [
    agg for agg in all_period_aggregations
    if period2_start <= agg.period_start <= period2_end
]

# 4. Calculate RFM metrics for each period
period1_rfm = calculate_rfm(
    period_aggregations=period1_aggs,
    observation_end=period1_end
)
period2_rfm = calculate_rfm(
    period_aggregations=period2_aggs,
    observation_end=period2_end
)

# 5. (Optional) Get all customer history for reactivation tracking
all_customer_ids = list(set(cust.customer_id for cust in customers))

# 6. Run Lens 2 analysis
lens2_results = analyze_period_comparison(
    period1_rfm=period1_rfm,
    period2_rfm=period2_rfm,
    all_customer_history=all_customer_ids  # Enable reactivation tracking
)

# 7. View migration results
print(f"Customer Migration:")
print(f"  Retained: {len(lens2_results.migration.retained)}")
print(f"  Churned: {len(lens2_results.migration.churned)}")
print(f"  New: {len(lens2_results.migration.new)}")
print(f"  Reactivated: {len(lens2_results.migration.reactivated)}")

print(f"\nRetention Metrics:")
print(f"  Retention Rate: {lens2_results.retention_rate}%")
print(f"  Churn Rate: {lens2_results.churn_rate}%")
print(f"  Reactivation Rate: {lens2_results.reactivation_rate}%")

print(f"\nBusiness Metrics:")
print(f"  Customer Count Change: {lens2_results.customer_count_change:+d}")
print(f"  Revenue Change: {lens2_results.revenue_change_pct:+}%")
print(f"  AOV Change: {lens2_results.avg_order_value_change_pct:+}%")
```

#### Interpreting Lens 2 Results

**Retention and Churn Rates**
- **Retention Rate > 80%**: Excellent - strong customer stickiness
- **Retention Rate 60-80%**: Good - typical for healthy businesses
- **Retention Rate 40-60%**: Concerning - investigate customer satisfaction
- **Retention Rate < 40%**: Critical - major retention issues

Note: Retention + churn always equals 100% (they're complementary metrics).

**Reactivation Rate**
- **> 20%**: High reactivation - effective win-back campaigns
- **10-20%**: Moderate reactivation - standard marketing efforts
- **< 10%**: Low reactivation - opportunity to improve win-back strategies
- **0%**: No historical tracking provided (set `all_customer_history=None`)

**Revenue and AOV Changes**
- **Revenue up, AOV up**: Growing customer value - excellent
- **Revenue up, AOV down**: Volume growth with discounting - monitor margins
- **Revenue down, AOV up**: Losing customers but improving quality - mixed signal
- **Revenue down, AOV down**: Concerning - investigate pricing and competition

#### Example Output

```
Customer Migration:
  Retained: 287
  Churned: 56
  New: 48
  Reactivated: 12

Retention Metrics:
  Retention Rate: 83.67%
  Churn Rate: 16.33%
  Reactivation Rate: 3.59%

Business Metrics:
  Customer Count Change: -8
  Revenue Change: +5.23%
  AOV Change: +8.14%
```

**Interpretation:** This example shows healthy retention (83.67%) with modest customer decline (-8 customers). Despite fewer customers, revenue grew 5.23% due to improved average order value (+8.14%), suggesting successful upselling or premium product adoption. The 3.59% reactivation rate indicates effective win-back campaigns.

#### Common Migration Patterns

1. **High retention + positive revenue growth**
   - Healthy, growing business
   - Action: Scale acquisition while maintaining retention

2. **Declining retention + high new customer rate**
   - "Leaky bucket" - acquiring to replace churned customers
   - Action: Focus on improving onboarding and early retention

3. **High reactivation rate + declining retention**
   - Win-back campaigns working but core retention failing
   - Action: Address root causes of churn, not just symptoms

#### See Also

- **`tests/test_lens2.py`**: Comprehensive test cases with edge conditions
- **Lens 1 documentation**: Understanding single-period metrics
- **Lens 3 documentation**: Tracking cohort evolution over time

### Lens 3: Single Cohort Evolution

Lens 3 tracks how a single acquisition cohort's behavior evolves over time from first purchase, answering questions like:
- How does retention decay over the cohort's lifetime?
- How does revenue per customer change as the cohort matures?
- What percentage of the cohort remains active each period?
- How do purchase patterns evolve after acquisition?

#### Running Lens 3 Analysis

**Complete Example**

```python
from datetime import datetime
from dataclasses import asdict
from customer_base_audit.foundation.data_mart import CustomerDataMartBuilder, PeriodGranularity
from customer_base_audit.foundation.cohorts import create_monthly_cohorts, assign_cohorts
from customer_base_audit.analyses.lens3 import analyze_cohort_evolution
from customer_base_audit.synthetic.texas_clv_client import generate_texas_clv_client

# 1. Generate synthetic data (or load your own transactions)
customers, transactions, city_map = generate_texas_clv_client(total_customers=1000, seed=42)

# 2. Build the data mart with monthly granularity
builder = CustomerDataMartBuilder(period_granularities=[PeriodGranularity.MONTH])
mart = builder.build([asdict(txn) for txn in transactions])

# 3. Create monthly cohorts and assign customers
cohort_definitions = create_monthly_cohorts(
    customers=customers,
    start_date=datetime(2024, 1, 1),
    end_date=datetime(2024, 12, 31)
)
cohort_assignments = assign_cohorts(customers, cohort_definitions)

# 4. Select a specific cohort to analyze (e.g., January 2024 acquisitions)
cohort_name = "2024-01"
cohort_customer_ids = [
    cust_id for cust_id, coh_id in cohort_assignments.items()
    if coh_id == cohort_name
]

# Find the cohort's acquisition date
cohort_definition = next(c for c in cohort_definitions if c.cohort_id == cohort_name)
acquisition_date = cohort_definition.period_start

# 5. Get period aggregations for this cohort
all_period_aggregations = mart.periods[PeriodGranularity.MONTH]

# 6. Run Lens 3 analysis
lens3_results = analyze_cohort_evolution(
    cohort_name=cohort_name,
    acquisition_date=acquisition_date,
    period_aggregations=all_period_aggregations,
    cohort_customer_ids=cohort_customer_ids
)

# 7. View cohort evolution metrics
print(f"Cohort: {lens3_results.cohort_name}")
print(f"Acquisition Date: {lens3_results.acquisition_date.strftime('%Y-%m-%d')}")
print(f"Initial Cohort Size: {lens3_results.cohort_size}")
print(f"\nEvolution by Period:")

for period_metrics in lens3_results.periods[:6]:  # First 6 periods
    print(f"\nPeriod {period_metrics.period_number}:")
    print(f"  Active Customers: {period_metrics.active_customers}")
    print(f"  Retention Rate: {period_metrics.retention_rate:.2%}")
    print(f"  Avg Orders per Active Customer: {period_metrics.avg_orders_per_customer:.2f}")
    print(f"  Avg Revenue per Active Customer: ${period_metrics.avg_revenue_per_customer:,.2f}")
    print(f"  Total Cohort Revenue: ${period_metrics.total_revenue:,.2f}")
```

#### Interpreting Lens 3 Results

**Retention Curves**
- **Period 0 retention = 100%**: All customers make at least one purchase in acquisition period
- **Period 1 retention 60-80%**: Healthy second purchase rate
- **Period 1 retention 40-60%**: Moderate - improve onboarding
- **Period 1 retention < 40%**: Weak - critical onboarding issue

**Retention Decay Patterns**
- **Slow decay (convex curve)**: High-value, loyal customers - excellent
- **Linear decay**: Standard retention - typical for many businesses
- **Rapid early decay (concave curve)**: One-time buyer problem - concerning

**Revenue Evolution**
- **Increasing revenue per active customer**: Growing customer value over time - excellent
- **Stable revenue per active customer**: Consistent spending - good
- **Declining revenue per active customer**: Diminishing engagement - investigate product fit

**Key Metric: Period 0 → Period 1 Drop**
The retention drop from Period 0 to Period 1 is the most critical metric:
- **< 20% drop**: Excellent - strong product-market fit
- **20-40% drop**: Good - standard for most businesses
- **40-60% drop**: Concerning - weak second purchase conversion
- **> 60% drop**: Critical - investigate onboarding and initial experience

#### Example Output

```
Cohort: 2024-01
Acquisition Date: 2024-01-01
Initial Cohort Size: 85

Evolution by Period:

Period 0:
  Active Customers: 85
  Retention Rate: 100.00%
  Avg Orders per Active Customer: 1.24
  Avg Revenue per Active Customer: $1,452.35
  Total Cohort Revenue: $123,449.75

Period 1:
  Active Customers: 68
  Retention Rate: 80.00%
  Avg Orders per Active Customer: 2.15
  Avg Revenue per Active Customer: $1,823.47
  Total Cohort Revenue: $123,995.96

Period 2:
  Active Customers: 62
  Retention Rate: 72.94%
  Avg Orders per Active Customer: 1.89
  Avg Revenue per Active Customer: $1,654.28
  Total Cohort Revenue: $102,565.36

Period 3:
  Active Customers: 58
  Retention Rate: 68.24%
  Avg Orders per Active Customer: 1.76
  Avg Revenue per Active Customer: $1,598.12
  Total Cohort Revenue: $92,690.96
```

**Interpretation:** This cohort shows strong retention with only 20% churn after Period 0, and active customers are increasing their order frequency and spend in Period 1. The gradual retention decay (100% → 80% → 73% → 68%) indicates healthy long-term engagement. Increasing revenue per active customer in Period 1 suggests successful upselling or product expansion.

#### Common Cohort Patterns

1. **High Period 0 retention, steep Period 1 drop**
   - One-time buyer problem
   - Action: Improve post-purchase engagement and second purchase incentives

2. **Gradual retention decay with stable revenue**
   - Healthy cohort maturation
   - Action: Focus on extending customer lifetime

3. **Retention stable but declining revenue per customer**
   - Customer fatigue or decreasing engagement
   - Action: Introduce new products or refresh offering

#### Advanced: Comparing Multiple Cohorts

You can run Lens 3 on multiple cohorts to identify trends:

```python
# Compare Q1 cohorts
cohort_names = ["2024-01", "2024-02", "2024-03"]

for cohort_name in cohort_names:
    cohort_customer_ids = [
        cust_id for cust_id, coh_id in cohort_assignments.items()
        if coh_id == cohort_name
    ]

    cohort_def = next(c for c in cohort_definitions if c.cohort_id == cohort_name)

    lens3 = analyze_cohort_evolution(
        cohort_name=cohort_name,
        acquisition_date=cohort_def.period_start,
        period_aggregations=all_period_aggregations,
        cohort_customer_ids=cohort_customer_ids
    )

    # Compare Period 1 retention across cohorts
    period1 = lens3.periods[1] if len(lens3.periods) > 1 else None
    if period1:
        print(f"{cohort_name} Period 1 retention: {period1.retention_rate:.1%}")
```

This helps identify if retention is improving or declining over time for newer cohorts.

#### See Also

- **`tests/test_lens3.py`**: Comprehensive test cases with edge conditions
- **`customer_base_audit/foundation/cohorts.py`**: Cohort creation and assignment
- **Lens 2 documentation**: Period-to-period comparison for retention analysis

### Lens 4: Multi-Cohort Comparison
[To be filled in Phase 5]

### Lens 5: Overall Customer Base Health
[To be filled in Phase 5]

## Training CLV Models
[To be filled in Phase 3]

## Interpreting Results
[To be filled in Phase 4]

## Complete Workflow Examples
### Example 1: Analyzing Subscription Business
[To be filled in Phase 4]

### Example 2: E-commerce Customer Analysis
[To be filled in Phase 4]

## API Reference
[To be filled in Phase 3]

## Troubleshooting Common Issues

### Installation and Setup Issues

#### "Module not found" errors
```bash
# Ensure you installed the package
pip install -e .

# Verify installation
pip list | grep customer-base-audit

# If still failing, try reinstalling
pip uninstall customer-base-audit
pip install -e .
```

#### Python version incompatibility
```bash
# Check your Python version
python --version

# AutoCLV requires Python 3.12+
# Install Python 3.12 if needed, then:
python3.12 -m pip install -e .
```

#### Dependency conflicts
```bash
# Create a fresh virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in clean environment
pip install -e .
```

#### Import errors after installation
```python
# Make sure you're importing from the correct package
from customer_base_audit.foundation.data_mart import CustomerDataMartBuilder  # ✓ Correct
from customer_data_mart import CustomerDataMartBuilder  # ✗ Wrong
```

### Data Preparation Issues
[To be filled in Phase 2]

### Model Training Issues
[To be filled in Phase 3]

### Validation and Performance Issues
[To be filled in Phase 4]

### General Troubleshooting

#### Getting help
1. **Check the documentation**: Review this user guide and API reference
2. **Run tests**: `pytest -v` to see if core functionality works
3. **Enable verbose logging**: Most functions accept a `verbose` parameter
4. **Synthetic data**: Use `generate_texas_clv_client()` to test with known-good data
5. **GitHub Issues**: https://github.com/datablogin/AutoCLV/issues

#### Common gotchas
- ⚠️ **Datetime vs string**: `event_ts` must be a datetime object, not a string
- ⚠️ **Small datasets**: RFM scoring requires 100+ customers for stable quintiles
- ⚠️ **Missing acquisitions**: Cohort analysis requires customer acquisition dates
- ⚠️ **Timezone awareness**: Keep datetimes timezone-naive or all use the same timezone
