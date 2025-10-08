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
**Status:** Implementation in progress (Track A, Phase 2)

Lens 2 will compare two time periods to analyze customer migration patterns:
- Retained customers (active in both periods)
- Churned customers (active in period 1, inactive in period 2)
- New customers (first purchase in period 2)
- Resurrected customers (previously churned, now active again)

Documentation will be added once the Lens 2 module is implemented.

### Lens 3: Cohort Evolution
**Status:** Implementation in progress (Track B, Phase 2)

Lens 3 will track how a single acquisition cohort performs over time:
- Cohort retention curves
- Revenue evolution by cohort age
- Purchase frequency trends within cohorts
- Time-to-second-purchase distributions

Documentation will be added once the Lens 3 module is implemented.

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
