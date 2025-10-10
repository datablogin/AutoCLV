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

## Synthetic Data Toolkit

The Synthetic Data Toolkit provides realistic, configurable customer transaction data for testing, development, and validation. It generates statistically sound datasets that match real-world customer behavior patterns, enabling you to test CLV models and audit logic without requiring production data.

### Why Use Synthetic Data?

**Benefits:**
- ✅ **Privacy-safe**: No customer PII or sensitive business data
- ✅ **Reproducible**: Seeded generators produce identical results
- ✅ **Configurable**: Test edge cases (high churn, promotions, product recalls)
- ✅ **CI/CD friendly**: Every test run validates against fresh, known-good data
- ✅ **Onboarding**: New team members can run full pipelines immediately

**Use Cases:**
- Unit and integration testing
- Model validation and benchmarking
- Demo environments and user training
- Algorithm development without production data access
- Stress testing edge cases and business scenarios

### Quick Start: Texas CLV Generator

For basic testing, use the Texas CLV synthetic data generator:

```python
from customer_base_audit.synthetic.texas_clv_client import generate_texas_clv_client

# Generate 1000 customers across 4 Texas cities
customers, transactions, city_map = generate_texas_clv_client(
    total_customers=1000,
    seed=42  # For reproducibility
)

print(f"Generated {len(transactions)} transactions")
print(f"Customers: {len(customers)}")
print(f"Cities: {set(city_map.values())}")
# Output: Generated 15,000+ transactions, 1000 customers across Austin, Dallas, Houston, San Antonio
```

**What Texas CLV Generates:**
- Realistic customer acquisition dates spread over time
- Multi-line orders with varying quantities and prices
- Geographic segmentation (4 cities)
- Natural churn patterns
- Repeat purchase behavior

### Scenario Packs

Scenario packs are pre-configured business situations for targeted testing. Each scenario represents a realistic challenge or pattern you might encounter in production.

#### Available Scenarios

**1. BASELINE_SCENARIO** - Moderate, stable business
```python
from customer_base_audit.synthetic import BASELINE_SCENARIO, generate_customers, generate_transactions
from datetime import date

customers = generate_customers(500, date(2024, 1, 1), date(2024, 12, 31), seed=42)
transactions = generate_transactions(
    customers,
    date(2024, 1, 1),
    date(2024, 12, 31),
    scenario=BASELINE_SCENARIO
)

print(f"Generated {len(transactions)} transactions (baseline)")
```

**Parameters:**
- Churn: 8% monthly
- Orders per month: 1.2 per customer
- Mean unit price: $30.00
- Use for: General testing, benchmarking, stable business modeling

**2. HIGH_CHURN_SCENARIO** - Struggling retention
```python
from customer_base_audit.synthetic import HIGH_CHURN_SCENARIO

transactions = generate_transactions(
    customers,
    date(2024, 1, 1),
    date(2024, 12, 31),
    scenario=HIGH_CHURN_SCENARIO
)
```

**Parameters:**
- Churn: 30% monthly (very high)
- Orders per month: 0.8 per customer
- Mean unit price: $25.00
- Use for: Testing retention alerts, churn prediction models, win-back campaigns

**3. PRODUCT_RECALL_SCENARIO** - Sudden drop in activity
```python
from customer_base_audit.synthetic import PRODUCT_RECALL_SCENARIO

transactions = generate_transactions(
    customers,
    date(2024, 1, 1),
    date(2024, 12, 31),
    scenario=PRODUCT_RECALL_SCENARIO
)
```

**Parameters:**
- Recall month: June (month 6)
- Activity multiplier: 0.3 (70% drop in orders - 30% of normal volume)
- Churn: 15% monthly (elevated after recall)
- Use for: Testing anomaly detection, revenue drop handling, crisis recovery

**4. HEAVY_PROMOTION_SCENARIO** - Black Friday / Holiday spike
```python
from customer_base_audit.synthetic import HEAVY_PROMOTION_SCENARIO

transactions = generate_transactions(
    customers,
    date(2024, 1, 1),
    date(2024, 12, 31),
    scenario=HEAVY_PROMOTION_SCENARIO
)
```

**Parameters:**
- Promo month: November (Black Friday)
- Promo uplift: 3.0x (3x normal order volume)
- Quantity per order: 2.0 (customers buy more)
- Churn: 5% monthly (low during promotion)
- Use for: Testing seasonality handling, promotion ROI analysis, capacity planning

**5. PRODUCT_LAUNCH_SCENARIO** - Gradual ramp-up
```python
from customer_base_audit.synthetic import PRODUCT_LAUNCH_SCENARIO

transactions = generate_transactions(
    customers,
    date(2023, 1, 1),
    date(2023, 12, 31),
    scenario=PRODUCT_LAUNCH_SCENARIO  # Launches March 15, 2023
)
```

**Parameters:**
- Launch date: March 15, 2023
- Post-launch uplift: Gradual increase (5% per month, capping at 75%)
- Mean unit price: $45.00 (premium product)
- Use for: Testing new product launches, ramp-up forecasting, early adoption patterns

**6. SEASONAL_BUSINESS_SCENARIO** - December peak season
```python
from customer_base_audit.synthetic import SEASONAL_BUSINESS_SCENARIO

transactions = generate_transactions(
    customers,
    date(2024, 1, 1),
    date(2024, 12, 31),
    scenario=SEASONAL_BUSINESS_SCENARIO
)
```

**Parameters:**
- Peak month: December
- Promo uplift: 2.5x
- Base orders per month: 0.9 (lower baseline, compensated by peak)
- Churn: 12% monthly (moderate - seasonal customers)
- Use for: Testing holiday businesses (gifts, tax prep, seasonal retail)

**7. STABLE_BUSINESS_SCENARIO** - Mature, low-churn business
```python
from customer_base_audit.synthetic import STABLE_BUSINESS_SCENARIO

transactions = generate_transactions(
    customers,
    date(2024, 1, 1),
    date(2024, 12, 31),
    scenario=STABLE_BUSINESS_SCENARIO
)
```

**Parameters:**
- Churn: 4% monthly (very low)
- Orders per month: 2.0 per customer (high repeat rate)
- Use for: Testing subscription businesses, SaaS models, high-retention verticals

### Advanced: Custom Scenarios

Create custom scenarios with ScenarioConfig:

```python
from customer_base_audit.synthetic import ScenarioConfig, generate_customers, generate_transactions
from datetime import date

# Define a custom "post-acquisition dip" scenario
custom_scenario = ScenarioConfig(
    promo_month=3,           # March slump after holiday season
    promo_uplift=0.6,        # 40% drop in orders
    churn_hazard=0.12,       # 12% monthly churn
    base_orders_per_month=1.5,
    mean_unit_price=35.0,
    price_variability=0.4,   # Moderate price variance
    quantity_mean=1.5,
    seed=999                 # Reproducible results
)

customers = generate_customers(300, date(2024, 1, 1), date(2024, 12, 31), seed=999)
transactions = generate_transactions(
    customers,
    date(2024, 1, 1),
    date(2024, 12, 31),
    scenario=custom_scenario
)

print(f"Custom scenario generated {len(transactions)} transactions")
```

**ScenarioConfig Parameters:**

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `promo_month` | int or None | Month (1-12) with modified activity | None |
| `promo_uplift` | float | Multiplier for promo month (>1 = spike, <1 = drop) | 1.5 |
| `launch_date` | date or None | Product launch date (gradual 5% per month ramp-up, capping at +75%) | None |
| `churn_hazard` | float | Monthly churn probability (0.0-1.0) | 0.08 |
| `base_orders_per_month` | float | Average orders per active customer (must be >= 0.0) | 1.2 |
| `mean_unit_price` | float | Average item price (must be > 0.0) | $30.00 |
| `price_variability` | float | Price variance coefficient (0.0 < value <= 1.0) | 0.4 |
| `quantity_mean` | float | Average quantity per order line (must be > 0.0) | 1.3 |
| `seed` | int or None | RNG seed for reproducibility (applies to transaction generation) | None |

**Validation Rules:**
- `promo_month` must be 1-12 or None
- `promo_uplift` must be > 0.0 (use < 1.0 for drops like product recalls)
- `churn_hazard` must be 0.0-1.0
- All monetary/quantity values must be positive
- `price_variability` must be in range (0.0, 1.0]
- Violations raise `ValueError` with descriptive error message

**Seed Behavior:**
- `ScenarioConfig.seed` controls transaction generation RNG
- `generate_customers(seed=X)` controls customer acquisition dates independently
- Use matching seeds for both to ensure full reproducibility

### Data Validation

The toolkit includes statistical validation functions to ensure generated data quality:

#### 1. Check Spend Distribution

Validates that transaction amounts follow realistic patterns:

```python
from customer_base_audit.synthetic import check_spend_distribution_is_realistic

result = check_spend_distribution_is_realistic(transactions)
print(f"✓ {result.message}" if result.ok else f"✗ {result.message}")
# Output: ✓ spend distribution realistic: mean=45.67, std=32.12, CV=0.70
```

**What it checks:**
- Mean transaction value is positive
- Coefficient of variation (CV) is 0.1-5.0 (typical for retail)
- Optional: Validate against expected mean/std with tolerance

**Use this when:** Testing synthetic generators, validating data transformations

#### 2. Check Cohort Decay Pattern

Validates that customer retention follows realistic decay:

```python
from customer_base_audit.synthetic import check_cohort_decay_pattern

result = check_cohort_decay_pattern(transactions, customers, max_expected_churn_rate=0.5)
print(f"✓ {result.message}" if result.ok else f"✗ {result.message}")
# Output: ✓ cohort decay patterns are realistic
```

**What it checks:**
- Retention doesn't increase unrealistically (>2x from one period to next)
- Cohorts show natural decay over time
- Small increases are OK (reactivations, seasonality)

**Use this when:** Validating cohort analysis, testing retention models

#### 3. Check No Duplicate Transactions

Ensures transaction uniqueness:

```python
from customer_base_audit.synthetic import check_no_duplicate_transactions

result = check_no_duplicate_transactions(transactions)
print(f"✓ {result.message}" if result.ok else f"✗ {result.message}")
# Output: ✓ all 15,234 transactions are unique (multi-line orders allowed)
```

**What it checks:**
- No exact duplicates (same order_id, timestamp, customer, product, etc.)
- Note: Multiple lines per order_id are valid (different products)

**Use this when:** Verifying data quality, testing deduplication logic

#### 4. Check Temporal Coverage

Validates transaction time spans:

```python
from customer_base_audit.synthetic import check_temporal_coverage

result = check_temporal_coverage(transactions, customers, min_months_with_activity=3)
print(f"✓ {result.message}" if result.ok else f"✗ {result.message}")
# Output: ✓ temporal coverage adequate: 12 months with activity
```

**What it checks:**
- Minimum months with transactions (default: 1)
- No transactions before customer acquisition dates
- Adequate temporal spread for time-series analysis

**Return behavior:**
- Returns `ValidationResult(ok=True, message="...")` on success
- Returns `ValidationResult(ok=False, message="...")` on failure (does NOT raise exceptions)
- Check `result.ok` to determine pass/fail status

**Use this when:** Validating data marts, testing cohort assignment

#### 5. Check Promo Spike Signal

Validates promotional spikes are detectable:

```python
from customer_base_audit.synthetic import check_promo_spike_signal

result = check_promo_spike_signal(transactions, promo_month=11, min_ratio=1.5)
print(f"✓ {result.message}" if result.ok else f"✗ {result.message}")
# Output: ✓ promo spike detected: ratio=2.87
```

**What it checks:**
- Promo month spend is >= `min_ratio` × average of other months
- Useful for verifying scenarios with promotions or recalls

**Use this when:** Testing anomaly detection, validating scenario configs

### Integration Testing Example

Complete example using synthetic data through the full pipeline:

```python
from datetime import date, datetime
from dataclasses import asdict
from customer_base_audit.synthetic import (
    generate_customers,
    generate_transactions,
    BASELINE_SCENARIO,
    check_spend_distribution_is_realistic,
    check_no_duplicate_transactions,
    check_temporal_coverage
)
from customer_base_audit.foundation.data_mart import CustomerDataMartBuilder, PeriodGranularity
from customer_base_audit.foundation.rfm import calculate_rfm, calculate_rfm_scores
from customer_base_audit.analyses.lens1 import analyze_single_period

# 1. Generate synthetic data
customers = generate_customers(1000, date(2024, 1, 1), date(2024, 12, 31), seed=42)
transactions = generate_transactions(
    customers,
    date(2024, 1, 1),
    date(2024, 12, 31),
    scenario=BASELINE_SCENARIO
)

# 2. Validate generated data
assert check_spend_distribution_is_realistic(transactions).ok
assert check_no_duplicate_transactions(transactions).ok
assert check_temporal_coverage(transactions, customers, min_months_with_activity=6).ok
print("✓ All validations passed")

# 3. Build data mart
builder = CustomerDataMartBuilder(period_granularities=[PeriodGranularity.MONTH])
mart = builder.build([asdict(t) for t in transactions])

# 4. Run analysis
period_aggregations = mart.periods[PeriodGranularity.MONTH]
rfm_metrics = calculate_rfm(period_aggregations, observation_end=datetime(2024, 12, 31, 23, 59, 59))
rfm_scores = calculate_rfm_scores(rfm_metrics)
lens1_results = analyze_single_period(rfm_metrics, rfm_scores)

# 5. Verify results
print(f"Total Customers: {lens1_results.total_customers}")
print(f"One-Time Buyers: {lens1_results.one_time_buyer_pct}%")
print(f"Total Revenue: ${lens1_results.total_revenue:,.2f}")

# Expected results with BASELINE_SCENARIO (seed=42):
# - Low one-time buyer percentage (good retention)
# - Moderate revenue concentration (healthy distribution)
# - Positive RFM distribution across all quintiles
```

### Testing Different Scenarios

Compare multiple scenarios to validate model robustness:

```python
from customer_base_audit.synthetic import (
    BASELINE_SCENARIO,
    HIGH_CHURN_SCENARIO,
    STABLE_BUSINESS_SCENARIO,
    generate_customers,
    generate_transactions
)

scenarios = {
    "Baseline": BASELINE_SCENARIO,
    "High Churn": HIGH_CHURN_SCENARIO,
    "Stable": STABLE_BUSINESS_SCENARIO
}

customers = generate_customers(500, date(2024, 1, 1), date(2024, 12, 31), seed=42)

for name, scenario in scenarios.items():
    txns = generate_transactions(customers, date(2024, 1, 1), date(2024, 12, 31), scenario=scenario)

    # Count active customers in final month
    final_month_customers = len(set(
        t.customer_id for t in txns
        if t.event_ts.year == 2024 and t.event_ts.month == 12
    ))

    print(f"{name:20} | {len(txns):6,} txns | {final_month_customers:4} active in Dec")

# Expected output (with seed=42, 500 customers):
# Baseline              | ~12,000 txns | ~280 active in Dec
# High Churn            | ~8,000 txns  | ~150 active in Dec
# Stable                | ~18,000 txns | ~420 active in Dec
#
# Note: Outputs are deterministic with fixed seeds but will vary with different
# customer counts or date ranges. Run this example to see exact values for your setup.
```

### Performance and Scale

**Generation Performance:**
- 1,000 customers × 12 months: < 1 second
- 10,000 customers × 12 months: 1-3 seconds
- 100,000 customers × 12 months: 10-30 seconds

**Memory Usage:**
- Memory-efficient generator (O(n) space where n = transaction count)
- Transactions stored as lightweight dataclass instances
- Typical usage: 10,000 transactions consume ~5-10 MB

**Validation Performance:**
- Most validators are O(n) complexity: spend distribution, duplicates, temporal coverage
- `check_cohort_decay_pattern` is O(n log n) due to sorting retention data by period
- 100,000 transactions validate in < 1 second

### Best Practices

**DO:**
- ✅ Use fixed seeds for reproducible tests
- ✅ Validate generated data before running analyses
- ✅ Test multiple scenarios to ensure model robustness
- ✅ Use scenario packs for common patterns
- ✅ Document which scenario you're using in test names

**DON'T:**
- ❌ Assume synthetic data perfectly matches your production patterns
- ❌ Use synthetic data to make real business decisions
- ❌ Use synthetic data for compliance/audit requirements (not real customer data)
- ❌ Skip validation checks (always run validators)
- ❌ Use random seeds in CI (breaks reproducibility)
- ❌ Generate excessive data (>100k customers) without need

### See Also

- **`tests/test_synthetic_data.py`**: 25+ test cases demonstrating all scenarios
- **`tests/test_integration_synthetic_pipeline.py`**: Full pipeline integration tests
- **`customer_base_audit/synthetic/generator.py`**: Implementation details
- **`customer_base_audit/synthetic/scenarios.py`**: Scenario definitions
- **`customer_base_audit/synthetic/validation.py`**: Validation functions

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
