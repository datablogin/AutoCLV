# Testing Features with Synthetic Data

This guide explains how to test AutoCLV features using the comprehensive synthetic data generation toolkit. The synthetic data generator creates realistic customer transactions that simulate various business scenarios, enabling thorough testing without real customer data.

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Available Scenarios](#available-scenarios)
- [Step-by-Step Testing Workflow](#step-by-step-testing-workflow)
- [Complete Example: Multi-Scenario Testing](#complete-example-multi-scenario-testing)
- [Best Practices](#best-practices)
- [Common Patterns](#common-patterns)
- [Troubleshooting](#troubleshooting)

## Overview

### Why Use Synthetic Data?

- **No PII/Privacy Concerns**: Test with realistic data without customer privacy issues
- **Reproducible**: Same seed produces identical data for consistent testing
- **Diverse Scenarios**: Test edge cases (high churn, promotions, seasonality)
- **Scalable**: Generate 100 to 100,000+ customers as needed
- **Fast Iteration**: Generate new datasets in seconds

### What's Included

The synthetic data toolkit (`customer_base_audit/synthetic/`) provides:

1. **Customer Generation**: Create customers with acquisition dates
2. **Transaction Generation**: Generate purchase history with realistic patterns
3. **Pre-configured Scenarios**: 7 ready-to-use business scenarios
4. **Texas CLV Client**: Multi-city example with store openings and promotions

## Quick Start

### Basic Example

```python
from datetime import date
from customer_base_audit.synthetic import generate_customers, generate_transactions

# Generate 1000 customers acquired over 2 years
customers = generate_customers(
    n_customers=1000,
    start_date=date(2023, 1, 1),
    end_date=date(2024, 12, 31),
    seed=42  # For reproducibility
)

# Generate transactions for these customers
transactions = generate_transactions(
    customers,
    start=date(2023, 1, 1),
    end=date(2024, 12, 31),
    catalog=["SKU-A", "SKU-B", "SKU-C"],  # Optional product catalog
    seed=42
)

print(f"Generated {len(transactions)} transactions from {len(customers)} customers")
```

### Using Pre-configured Scenarios

```python
from customer_base_audit.synthetic.scenarios import HIGH_CHURN_SCENARIO

# Generate transactions with high churn behavior
transactions = generate_transactions(
    customers,
    start=date(2023, 1, 1),
    end=date(2024, 12, 31),
    scenario=HIGH_CHURN_SCENARIO  # 30% monthly churn
)
```

## Available Scenarios

Seven pre-configured scenarios simulate common business situations:

### 1. BASELINE_SCENARIO (Default)

**Use case**: General testing, moderate behavior

```python
from customer_base_audit.synthetic.scenarios import BASELINE_SCENARIO
```

**Characteristics**:
- Churn: 8% monthly
- Base orders: 1.2/month
- Mean price: $30
- Moderate variability

**Expected outcomes**: Balanced customer base, typical CLV distribution

---

### 2. HIGH_CHURN_SCENARIO

**Use case**: Test struggling business, retention strategies

```python
from customer_base_audit.synthetic.scenarios import HIGH_CHURN_SCENARIO
```

**Characteristics**:
- Churn: **30% monthly** (very high)
- Base orders: 0.8/month (low)
- Mean price: $25 (lower)
- High price variance

**Expected outcomes**:
- Many one-time buyers
- Low average CLV
- High percentage of zero-CLV customers (8-10%)
- Short customer lifetimes

---

### 3. STABLE_BUSINESS_SCENARIO

**Use case**: Test healthy business, benchmark best-case

```python
from customer_base_audit.synthetic.scenarios import STABLE_BUSINESS_SCENARIO
```

**Characteristics**:
- Churn: **4% monthly** (very low)
- Base orders: 2.0/month (high repeat rate)
- Mean price: $32
- Consistent pricing

**Expected outcomes**:
- High average CLV (100%+ above baseline)
- Few one-time buyers (1-2%)
- Long customer lifetimes
- Strong repeat purchase patterns

---

### 4. HEAVY_PROMOTION_SCENARIO

**Use case**: Test seasonal spikes (Black Friday, holidays)

```python
from customer_base_audit.synthetic.scenarios import HEAVY_PROMOTION_SCENARIO
```

**Characteristics**:
- Promo month: November
- Promo uplift: **3x normal volume**
- Churn: 5% (low during promo)
- Higher quantities per order (2.0 avg)

**Expected outcomes**:
- Highest transaction volumes
- Revenue spike in November
- Highest total CLV
- Temporary boost to retention

---

### 5. SEASONAL_BUSINESS_SCENARIO

**Use case**: Test seasonal peaks (retail, tourism)

```python
from customer_base_audit.synthetic.scenarios import SEASONAL_BUSINESS_SCENARIO
```

**Characteristics**:
- Promo month: December (peak season)
- Promo uplift: 2.5x
- Churn: 12% (seasonal customers)
- Lower baseline (0.9/month)

**Expected outcomes**:
- Strong December spike
- More sporadic purchase patterns
- Mid-range CLV
- Higher customer volatility

---

### 6. PRODUCT_RECALL_SCENARIO

**Use case**: Test crisis management, recovery

```python
from customer_base_audit.synthetic.scenarios import PRODUCT_RECALL_SCENARIO
```

**Characteristics**:
- Recall month: June
- Order drop: **70% decrease** (0.3x uplift)
- Elevated churn: 15%
- Otherwise healthy (1.5 orders/month baseline)

**Expected outcomes**:
- Sharp June revenue drop
- Increased churn post-recall
- Recovery in subsequent months
- Lost customer lifetime value

---

### 7. PRODUCT_LAUNCH_SCENARIO

**Use case**: Test new product ramp-up

```python
from customer_base_audit.synthetic.scenarios import PRODUCT_LAUNCH_SCENARIO
```

**Characteristics**:
- Launch date: March 15, 2023
- Gradual ramp-up after launch
- Low churn: 6%
- Premium pricing: $45 mean
- Starts slow (0.5 orders/month)

**Expected outcomes**:
- No transactions before launch date
- Increasing volumes post-launch
- Higher price points
- Growing customer base

---

## Step-by-Step Testing Workflow

### Step 1: Define Test Objective

**Example**: Test CLV Calculator across different business conditions

```python
# What are you testing?
# - Does CLV correctly reflect churn rates?
# - How do promotional periods affect predictions?
# - Are edge cases (one-time buyers) handled correctly?
```

### Step 2: Select Appropriate Scenarios

```python
scenarios_to_test = [
    ("Baseline", BASELINE_SCENARIO),
    ("High Churn", HIGH_CHURN_SCENARIO),
    ("Stable Business", STABLE_BUSINESS_SCENARIO),
]
```

**Tip**: Start with 2-3 contrasting scenarios (e.g., High Churn vs Stable) to validate expected differences.

### Step 3: Generate Synthetic Data

```python
from datetime import date
from customer_base_audit.synthetic import generate_customers, generate_transactions

def generate_test_data(scenario_config, n_customers=300):
    """Generate synthetic data for a scenario."""
    # Customer acquisition over 2 years
    customers = generate_customers(
        n_customers,
        start_date=date(2023, 1, 1),
        end_date=date(2024, 12, 31),
        seed=scenario_config.seed  # Use scenario's seed for reproducibility
    )

    # Transaction history
    transactions = generate_transactions(
        customers,
        start=date(2023, 1, 1),
        end=date(2024, 12, 31),
        catalog=["SKU-A", "SKU-B", "SKU-C", "SKU-D", "SKU-E"],
        scenario=scenario_config
    )

    return customers, transactions
```

### Step 4: Prepare Data for Your Feature

**Example**: Convert transactions to BG/NBD and Gamma-Gamma inputs

```python
from customer_base_audit.foundation.data_mart import CustomerDataMartBuilder, PeriodGranularity
from customer_base_audit.models.model_prep import prepare_bg_nbd_inputs, prepare_gamma_gamma_inputs
from datetime import datetime

def prepare_model_inputs(transactions):
    """Aggregate transactions into model inputs."""
    # Build data mart from transactions
    builder = CustomerDataMartBuilder([PeriodGranularity.MONTH])
    mart = builder.build([t.__dict__ for t in transactions])

    # Prepare BG/NBD inputs
    observation_start = datetime(2023, 1, 1)
    observation_end = datetime(2024, 12, 31, 23, 59, 59)

    bgnbd_data = prepare_bg_nbd_inputs(
        period_aggregations=mart.periods[PeriodGranularity.MONTH],
        observation_start=observation_start,
        observation_end=observation_end
    )

    # Prepare Gamma-Gamma inputs
    gg_data = prepare_gamma_gamma_inputs(
        period_aggregations=mart.periods[PeriodGranularity.MONTH],
        min_frequency=2
    )
    # Convert monetary_value from Decimal to float for model fitting
    gg_data['monetary_value'] = gg_data['monetary_value'].astype(float)

    return bgnbd_data, gg_data
```

### Step 5: Run Your Feature

```python
from customer_base_audit.models.clv_calculator import CLVCalculator

def test_feature(scenario_name, scenario_config):
    """Test feature with a scenario."""
    print(f"Testing {scenario_name}...")

    # Generate data
    customers, transactions = generate_test_data(scenario_config)

    # Prepare inputs
    bg_nbd_data = prepare_model_inputs(transactions)

    # Run feature
    # ... train models, calculate CLV, etc.

    # Collect metrics
    return {
        "scenario": scenario_name,
        "total_transactions": len(transactions),
        "avg_clv": result["clv"].mean(),
        # ... other metrics
    }
```

### Step 6: Validate Results

```python
def validate_results(results):
    """Validate that results align with expected behavior."""
    # High churn should have lower CLV than stable
    high_churn = next(r for r in results if r["scenario"] == "High Churn")
    stable = next(r for r in results if r["scenario"] == "Stable Business")

    assert stable["avg_clv"] > high_churn["avg_clv"], \
        "Stable business should have higher CLV than high churn"

    print("✓ High churn has lower CLV than stable business")

    # Heavy promotion should have more transactions
    promotion = next(r for r in results if r["scenario"] == "Heavy Promotion")
    baseline = next(r for r in results if r["scenario"] == "Baseline")

    assert promotion["total_transactions"] > baseline["total_transactions"], \
        "Promotion should generate more transactions"

    print("✓ Promotion scenario has higher transaction volume")
```

## Complete Example: Multi-Scenario Testing

See `examples/clv_scenario_comparison.py` for a full working example that:

1. Tests CLV Calculator across 5 scenarios
2. Generates 300 customers per scenario
3. Trains BG/NBD and Gamma-Gamma models
4. Calculates and compares CLV scores
5. Validates expected business outcomes

**Key findings from example**:
- High Churn reduces CLV by **61.2%** vs baseline
- Stable Business increases CLV by **100.2%** vs baseline
- Heavy Promotion has highest CLV (**$787** avg)

**To run**:
```bash
python examples/clv_scenario_comparison.py
```

## Best Practices

### 1. Use Consistent Seeds for Reproducibility

```python
# Good: Reproducible test
customers = generate_customers(100, start, end, seed=42)
transactions = generate_transactions(customers, start, end, seed=42)

# Bad: Non-reproducible test (random seed each time)
customers = generate_customers(100, start, end)  # seed varies
```

### 2. Start Small, Then Scale

```python
# Development: 100-300 customers (fast iteration)
customers = generate_customers(100, start, end, seed=42)

# Testing: 1,000 customers (realistic scale)
customers = generate_customers(1000, start, end, seed=42)

# Stress testing: 10,000+ customers (performance validation)
customers = generate_customers(10000, start, end, seed=42)
```

### 3. Test Contrasting Scenarios

Always include at least one "good" and one "bad" scenario:

```python
contrasting_scenarios = [
    ("Best Case", STABLE_BUSINESS_SCENARIO),    # Low churn
    ("Worst Case", HIGH_CHURN_SCENARIO),        # High churn
    ("Realistic", BASELINE_SCENARIO),           # Moderate
]
```

### 4. Validate Edge Cases

```python
# Check for edge cases in results
zero_clv_customers = results[results["clv"] == 0]
print(f"One-time buyers: {len(zero_clv_customers)} ({len(zero_clv_customers)/len(results)*100:.1f}%)")

# Expected: High churn has more zero-CLV customers than stable
assert high_churn_zero_pct > stable_zero_pct
```

### 5. Document Expected Outcomes

```python
def test_clv_with_high_churn():
    """
    Test CLV Calculator with HIGH_CHURN_SCENARIO.

    Expected outcomes:
    - Average CLV should be 40-70% lower than baseline
    - 8-10% of customers should have zero CLV (one-time buyers)
    - Average P(alive) should be lower than baseline
    - Predicted purchases should be reduced
    """
    # ... test implementation
```

### 6. Compare Relative Differences, Not Absolute Values

```python
# Good: Compare relative differences
churn_impact_pct = (baseline_clv - high_churn_clv) / baseline_clv * 100
assert churn_impact_pct > 50, "High churn should reduce CLV by >50%"

# Avoid: Hardcoded absolute values (fragile)
assert high_churn_clv == 133.80  # Breaks if model changes
```

## Common Patterns

### Pattern 1: Scenario Comparison Loop

```python
scenarios = [
    ("Baseline", BASELINE_SCENARIO),
    ("High Churn", HIGH_CHURN_SCENARIO),
    ("Stable", STABLE_BUSINESS_SCENARIO),
]

results = []
for scenario_name, scenario_config in scenarios:
    customers, transactions = generate_test_data(scenario_config)
    result = run_analysis(customers, transactions)
    result["scenario"] = scenario_name
    results.append(result)

# Compare results
comparison_df = pd.DataFrame(results)
comparison_df = comparison_df.sort_values("avg_metric", ascending=False)
print(comparison_df)
```

### Pattern 2: Time-Series Testing

```python
# Test behavior over time windows
def test_by_time_window(transactions):
    """Test quarterly performance."""
    quarters = [
        (date(2023, 1, 1), date(2023, 3, 31), "Q1 2023"),
        (date(2023, 4, 1), date(2023, 6, 30), "Q2 2023"),
        (date(2023, 7, 1), date(2023, 9, 30), "Q3 2023"),
        (date(2023, 10, 1), date(2023, 12, 31), "Q4 2023"),
    ]

    for start, end, label in quarters:
        quarter_txns = [t for t in transactions if start <= t.event_ts.date() <= end]
        revenue = sum(t.quantity * t.unit_price for t in quarter_txns)
        print(f"{label}: {len(quarter_txns)} txns, ${revenue:,.2f}")
```

### Pattern 3: Cohort Analysis

```python
from customer_base_audit.foundation.cohorts import create_monthly_cohorts, assign_cohorts

# Create monthly cohorts
cohorts = create_monthly_cohorts(
    customers,
    start_date=date(2023, 1, 1),
    end_date=date(2024, 12, 31)
)

# Assign customers to cohorts
cohort_assignments = assign_cohorts(customers, cohorts)

# Analyze by cohort
for cohort in cohorts:
    cohort_customers = [c for c in customers if cohort_assignments[c.customer_id] == cohort.cohort_id]
    print(f"Cohort {cohort.cohort_id}: {len(cohort_customers)} customers")
```

### Pattern 4: Data Quality Validation

```python
def validate_synthetic_data(customers, transactions):
    """Validate synthetic data quality before testing."""
    # No duplicate customer IDs
    assert len(customers) == len(set(c.customer_id for c in customers))

    # All transactions have valid customer IDs
    customer_ids = {c.customer_id for c in customers}
    for txn in transactions:
        assert txn.customer_id in customer_ids

    # No transactions before acquisition
    for txn in transactions:
        customer = next(c for c in customers if c.customer_id == txn.customer_id)
        txn_date = txn.event_ts.date() if hasattr(txn.event_ts, "date") else txn.event_ts
        assert txn_date >= customer.acquisition_date

    print("✓ Data validation passed")
```

## Troubleshooting

### Issue: "T must be positive" Error

**Cause**: Customer acquired on observation end date (T = 0)

**Solution**: Filter out customers with T <= 0

```python
# Skip customers acquired on observation end date
if T <= 0:
    continue
```

### Issue: "Gamma-Gamma requires frequency >= 2"

**Cause**: Too many one-time buyers (common in HIGH_CHURN_SCENARIO)

**Solution**: This is expected behavior. Document the percentage:

```python
one_time_buyers = data[data["frequency"] < 2]
print(f"One-time buyers: {len(one_time_buyers)} ({len(one_time_buyers)/len(data)*100:.1f}%)")
```

### Issue: Inconsistent Results Across Runs

**Cause**: Not using consistent seeds

**Solution**: Always specify seed parameter:

```python
# Use scenario's built-in seed
customers = generate_customers(1000, start, end, seed=scenario_config.seed)
```

### Issue: Model Training Fails

**Cause**: Insufficient data (too few customers or too few repeat purchasers)

**Solution**: Generate more customers or use a scenario with higher repeat rates:

```python
# Instead of 100 customers, use 300+
customers = generate_customers(300, start, end, seed=42)

# Or use a scenario with higher repeat rates
transactions = generate_transactions(
    customers, start, end, scenario=STABLE_BUSINESS_SCENARIO
)
```

### Issue: Unrealistic Transaction Volumes

**Cause**: Scenario parameters may need adjustment

**Solution**: Create custom scenario config:

```python
from customer_base_audit.synthetic.generator import ScenarioConfig

custom_scenario = ScenarioConfig(
    base_orders_per_month=1.5,  # Adjust as needed
    churn_hazard=0.10,
    mean_unit_price=35.0,
    seed=42
)
```

## Advanced Usage

### Custom Scenario Configuration

```python
from customer_base_audit.synthetic.generator import ScenarioConfig

# Create custom scenario
custom_scenario = ScenarioConfig(
    promo_month=3,              # March promotion
    promo_uplift=2.0,           # 2x order volume
    launch_date=date(2023, 6, 1),  # Product launch
    churn_hazard=0.07,          # 7% monthly churn
    base_orders_per_month=1.8,  # High baseline
    mean_unit_price=40.0,       # Premium pricing
    price_variability=0.3,      # Low variance (consistent)
    quantity_mean=1.5,          # 1.5 items per order
    seed=99                     # Custom seed
)
```

### Multi-City Testing (Texas CLV Pattern)

```python
from customer_base_audit.synthetic.texas_clv_client import generate_texas_clv_client

# Generate multi-city synthetic data
customers, transactions, city_map = generate_texas_clv_client(
    total_customers=1000,
    seed=42
)

# City distribution
cities = set(city_map.values())
for city in cities:
    city_customers = [cid for cid, c in city_map.items() if c == city]
    print(f"{city}: {len(city_customers)} customers")
```

## Summary Checklist

Before running your test:

- [ ] Select 2-3 contrasting scenarios
- [ ] Use consistent seeds for reproducibility
- [ ] Start with 100-300 customers for development
- [ ] Validate synthetic data quality
- [ ] Document expected outcomes
- [ ] Compare relative differences, not absolute values
- [ ] Test edge cases (one-time buyers, zero CLV)
- [ ] Validate results align with business logic

## Additional Resources

- **Examples Directory**: See `examples/` for working demonstrations
  - `clv_demo.py`: Basic CLV calculation with Texas CLV data
  - `clv_scenario_comparison.py`: Multi-scenario validation test
  - `bg_nbd_demo.py`: BG/NBD + Gamma-Gamma model demo

- **Synthetic Data Module**: `customer_base_audit/synthetic/`
  - `generator.py`: Core generation functions
  - `scenarios.py`: Pre-configured scenarios
  - `texas_clv_client.py`: Multi-city example

- **Tests**: `tests/test_synthetic_data.py`
  - Validation examples
  - Scenario testing patterns
  - Data quality checks

---

**Questions or Issues?**

If you encounter problems or have suggestions for improving this guide, please create an issue in the repository.
