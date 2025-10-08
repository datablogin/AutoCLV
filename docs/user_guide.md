# AutoCLV User Guide

## Version Compatibility
[To be filled in Phase 1]

## Installation and Setup
[To be filled in Phase 1]

## Configuration Options
[To be filled in Phase 2]

## Data Preparation Requirements
[To be filled in Phase 1]

## Running Five Lenses Audit

The Five Lenses framework from "The Customer-Base Audit" provides a comprehensive view of your customer base through five complementary analyses. Each lens answers specific questions about customer behavior and business health.

### Lens 1: Single Period Analysis

Lens 1 provides a snapshot view of your customer base within a single time period, answering fundamental questions:
- How many customers are there?
- What percentage are one-time buyers?
- How concentrated is revenue? (Pareto analysis)
- What is the distribution across RFM segments?

#### Running Lens 1 Analysis

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

# 4. Calculate RFM scores (optional but recommended)
rfm_scores = calculate_rfm_scores(rfm_metrics)

# 5. Run Lens 1 analysis
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
Median Customer Value: $3233.89
```

**Interpretation:** This Texas CLV dataset shows excellent customer retention (only 2.92% one-time buyers) with relatively distributed revenue (top 10% contribute 23%). The high average orders per customer (15.74) indicates strong repeat purchase behavior, typical of a successful subscription or high-engagement retail business.

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
[To be filled in Phase 1]

### Data Preparation Issues
[To be filled in Phase 2]

### Model Training Issues
[To be filled in Phase 3]

### Validation and Performance Issues
[To be filled in Phase 4]

### General Troubleshooting
[To be filled throughout]
