# AutoCLV API Reference

Comprehensive API documentation for the AutoCLV toolkit. This reference covers all public modules, classes, and functions for customer lifetime value analysis and customer base audits.

## Table of Contents

- [Foundation Modules](#foundation-modules)
  - [RFM Analysis](#rfm-analysis)
  - [Data Mart](#data-mart)
  - [Cohorts](#cohorts)
- [Analysis Modules (Five Lenses)](#analysis-modules-five-lenses)
  - [Lens 1: Single Period Analysis](#lens-1-single-period-analysis)
  - [Lens 2: Period-to-Period Comparison](#lens-2-period-to-period-comparison)
  - [Lens 3: Cohort Evolution](#lens-3-cohort-evolution)
- [CLV Models](#clv-models)
  - [BG/NBD Model](#bgnbd-model)
  - [Gamma-Gamma Model](#gamma-gamma-model)
  - [CLV Calculator](#clv-calculator)
  - [Model Preparation](#model-preparation)
- [Synthetic Data](#synthetic-data)
  - [Data Generation](#data-generation)
  - [Validation](#validation)
- [Validation Framework](#validation-framework)

---

## Foundation Modules

### RFM Analysis

`customer_base_audit.foundation.rfm`

RFM (Recency, Frequency, Monetary) analysis is foundational for customer segmentation and CLV modeling. These functions calculate RFM metrics from transaction data.

#### `RFMMetrics`

**Dataclass** representing RFM metrics for a single customer.

**Attributes:**
- `customer_id` (str): Unique customer identifier
- `recency_days` (int): Days since last purchase (from observation_end)
- `frequency` (int): Total number of purchases in observation period
- `monetary` (Decimal): Average transaction value (total_spend / frequency)
- `observation_start` (datetime): Start date of observation period
- `observation_end` (datetime): End date of observation period
- `total_spend` (Decimal): Total spend in observation period

**Validation:** Automatically validates that recency >= 0, frequency > 0, monetary >= 0, and monetary = total_spend / frequency.

#### `calculate_rfm(period_aggregations, observation_end)`

Calculate RFM metrics from period aggregations.

**Parameters:**
- `period_aggregations` (Sequence[PeriodAggregation]): Customer period data
- `observation_end` (datetime): End of observation period

**Returns:**
- `list[RFMMetrics]`: RFM metrics for each customer

**Raises:**
- `ValueError`: If `period_aggregations` is empty
- `ValueError`: If `observation_end` is before the earliest transaction date
- `TypeError`: If datetime objects have mismatched timezones

**Example:**
```python
from datetime import datetime
from customer_base_audit.foundation.rfm import calculate_rfm
from customer_base_audit.foundation.data_mart import CustomerDataMartBuilder, PeriodGranularity

# Build data mart from transactions
builder = CustomerDataMartBuilder([PeriodGranularity.MONTH])
mart = builder.build([t.__dict__ for t in transactions])

# Calculate RFM metrics
rfm_metrics = calculate_rfm(
    period_aggregations=mart.periods[PeriodGranularity.MONTH],
    observation_end=datetime(2024, 12, 31, 23, 59, 59)
)

# Access metrics
for rfm in rfm_metrics[:5]:
    print(f"{rfm.customer_id}: R={rfm.recency_days}, F={rfm.frequency}, M=${rfm.monetary}")
```

**Note on Recency:** Uses actual last transaction timestamp when available for accurate CLV modeling. Falls back to period_end for backward compatibility.

**Timezone Handling:** All datetime values must use the same timezone (UTC recommended for production).

#### `RFMScore`

**Dataclass** representing RFM scores (1-5 scale) for a customer.

**Attributes:**
- `customer_id` (str): Unique customer identifier
- `r_score` (int): Recency score (1-5, where 5 = most recent)
- `f_score` (int): Frequency score (1-5, where 5 = most frequent)
- `m_score` (int): Monetary score (1-5, where 5 = highest spend)
- `rfm_segment` (str): Combined segment string (e.g., "555", "432")

#### `calculate_rfm_scores(rfm_metrics)`

Calculate quintile-based RFM scores (1-5 scale) from RFM metrics.

**Parameters:**
- `rfm_metrics` (Sequence[RFMMetrics]): RFM metrics to score

**Returns:**
- `list[RFMScore]`: RFM scores for each customer

**Scoring Logic:**
- **Recency**: Lower days = higher score (5 = most recent)
- **Frequency**: More purchases = higher score (5 = most frequent)
- **Monetary**: Higher spend = higher score (5 = highest value)

**Example:**
```python
from customer_base_audit.foundation.rfm import calculate_rfm_scores

# Calculate scores
rfm_scores = calculate_rfm_scores(rfm_metrics)

# Find best customers (555 segment)
best_customers = [s for s in rfm_scores if s.rfm_segment == "555"]
print(f"Best customers: {len(best_customers)}")

# Find at-risk customers (111 segment)
at_risk = [s for s in rfm_scores if s.rfm_segment == "111"]
print(f"At-risk customers: {len(at_risk)}")
```

---

### Data Mart

`customer_base_audit.foundation.data_mart`

The data mart aggregates transaction data by customer and time period, providing the foundation for RFM and cohort analyses.

#### `PeriodGranularity`

**Enum** defining time period granularities.

**Values:**
- `MONTH`: Monthly aggregation
- `QUARTER`: Quarterly aggregation
- `YEAR`: Yearly aggregation

#### `PeriodAggregation`

**Dataclass** representing customer activity in a time period.

**Key Attributes:**
- `customer_id` (str): Customer identifier
- `period_start` (datetime): Period start date
- `period_end` (datetime): Period end date
- `purchase_count` (int): Number of purchases in period
- `total_revenue` (Decimal): Total revenue in period
- `avg_order_value` (Decimal): Average order value
- `days_since_last_purchase` (int): Days since last purchase (from period_end)
- `customer_age_days` (int): Customer age in days
- `last_transaction_ts` (datetime | None): Actual timestamp of last transaction

#### `CustomerDataMartBuilder`

**Class** for building customer data marts from transactions.

**Constructor:**
```python
CustomerDataMartBuilder(period_granularities: list[PeriodGranularity])
```

**Methods:**

##### `build(transactions) -> CustomerDataMart`

Build a customer data mart from transaction dictionaries.

**Parameters:**
- `transactions` (list[dict]): Transaction records with fields:
  - `customer_id` (str): Required
  - `event_ts` (datetime): Required
  - `unit_price` (float): Required
  - `quantity` (int): Optional (default: 1)

**Returns:**
- `CustomerDataMart`: Contains orders and period aggregations

**Raises:**
- `ValueError`: If `transactions` is empty
- `KeyError`: If required fields (`customer_id`, `event_ts`, `unit_price`) are missing
- `TypeError`: If datetime objects have mismatched timezones

**Example:**
```python
from customer_base_audit.foundation.data_mart import (
    CustomerDataMartBuilder,
    PeriodGranularity
)

# Build data mart with monthly and quarterly aggregations
builder = CustomerDataMartBuilder(
    period_granularities=[PeriodGranularity.MONTH, PeriodGranularity.QUARTER]
)
mart = builder.build(transactions)

# Access monthly aggregations
monthly_data = mart.periods[PeriodGranularity.MONTH]
print(f"Monthly periods: {len(monthly_data)}")

# Access quarterly aggregations
quarterly_data = mart.periods[PeriodGranularity.QUARTER]
print(f"Quarterly periods: {len(quarterly_data)}")
```

---

### Cohorts

`customer_base_audit.foundation.cohorts`

Cohort analysis groups customers by acquisition date for tracking behavior over time.

#### `CohortDefinition`

**Dataclass** defining a customer cohort.

**Attributes:**
- `cohort_id` (str): Unique cohort identifier (e.g., "2024-01")
- `period_start` (datetime): Cohort acquisition period start
- `period_end` (datetime): Cohort acquisition period end
- `granularity` (PeriodGranularity): Time granularity (MONTH, QUARTER, YEAR)

#### `create_monthly_cohorts(customers, start_date, end_date)`

Create monthly cohorts from customer acquisition dates.

**Parameters:**
- `customers` (Sequence[Customer]): Customer records with acquisition_date
- `start_date` (datetime): First cohort month
- `end_date` (datetime): Last cohort month

**Returns:**
- `list[CohortDefinition]`: Monthly cohort definitions

**Example:**
```python
from datetime import datetime
from customer_base_audit.foundation.cohorts import (
    create_monthly_cohorts,
    assign_cohorts
)

# Create monthly cohorts for 2024
cohort_defs = create_monthly_cohorts(
    customers=customers,
    start_date=datetime(2024, 1, 1),
    end_date=datetime(2024, 12, 31)
)
print(f"Created {len(cohort_defs)} monthly cohorts")

# Assign customers to cohorts
cohort_assignments = assign_cohorts(customers, cohort_defs)

# Get January 2024 cohort customers
jan_cohort_customers = [
    cust_id for cust_id, cohort_id in cohort_assignments.items()
    if cohort_id == "2024-01"
]
```

#### `assign_cohorts(customers, cohort_definitions)`

Assign customers to cohorts based on acquisition date.

**Parameters:**
- `customers` (Sequence[Customer]): Customer records
- `cohort_definitions` (Sequence[CohortDefinition]): Cohort definitions

**Returns:**
- `dict[str, str]`: Mapping of customer_id -> cohort_id

---

## Analysis Modules (Five Lenses)

The Five Lenses framework provides multiple perspectives on customer base health. AutoCLV implements Lenses 1-3.

### Lens 1: Single Period Analysis

`customer_base_audit.analyses.lens1`

Lens 1 provides a snapshot view of customer base health in a single time period.

#### `Lens1Metrics`

**Dataclass** containing Lens 1 analysis results.

**Key Metrics:**
- `total_customers` (int): Total active customers
- `total_revenue` (Decimal): Total revenue
- `one_time_buyers` (int): Customers with only 1 purchase
- `one_time_buyer_pct` (float): Percentage of one-time buyers
- `avg_orders_per_customer` (float): Average orders per customer
- `median_customer_value` (Decimal): Median customer lifetime value
- `top_10pct_revenue_contribution` (float): % of revenue from top 10% customers
- `top_20pct_revenue_contribution` (float): % of revenue from top 20% customers
- `rfm_distribution` (dict[str, int]): Distribution of RFM segments

#### `analyze_single_period(rfm_metrics, rfm_scores)`

Perform Lens 1 analysis on a single time period.

**Parameters:**
- `rfm_metrics` (Sequence[RFMMetrics]): RFM metrics for the period
- `rfm_scores` (Sequence[RFMScore]): RFM scores for the period

**Returns:**
- `Lens1Metrics`: Comprehensive snapshot metrics

**Example:**
```python
from customer_base_audit.analyses.lens1 import analyze_single_period
from customer_base_audit.foundation.rfm import calculate_rfm, calculate_rfm_scores

# Calculate RFM
rfm_metrics = calculate_rfm(period_aggregations, observation_end)
rfm_scores = calculate_rfm_scores(rfm_metrics)

# Run Lens 1 analysis
lens1 = analyze_single_period(rfm_metrics, rfm_scores)

print(f"Total Customers: {lens1.total_customers:,}")
print(f"One-Time Buyers: {lens1.one_time_buyer_pct:.1f}%")
print(f"Top 10% Revenue: {lens1.top_10pct_revenue_contribution:.1f}%")
print(f"Median Customer Value: ${lens1.median_customer_value:.2f}")
```

**Use Cases:**
- Quick health check of customer base
- Identify revenue concentration risks
- Measure customer engagement (one-time buyer %)
- Segment analysis by RFM scores

---

### Lens 2: Period-to-Period Comparison

`customer_base_audit.analyses.lens2`

Lens 2 compares two time periods to track customer migration and retention.

#### `CustomerMigration`

**Dataclass** tracking customer movement between periods.

**Attributes:**
- `retained` (set[str]): Customers active in both periods
- `churned` (set[str]): Active in period 1, inactive in period 2
- `new` (set[str]): First purchase in period 2
- `reactivated` (set[str]): Inactive in period 1, returned in period 2

#### `Lens2Metrics`

**Dataclass** containing Lens 2 analysis results.

**Key Metrics:**
- `migration` (CustomerMigration): Customer movement patterns
- `retention_rate` (float): % of period 1 customers retained
- `churn_rate` (float): % of period 1 customers churned
- `reactivation_rate` (float): % of inactive customers reactivated
- `customer_count_change` (int): Net change in active customers
- `revenue_change_pct` (float): % change in revenue
- `avg_order_value_change_pct` (float): % change in AOV

#### `analyze_period_comparison(period1_rfm, period2_rfm, all_customer_history)`

Compare two time periods to analyze customer migration and business trends.

**Parameters:**
- `period1_rfm` (Sequence[RFMMetrics]): RFM for first period
- `period2_rfm` (Sequence[RFMMetrics]): RFM for second period
- `all_customer_history` (Sequence[str]): All known customer IDs (for churn detection)

**Returns:**
- `Lens2Metrics`: Period comparison metrics

**Example:**
```python
from customer_base_audit.analyses.lens2 import analyze_period_comparison
from datetime import datetime

# Compare Q3 vs Q4 2024
q3_start = datetime(2024, 7, 1)
q3_end = datetime(2024, 9, 30, 23, 59, 59)
q4_start = datetime(2024, 10, 1)
q4_end = datetime(2024, 12, 31, 23, 59, 59)

# Calculate RFM for each period
q3_aggs = [agg for agg in all_aggs if q3_start <= agg.period_start <= q3_end]
q4_aggs = [agg for agg in all_aggs if q4_start <= agg.period_start <= q4_end]

q3_rfm = calculate_rfm(q3_aggs, q3_end)
q4_rfm = calculate_rfm(q4_aggs, q4_end)

all_customer_ids = list(set(c.customer_id for c in customers))

# Run Lens 2 analysis
lens2 = analyze_period_comparison(q3_rfm, q4_rfm, all_customer_ids)

print(f"Retention Rate: {lens2.retention_rate:.1f}%")
print(f"Churned: {len(lens2.migration.churned):,} customers")
print(f"New: {len(lens2.migration.new):,} customers")
print(f"Revenue Change: {lens2.revenue_change_pct:+.1f}%")
```

**Use Cases:**
- Track retention and churn trends
- Identify new customer acquisition effectiveness
- Monitor reactivation success
- Understand business growth drivers

---

### Lens 3: Cohort Evolution

`customer_base_audit.analyses.lens3`

Lens 3 tracks how a single cohort's behavior evolves over time since acquisition.

#### `CohortPeriodMetrics`

**Dataclass** representing cohort metrics for a single time period.

**Key Attributes:**
- `period_number` (int): Periods since acquisition (0 = acquisition period)
- `active_customers` (int): Customers active in this period
- `retention_rate` (float): % of cohort still active (cumulative)
- `avg_orders_per_customer` (float): Average orders per active customer
- `avg_revenue_per_customer` (Decimal): Average revenue per active customer
- `total_revenue` (Decimal): Total cohort revenue in this period

#### `Lens3Metrics`

**Dataclass** containing Lens 3 analysis results.

**Attributes:**
- `cohort_name` (str): Cohort identifier
- `cohort_size` (int): Initial cohort size
- `acquisition_date` (datetime): Cohort acquisition date
- `periods` (list[CohortPeriodMetrics]): Period-by-period metrics

#### `analyze_cohort_evolution(cohort_name, acquisition_date, period_aggregations, cohort_customer_ids)`

Analyze how a cohort evolves over time since acquisition.

**Parameters:**
- `cohort_name` (str): Cohort identifier
- `acquisition_date` (datetime): Cohort acquisition date
- `period_aggregations` (Sequence[PeriodAggregation]): All period data
- `cohort_customer_ids` (Sequence[str]): Customer IDs in this cohort

**Returns:**
- `Lens3Metrics`: Cohort evolution metrics

**Example:**
```python
from customer_base_audit.analyses.lens3 import (
    analyze_cohort_evolution,
    calculate_retention_curve
)
from customer_base_audit.foundation.cohorts import create_monthly_cohorts, assign_cohorts

# Create cohorts
cohort_defs = create_monthly_cohorts(customers, start_date, end_date)
cohort_assignments = assign_cohorts(customers, cohort_defs)

# Analyze January 2024 cohort
cohort_name = "2024-01"
cohort_customer_ids = [
    cust_id for cust_id, coh_id in cohort_assignments.items()
    if coh_id == cohort_name
]
cohort_def = next(c for c in cohort_defs if c.cohort_id == cohort_name)

lens3 = analyze_cohort_evolution(
    cohort_name=cohort_name,
    acquisition_date=cohort_def.period_start,
    period_aggregations=period_aggregations,
    cohort_customer_ids=cohort_customer_ids
)

print(f"Cohort: {lens3.cohort_name}")
print(f"Size: {lens3.cohort_size:,}")
print("\nPeriod-by-Period Metrics:")
for period in lens3.periods[:6]:
    print(f"  Period {period.period_number}: {period.retention_rate:.1%} retention")

# Calculate retention curve
retention_curve = calculate_retention_curve(lens3)
```

**Use Cases:**
- Measure cohort retention over time
- Identify healthy vs. declining cohorts
- Forecast future cohort behavior
- Compare cohort quality

#### `calculate_retention_curve(cohort_metrics)`

Calculate retention curve from cohort metrics.

**Parameters:**
- `cohort_metrics` (Lens3Metrics): Cohort evolution data

**Returns:**
- `Mapping[int, float]`: Map of period_number -> retention_rate

---

## CLV Models

### BG/NBD Model

`customer_base_audit.models.bg_nbd`

The BG/NBD (Beta-Geometric/Negative Binomial Distribution) model predicts customer purchase frequency and probability of being "alive" (not churned).

#### `BGNBDConfig`

**Dataclass** configuring BG/NBD model training.

**Attributes:**
- `method` (str): Fitting method - "map" (Maximum A Posteriori) or "mcmc" (Markov Chain Monte Carlo)
- `mcmc_samples` (int): MCMC samples (default: 1000, only for method="mcmc")
- `mcmc_tune` (int): MCMC tuning steps (default: 500)
- `random_seed` (int | None): Random seed for reproducibility

**Example:**
```python
from customer_base_audit.models.bg_nbd import BGNBDConfig

# Fast MAP estimation (recommended for >1000 customers)
config_map = BGNBDConfig(method="map")

# MCMC for uncertainty quantification (<1000 customers)
config_mcmc = BGNBDConfig(
    method="mcmc",
    mcmc_samples=2000,
    mcmc_tune=1000,
    random_seed=42
)
```

#### `BGNBDModelWrapper`

**Class** wrapping pymc-marketing BG/NBD model with AutoCLV conventions.

**Constructor:**
```python
BGNBDModelWrapper(config: BGNBDConfig)
```

**Methods:**

##### `fit(data)`

Fit BG/NBD model to customer data.

**Parameters:**
- `data` (pd.DataFrame): Customer data with columns:
  - `frequency` (int): Number of repeat purchases
  - `recency` (float): Time of last purchase
  - `T` (float): Customer age (observation period)

**Returns:**
- `self`: Fitted model

**Example:**
```python
from customer_base_audit.models.bg_nbd import BGNBDModelWrapper, BGNBDConfig
from customer_base_audit.models.model_prep import prepare_bg_nbd_inputs
from customer_base_audit.foundation.data_mart import CustomerDataMartBuilder, PeriodGranularity
from datetime import datetime

# Build data mart
builder = CustomerDataMartBuilder([PeriodGranularity.MONTH])
mart = builder.build([t.__dict__ for t in transactions])

# Prepare data for BG/NBD
bgnbd_data = prepare_bg_nbd_inputs(
    period_aggregations=mart.periods[PeriodGranularity.MONTH],
    observation_start=datetime(2024, 1, 1),
    observation_end=datetime(2024, 12, 31, 23, 59, 59)
)

# Train model
config = BGNBDConfig(method="map")
model = BGNBDModelWrapper(config)
model.fit(bgnbd_data)

print(f"Model parameters: {model.params_}")
```

##### `predict(data, t)`

Predict number of purchases in next t time periods.

**Parameters:**
- `data` (pd.DataFrame): Customer RFM data
- `t` (float): Time periods to predict (e.g., 90 days)

**Returns:**
- `pd.DataFrame`: Predictions with columns:
  - `customer_id` (str)
  - `predicted_purchases` (float)

##### `calculate_probability_alive(data)`

Calculate probability that each customer is still "alive" (not churned).

**Parameters:**
- `data` (pd.DataFrame): Customer RFM data

**Returns:**
- `pd.DataFrame`: Probabilities with columns:
  - `customer_id` (str)
  - `prob_alive` (float): Probability between 0 and 1

---

### Gamma-Gamma Model

`customer_base_audit.models.gamma_gamma`

The Gamma-Gamma model predicts average transaction value per customer.

#### `GammaGammaConfig`

**Dataclass** configuring Gamma-Gamma model training.

**Attributes:**
- `method` (str): Fitting method - "map" or "mcmc"
- `mcmc_samples` (int): MCMC samples (default: 1000)
- `mcmc_tune` (int): MCMC tuning steps (default: 500)
- `random_seed` (int | None): Random seed

#### `GammaGammaModelWrapper`

**Class** wrapping pymc-marketing Gamma-Gamma model.

**Constructor:**
```python
GammaGammaModelWrapper(config: GammaGammaConfig)
```

**Methods:**

##### `fit(data)`

Fit Gamma-Gamma model to customer data.

**Parameters:**
- `data` (pd.DataFrame): Customer data with columns:
  - `frequency` (int): Number of purchases (must be >= 2)
  - `monetary_value` (float): Average transaction value

**Returns:**
- `self`: Fitted model

**Important:** Gamma-Gamma model requires customers with 2+ purchases to estimate their "true" average spend.

**Example:**
```python
from customer_base_audit.models.gamma_gamma import GammaGammaModelWrapper, GammaGammaConfig

# Filter to customers with 2+ purchases
gamma_data = model_data[model_data['frequency'] >= 2].copy()

# Train model
gg_config = GammaGammaConfig(method="map")
gg_model = GammaGammaModelWrapper(gg_config)
gg_model.fit(gamma_data)

print(f"Trained on {len(gamma_data)} customers with 2+ purchases")
```

##### `predict_spend(data)`

Predict expected average transaction value per customer.

**Parameters:**
- `data` (pd.DataFrame): Customer data with frequency and monetary_value

**Returns:**
- `pd.DataFrame`: Predictions with columns:
  - `customer_id` (str)
  - `predicted_avg_spend` (float)

---

### CLV Calculator

`customer_base_audit.models.clv_calculator`

High-level CLV calculation combining BG/NBD and Gamma-Gamma models.

#### `CLVScore`

**Dataclass** containing CLV prediction for a customer.

**Attributes:**
- `customer_id` (str): Customer identifier
- `predicted_purchases` (float): Expected purchases in time period
- `predicted_avg_value` (Decimal): Expected average transaction value
- `clv` (Decimal): Customer lifetime value (purchases × value)
- `probability_alive` (float | None): Probability customer is active
- `time_period_days` (int): Prediction time horizon

#### `CLVCalculator`

**Class** for calculating CLV scores.

**Constructor:**
```python
CLVCalculator(bg_nbd_model, gamma_gamma_model, time_horizon_months=12, discount_rate=0.1, profit_margin=1.0)
```

**Parameters:**
- `bg_nbd_model` (BGNBDModelWrapper): Fitted BG/NBD model
- `gamma_gamma_model` (GammaGammaModelWrapper): Fitted Gamma-Gamma model
- `time_horizon_months` (int): Prediction horizon in months (default: 12)
- `discount_rate` (Decimal): Annual discount rate (default: 0.1 = 10%)
- `profit_margin` (Decimal): Profit margin multiplier (default: 1.0 = 100%)

**Methods:**

##### `calculate_clv(bg_nbd_data, gamma_gamma_data, include_confidence_intervals=False)`

Calculate CLV for all customers.

**Parameters:**
- `bg_nbd_data` (pd.DataFrame): DataFrame with columns [customer_id, frequency, recency, T]
- `gamma_gamma_data` (pd.DataFrame): DataFrame with columns [customer_id, frequency, monetary_value] for repeat customers
- `include_confidence_intervals` (bool): If True, calculate confidence intervals (requires MCMC models). Default: False. Not yet implemented.

**Returns:**
- `pd.DataFrame`: CLV predictions with columns:
  - `customer_id` (str)
  - `predicted_purchases` (float)
  - `predicted_avg_value` (float)
  - `prob_alive` (float)
  - `clv` (float)

**Example:**
```python
from customer_base_audit.models.clv_calculator import CLVCalculator

# Train models (see above examples)
bgnbd_model.fit(bgnbd_data)
gamma_gamma_model.fit(gg_data)

# Calculate 3-month CLV
calculator = CLVCalculator(
    bg_nbd_model=bgnbd_model,
    gamma_gamma_model=gamma_gamma_model,
    time_horizon_months=3,
    discount_rate=0.10
)
clv_df = calculator.calculate_clv(bgnbd_data, gg_data)

# Find top customers
top_10 = clv_df.nlargest(10, 'clv')
for _, row in top_10.iterrows():
    print(f"{row['customer_id']}: ${row['clv']:.2f} CLV")
```

---

### Model Preparation

`customer_base_audit.models.model_prep`

Utilities for preparing transaction data for CLV models.

#### `prepare_bg_nbd_inputs(period_aggregations, observation_start, observation_end)`

Prepare period aggregations for BG/NBD model input.

**Parameters:**
- `period_aggregations` (Sequence[PeriodAggregation]): List of period-level customer aggregations
- `observation_start` (datetime): Start date of observation period (used for validation)
- `observation_end` (datetime): End date of observation period

**Returns:**
- `pd.DataFrame`: BG/NBD model-ready data with columns:
  - `customer_id` (str)
  - `frequency` (int): Number of repeat purchases (total_orders - 1)
  - `recency` (float): Time from first purchase to last purchase (in days)
  - `T` (float): Customer age from first purchase to observation_end (in days)

**Example:**
```python
from customer_base_audit.models.model_prep import prepare_bg_nbd_inputs
from datetime import datetime

bgnbd_data = prepare_bg_nbd_inputs(
    period_aggregations=mart.periods[PeriodGranularity.MONTH],
    observation_start=datetime(2024, 1, 1),
    observation_end=datetime(2024, 12, 31, 23, 59, 59)
)

print(f"Prepared {len(bgnbd_data)} customers for BG/NBD modeling")
print(f"Repeat customers: {len(bgnbd_data[bgnbd_data['frequency'] > 0])}")
```

#### `prepare_gamma_gamma_inputs(period_aggregations, min_frequency)`

Prepare period aggregations for Gamma-Gamma model input.

**Parameters:**
- `period_aggregations` (Sequence[PeriodAggregation]): List of period-level customer aggregations
- `min_frequency` (int): Minimum number of transactions required (default: 2)

**Returns:**
- `pd.DataFrame`: Gamma-Gamma model-ready data with columns:
  - `customer_id` (str)
  - `frequency` (int): Number of purchases
  - `monetary_value` (Decimal): Average transaction value (use `.astype(float)` for model fitting)

**Example:**
```python
from customer_base_audit.models.model_prep import prepare_gamma_gamma_inputs

gg_data = prepare_gamma_gamma_inputs(
    period_aggregations=mart.periods[PeriodGranularity.MONTH],
    min_frequency=2
)

# Convert monetary_value from Decimal to float for model fitting
gg_data['monetary_value'] = gg_data['monetary_value'].astype(float)

print(f"Prepared {len(gg_data)} repeat customers for Gamma-Gamma modeling")
```

---

## Synthetic Data

### Data Generation

`customer_base_audit.synthetic.generator`

Generate realistic synthetic customer transaction data for testing and demos.

#### `Customer`

**Dataclass** representing a customer.

**Attributes:**
- `customer_id` (str): Unique identifier
- `acquisition_date` (date): Date customer was acquired

#### `Transaction`

**Dataclass** representing a transaction.

**Attributes:**
- `transaction_id` (str): Unique identifier
- `customer_id` (str): Customer who made purchase
- `transaction_date` (date): Transaction date
- `product_id` (str): Product purchased
- `unit_price` (Decimal): Price per unit
- `quantity` (int): Quantity purchased

#### `ScenarioConfig`

**Dataclass** configuring synthetic data generation.

**Key Parameters:**
- `base_monthly_churn` (float): Monthly churn rate (default: 0.05)
- `base_orders_per_month` (float): Average orders/month (default: 0.4)
- `promo_month` (int | None): Month with promotional spike
- `promo_uplift` (float): Promotion multiplier (default: 2.0)
- `launch_date` (date | None): Product launch date
- `seed` (int | None): Random seed for reproducibility

**Example:**
```python
from customer_base_audit.synthetic import ScenarioConfig, BASELINE_SCENARIO

# Use predefined baseline
config = BASELINE_SCENARIO

# Or customize
custom_config = ScenarioConfig(
    base_monthly_churn=0.10,  # 10% monthly churn
    promo_month=6,            # June promotion
    promo_uplift=3.0,         # 3x orders during promo
    seed=42                   # Reproducible
)
```

**Predefined Scenarios:**
- `BASELINE_SCENARIO`: Moderate, stable business
- `HIGH_CHURN_SCENARIO`: 30% monthly churn
- `PRODUCT_RECALL_SCENARIO`: 70% activity drop in June
- `HEAVY_PROMOTION_SCENARIO`: 5x activity boost in specific month
- `PRODUCT_LAUNCH_SCENARIO`: Gradual ramp-up after launch
- `SEASONAL_BUSINESS_SCENARIO`: Strong Q4 seasonality
- `STABLE_BUSINESS_SCENARIO`: Very low churn (2%)

#### `generate_customers(total_customers, start_date, end_date, seed)`

Generate customers with random acquisition dates.

**Parameters:**
- `total_customers` (int): Number of customers to generate
- `start_date` (date): Earliest acquisition date
- `end_date` (date): Latest acquisition date
- `seed` (int | None): Random seed

**Returns:**
- `list[Customer]`: Generated customers

#### `generate_transactions(customers, start_date, end_date, scenario)`

Generate transactions for customers based on a scenario.

**Parameters:**
- `customers` (Sequence[Customer]): Customers to generate transactions for
- `start_date` (date): Transaction period start
- `end_date` (date): Transaction period end
- `scenario` (ScenarioConfig): Scenario configuration

**Returns:**
- `list[Transaction]`: Generated transactions

**Example:**
```python
from datetime import date
from customer_base_audit.synthetic import (
    generate_customers,
    generate_transactions,
    BASELINE_SCENARIO
)

# Generate 1000 customers
customers = generate_customers(
    total_customers=1000,
    start_date=date(2024, 1, 1),
    end_date=date(2024, 12, 31),
    seed=42
)

# Generate transactions
transactions = generate_transactions(
    customers=customers,
    start_date=date(2024, 1, 1),
    end_date=date(2024, 12, 31),
    scenario=BASELINE_SCENARIO
)

print(f"Generated {len(transactions):,} transactions")
```

---

### Validation

`customer_base_audit.synthetic.validation`

Validate synthetic data quality.

#### `check_non_negative_amounts(transactions)`

Verify all transaction amounts are non-negative.

**Parameters:**
- `transactions` (Sequence[Transaction]): Transactions to validate

**Raises:**
- `ValueError`: If negative amounts found

#### `check_retention_rates(customers, transactions)`

Validate retention rates match expected patterns.

**Parameters:**
- `customers` (Sequence[Customer]): Customer list
- `transactions` (Sequence[Transaction]): Transaction list

**Returns:**
- `bool`: True if validation passes

**Example:**
```python
from customer_base_audit.synthetic.validation import (
    check_non_negative_amounts,
    check_retention_rates
)

# Validate generated data
check_non_negative_amounts(transactions)
check_retention_rates(customers, transactions)
print("✓ Synthetic data validation passed")
```

---

## Validation Framework

`customer_base_audit.validation.validation`

Production model validation and diagnostics.

#### `validate_model_assumptions(model_data, model_type)`

Validate data meets model assumptions.

**Parameters:**
- `model_data` (pd.DataFrame): Model input data
- `model_type` (str): "bgnbd" or "gamma_gamma"

**Returns:**
- `dict`: Validation results with warnings/errors

#### `calculate_prediction_intervals(model, data, confidence_level)`

Calculate prediction intervals for model outputs (MCMC only).

**Parameters:**
- `model` (BGNBDModelWrapper | GammaGammaModelWrapper): Fitted MCMC model
- `data` (pd.DataFrame): Customer data
- `confidence_level` (float): Confidence level (e.g., 0.95 for 95%)

**Returns:**
- `pd.DataFrame`: Predictions with confidence intervals

**Example:**
```python
from customer_base_audit.validation.validation import (
    validate_model_assumptions,
    calculate_prediction_intervals
)

# Validate assumptions before training
validation_results = validate_model_assumptions(model_data, model_type="bgnbd")
if validation_results['errors']:
    print("Errors:", validation_results['errors'])

# Get prediction intervals (MCMC only)
if config.method == "mcmc":
    intervals = calculate_prediction_intervals(
        model=bgnbd_model,
        data=model_data,
        confidence_level=0.95
    )
    print(intervals[['customer_id', 'lower_bound', 'predicted', 'upper_bound']].head())
```

---

## Quick Reference

### Common Workflows

#### 1. Basic RFM Analysis
```python
from customer_base_audit.foundation.data_mart import CustomerDataMartBuilder, PeriodGranularity
from customer_base_audit.foundation.rfm import calculate_rfm, calculate_rfm_scores
from datetime import datetime

# Build mart → Calculate RFM → Score customers
builder = CustomerDataMartBuilder([PeriodGranularity.MONTH])
mart = builder.build(transactions)
rfm = calculate_rfm(mart.periods[PeriodGranularity.MONTH], datetime(2024, 12, 31))
scores = calculate_rfm_scores(rfm)
```

#### 2. Five Lenses Analysis
```python
from customer_base_audit.analyses import lens1, lens2, lens3

# Lens 1: Snapshot
lens1_results = lens1.analyze_single_period(rfm, scores)

# Lens 2: Comparison
lens2_results = lens2.analyze_period_comparison(q1_rfm, q2_rfm, all_customers)

# Lens 3: Cohort
lens3_results = lens3.analyze_cohort_evolution(cohort_name, acq_date, aggs, cust_ids)
```

#### 3. CLV Prediction
```python
from customer_base_audit.models import BGNBDModelWrapper, GammaGammaModelWrapper, CLVCalculator
from customer_base_audit.models import BGNBDConfig, GammaGammaConfig
from customer_base_audit.models.model_prep import prepare_bg_nbd_inputs, prepare_gamma_gamma_inputs

# Prepare → Train → Predict
bgnbd_data = prepare_bg_nbd_inputs(mart.periods[PeriodGranularity.MONTH], start, end)
gg_data = prepare_gamma_gamma_inputs(mart.periods[PeriodGranularity.MONTH], min_frequency=2)
gg_data['monetary_value'] = gg_data['monetary_value'].astype(float)
bgnbd = BGNBDModelWrapper(BGNBDConfig(method="map")).fit(bgnbd_data)
gg = GammaGammaModelWrapper(GammaGammaConfig(method="map")).fit(gg_data)
calculator = CLVCalculator(bgnbd, gg, time_horizon_months=3)
clv_df = calculator.calculate_clv(bgnbd_data, gg_data)
```

#### 4. Synthetic Data Generation
```python
from customer_base_audit.synthetic import generate_customers, generate_transactions, BASELINE_SCENARIO
from datetime import date

# Generate → Validate → Use
customers = generate_customers(1000, date(2024,1,1), date(2024,12,31), seed=42)
txns = generate_transactions(customers, date(2024,1,1), date(2024,12,31), BASELINE_SCENARIO)
```

---

## Additional Resources

- **User Guide**: `docs/user_guide.md` - Comprehensive tutorials and best practices
- **Synthetic Data Guide**: `docs/TESTING_WITH_SYNTHETIC_DATA.md` - Generate realistic test data
- **Example Notebooks**: `examples/` directory
  - `01_texas_clv_walkthrough.ipynb` - Complete CLV workflow
  - `02_custom_cohorts.ipynb` - Advanced cohort analysis
  - `03_model_comparison.ipynb` - Comparing CLV approaches
  - `04_monitoring_drift.ipynb` - Model monitoring and drift detection

---

## Support

For questions, issues, or contributions:
- GitHub Issues: https://github.com/datablogin/AutoCLV/issues
- Documentation: https://github.com/datablogin/AutoCLV/tree/main/docs

---

*Last Updated: 2025-10-10*
