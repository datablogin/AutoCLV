# Model Validation Guide

**Comprehensive guide to validating CLV models in AutoCLV**

---

## Table of Contents

1. [Introduction](#introduction)
2. [Why Validation Matters](#why-validation-matters)
3. [Validation Metrics](#validation-metrics)
4. [Interpretation Guidelines](#interpretation-guidelines)
5. [Cross-Validation Methodology](#cross-validation-methodology)
6. [Model Comparison Framework](#model-comparison-framework)
7. [Diagnosing Poor Performance](#diagnosing-poor-performance)
8. [MCMC-Specific Diagnostics](#mcmc-specific-diagnostics)
9. [Practical Examples](#practical-examples)
10. [Best Practices](#best-practices)

---

## Introduction

Model validation is the process of assessing how well your CLV model predicts future customer behavior. A validated model gives you confidence that:
- Predictions are accurate enough for business decisions
- The model generalizes to new data (not just memorizing training data)
- You'll know when the model needs retraining

AutoCLV provides a comprehensive validation framework with industry-standard metrics and time-series cross-validation.

---

## Why Validation Matters

### Business Impact

**Without validation:**
- ❌ Marketing budgets allocated to wrong customers
- ❌ Revenue forecasts miss by 50%+
- ❌ High-value customers churn undetected
- ❌ Resources wasted on unprofitable segments

**With validation:**
- ✅ Confidence in customer lifetime value predictions
- ✅ Early detection of model degradation
- ✅ Data-driven decisions on when to retrain
- ✅ Quantifiable improvement from model iterations

### Real-World Scenarios

1. **E-commerce:** Predict next 90-day revenue per customer → allocate ad spend
2. **SaaS:** Identify customers likely to upgrade → prioritize sales outreach
3. **Retail:** Forecast holiday season revenue → optimize inventory
4. **Subscription:** Detect high-risk churners → trigger retention campaigns

In all cases, **model accuracy directly impacts ROI**.

---

## Validation Metrics

AutoCLV computes five key metrics to assess model performance:

### 1. MAE (Mean Absolute Error)

**Definition:** Average absolute difference between predicted and actual CLV.

```
MAE = mean(|actual - predicted|)
```

**Units:** Same as CLV (e.g., dollars)

**Example:**
```python
from customer_base_audit.validation import calculate_clv_metrics
import pandas as pd

actual = pd.Series([100, 150, 200, 50])
predicted = pd.Series([95, 160, 190, 55])

metrics = calculate_clv_metrics(actual, predicted)
print(f"MAE: ${metrics.mae}")  # MAE: $7.50
```

**Interpretation:**
- Lower is better
- Same units as CLV → easy to interpret
- **Not** affected by outliers as much as RMSE
- **Example:** MAE = $50 means predictions are off by $50 on average

**When to use:** General-purpose metric for reporting prediction error.

---

### 2. MAPE (Mean Absolute Percentage Error)

**Definition:** Average percentage error across customers (excludes zero actual values).

```
MAPE = mean(|actual - predicted| / |actual|) × 100%
```

**Units:** Percentage

**Example:**
```python
metrics = calculate_clv_metrics(actual, predicted)
print(f"MAPE: {metrics.mape}%")  # MAPE: 6.25%
```

**Interpretation:**
- Lower is better
- **Scale-independent** → compare models across different CLV ranges
- **Target:** MAPE < 20% (individual customer level)
- **Caveat:** Excludes customers with zero actual CLV (avoids division by zero)

**When to use:** Comparing models trained on different datasets or time periods.

**Warning:** Biased if many customers have zero actual CLV (e.g., high churn rates).

---

### 3. RMSE (Root Mean Squared Error)

**Definition:** Square root of average squared errors.

```
RMSE = sqrt(mean((actual - predicted)²))
```

**Units:** Same as CLV (e.g., dollars)

**Example:**
```python
metrics = calculate_clv_metrics(actual, predicted)
print(f"RMSE: ${metrics.rmse}")  # RMSE: $8.66
```

**Interpretation:**
- Lower is better
- **Penalizes large errors** more than MAE (squared term)
- Always ≥ MAE
- **Example:** RMSE = $80 means typical error is $80, with larger errors weighted heavily

**When to use:** When large errors are particularly costly (e.g., underestimating high-value customers).

---

### 4. ARPE (Aggregate Revenue Percent Error)

**Definition:** Percentage error at the aggregate level (total revenue).

```
ARPE = |sum(actual) - sum(predicted)| / sum(actual) × 100%
```

**Units:** Percentage

**Example:**
```python
metrics = calculate_clv_metrics(actual, predicted)
print(f"ARPE: {metrics.arpe}%")  # ARPE: 2.00%
```

**Interpretation:**
- Lower is better
- Measures **aggregate forecasting accuracy** (not individual customers)
- **Target:** ARPE < 10% (aggregate level)
- Individual errors can cancel out → low ARPE doesn't guarantee good individual predictions

**When to use:** Revenue forecasting, budget planning, inventory optimization.

**Key insight:** You can have high MAPE (poor individual predictions) but low ARPE (accurate total revenue). Both matter!

---

### 5. R² (Coefficient of Determination)

**Definition:** Proportion of variance explained by the model.

```
R² = 1 - (SS_residual / SS_total)
```

**Units:** Dimensionless (range: -∞ to 1.0)

**Example:**
```python
metrics = calculate_clv_metrics(actual, predicted)
print(f"R²: {metrics.r_squared}")  # R²: 0.95
```

**Interpretation:**
- **R² = 1.0:** Perfect predictions
- **R² = 0.5:** Model explains 50% of variance (better than mean baseline)
- **R² = 0.0:** Model equals a horizontal line at the mean
- **R² < 0.0:** Model worse than predicting the mean (red flag!)

**Target:** R² > 0.5 (model explains >50% of CLV variance)

**When to use:** Assessing overall model fit and comparing models.

**Warning:** R² can be negative! This indicates your model is worse than a naive mean baseline.

---

## Interpretation Guidelines

### What is "Good" Performance?

Performance targets depend on:
- **Data quality:** Clean data → better predictions
- **Business maturity:** Stable businesses → easier to model
- **Forecast horizon:** 30-day predictions → more accurate than 365-day
- **Customer heterogeneity:** Homogeneous segments → lower error

### General Targets

| Metric | Excellent | Good | Acceptable | Poor |
|--------|-----------|------|------------|------|
| **MAPE** | < 10% | 10-20% | 20-30% | > 30% |
| **ARPE** | < 5% | 5-10% | 10-20% | > 20% |
| **R²** | > 0.8 | 0.5-0.8 | 0.2-0.5 | < 0.2 |
| **MAE** | < 10% of mean CLV | 10-20% | 20-30% | > 30% |

### Context Matters

**Scenario 1: High-Frequency Business (coffee shop)**
- Short purchase cycles
- Lots of transaction history
- **Target:** MAPE < 15%, R² > 0.7

**Scenario 2: Low-Frequency Business (furniture)**
- Long purchase cycles
- Sparse transaction history
- **Target:** MAPE < 30%, R² > 0.4

**Scenario 3: Early-Stage Startup**
- Limited historical data
- Rapidly changing customer behavior
- **Target:** Focus on directional accuracy (rank ordering), not absolute error

### Red Flags

- ⚠️ **R² < 0:** Model is worse than predicting the mean (retrain or add features)
- ⚠️ **MAPE > 50%:** Predictions are essentially noise
- ⚠️ **ARPE < 10% but MAPE > 30%:** Individual errors cancel out (risky for targeting)
- ⚠️ **Increasing MAE over time:** Model drift detected (retrain needed)

---

## Cross-Validation Methodology

### Why Cross-Validation?

A single train/test split can be misleading:
- Model might get lucky on one test period
- Doesn't capture seasonal effects
- No sense of prediction stability

**Solution:** Time-series cross-validation with expanding windows.

### Expanding Window Approach

```
Fold 1: ████████████░░░ (Train: months 0-12, Test: 13-15)
Fold 2: ███████████████░░░ (Train: months 0-15, Test: 16-18)
Fold 3: ██████████████████░░░ (Train: months 0-18, Test: 19-21)
Fold 4: █████████████████████░░░ (Train: months 0-21, Test: 22-24)
```

**Why expanding?** Simulates production: you always retrain on all historical data.

### Implementation

```python
from customer_base_audit.validation import cross_validate_clv
import pandas as pd

# Define your model pipeline
def my_clv_pipeline(transactions, observation_end):
    """
    Takes transactions up to observation_end, returns CLV predictions.

    This function should:
    1. Build data mart
    2. Calculate RFM
    3. Train BG/NBD + Gamma-Gamma models
    4. Return DataFrame with columns: customer_id, clv
    """
    # ... your model training code ...
    return predictions_df  # Must have 'customer_id' and 'clv' columns

# Run cross-validation
metrics_list = cross_validate_clv(
    transactions=transactions,
    model_pipeline=my_clv_pipeline,
    n_folds=5,
    time_increment_months=3,
    initial_train_months=12,
    customer_id_col='customer_id',
    date_col='event_ts',
    clv_col='clv'
)

# Analyze results
print(f"Mean MAPE: {sum(m.mape for m in metrics_list) / len(metrics_list):.2f}%")
print(f"Mean R²: {sum(m.r_squared for m in metrics_list) / len(metrics_list):.3f}")
```

### Interpreting CV Results

**Stable performance across folds → Model is robust**
```
Fold 1: MAPE=15.2%, R²=0.75
Fold 2: MAPE=14.8%, R²=0.78
Fold 3: MAPE=16.1%, R²=0.72
Fold 4: MAPE=15.5%, R²=0.76
→ ✅ Consistent, trustworthy predictions
```

**Degrading performance over time → Model drift**
```
Fold 1: MAPE=15.0%, R²=0.75
Fold 2: MAPE=18.5%, R²=0.68
Fold 3: MAPE=24.2%, R²=0.55
Fold 4: MAPE=32.7%, R²=0.38
→ ⚠️ Customer behavior changing, retrain needed
```

---

## Model Comparison Framework

### Comparing Different Models

AutoCLV supports multiple modeling approaches:
1. **Historical Average:** Naive baseline (mean of past CLV)
2. **BG/NBD + Gamma-Gamma (MAP):** Fast, point estimates
3. **BG/NBD + Gamma-Gamma (MCMC):** Slower, uncertainty quantification

### Comparison Checklist

| Criterion | Historical Average | BG/NBD (MAP) | BG/NBD (MCMC) |
|-----------|-------------------|--------------|---------------|
| **Training Time** | Instant | ~5s (1K customers) | ~2min (1K customers) |
| **Prediction Time** | Instant | ~1s | ~1s |
| **Accuracy (MAPE)** | Baseline | Typically 10-30% better | Similar to MAP |
| **Uncertainty** | None | None | Credible intervals |
| **Use Case** | Baseline comparison | Production | Research, risk assessment |

### Example Comparison

```python
import pandas as pd
from customer_base_audit.validation import calculate_clv_metrics

# Collect predictions from all models
results = pd.DataFrame({
    'actual': actual_clv,
    'historical': historical_predictions,
    'map': map_predictions,
    'mcmc': mcmc_predictions
})

# Compare metrics
for model in ['historical', 'map', 'mcmc']:
    metrics = calculate_clv_metrics(results['actual'], results[model])
    print(f"\n{model.upper()} Model:")
    print(f"  MAPE: {metrics.mape:.2f}%")
    print(f"  ARPE: {metrics.arpe:.2f}%")
    print(f"  R²: {metrics.r_squared:.3f}")

# Example output:
# HISTORICAL Model:
#   MAPE: 32.50%
#   ARPE: 15.20%
#   R²: 0.420
#
# MAP Model:
#   MAPE: 18.30%    ← 44% improvement
#   ARPE: 8.10%     ← 47% improvement
#   R²: 0.710       ← 69% improvement
#
# MCMC Model:
#   MAPE: 18.50%    ← Similar to MAP
#   ARPE: 7.90%
#   R²: 0.715
```

### Statistical Significance Testing

When comparing models, use **paired t-test** on absolute errors:

```python
from scipy.stats import ttest_rel

errors_model_a = abs(actual - predictions_a)
errors_model_b = abs(actual - predictions_b)

t_stat, p_value = ttest_rel(errors_model_a, errors_model_b)

if p_value < 0.05:
    print(f"✅ Model B significantly better (p={p_value:.4f})")
else:
    print(f"⚠️ No significant difference (p={p_value:.4f})")
```

---

## Diagnosing Poor Performance

### Symptom: High MAPE (> 30%)

**Possible Causes:**
1. **Insufficient data:** < 100 customers or < 3 months history
2. **High customer heterogeneity:** Wide range of purchase behaviors
3. **Non-stationary behavior:** Trends, seasonality, regime changes
4. **Poor data quality:** Missing transactions, duplicate records

**Solutions:**
- Collect more data (especially repeat purchases)
- Segment customers before modeling (by product, geography, acquisition channel)
- Add seasonality features or use time-varying parameters
- Audit data pipeline for quality issues

---

### Symptom: Low R² (< 0.2)

**Possible Causes:**
1. **Model too simple:** BG/NBD assumes homogeneity
2. **Key features missing:** Need product, channel, or customer attributes
3. **Random customer behavior:** Impulse purchases, one-time buyers dominate

**Solutions:**
- Try hierarchical models (coming soon in AutoCLV)
- Add customer covariates (demographics, firmographics)
- Focus on repeat customers (filter out one-time buyers)
- Consider alternative models (Pareto/NBD for high churn)

---

### Symptom: Good ARPE but Poor MAPE

**Scenario:** ARPE = 8% (good) but MAPE = 35% (poor)

**Interpretation:** Aggregate revenue forecast is accurate, but individual customer predictions are noisy.

**Business Impact:**
- ✅ Revenue forecasting: Can rely on aggregate predictions
- ❌ Customer targeting: Cannot trust individual CLV scores

**Solutions:**
- Use decile-based targeting instead of absolute CLV thresholds
- Focus on rank ordering (top 10% vs bottom 10%) not absolute values
- Consider ensemble methods (average multiple models)

---

### Symptom: Negative R²

**Interpretation:** Your model is worse than predicting the mean for every customer.

**Common Causes:**
1. **Overfitting:** Model memorized training data
2. **Data leakage:** Test data contaminated training set
3. **Wrong model choice:** BG/NBD doesn't fit your business

**Solutions:**
- Check for data leakage (temporal train/test split correct?)
- Simplify model (fewer parameters, more regularization)
- Try alternative models (Gamma-Gamma might not fit your monetary distribution)

---

### Symptom: Increasing Error Over Time

**Example:**
```
Month 1: MAE = $45
Month 2: MAE = $52
Month 3: MAE = $61
Month 4: MAE = $75  ← Model degrading
```

**Interpretation:** Model drift detected (customer behavior changing).

**Causes:**
- Market changes (new competitors, economic downturn)
- Product changes (pricing, features, quality)
- Seasonal effects (holidays, back-to-school)

**Solutions:**
- Retrain model on recent data
- Use rolling windows (last 12 months only)
- Implement automated monitoring (see [Monitoring Notebook](../examples/04_monitoring_drift.ipynb))

---

## MCMC-Specific Diagnostics

When using `method='mcmc'`, additional diagnostics are available to assess MCMC convergence.

### R-hat (Gelman-Rubin Statistic)

**Purpose:** Detect if MCMC chains have converged to the same distribution.

```python
from customer_base_audit.validation.diagnostics import check_mcmc_convergence

diagnostics = check_mcmc_convergence(model.idata)
print(f"Converged: {diagnostics.converged}")
print(f"Max R-hat: {diagnostics.max_r_hat:.4f}")
```

**Interpretation:**
- **R-hat < 1.01:** Excellent convergence ✅
- **R-hat 1.01-1.05:** Good convergence ✅
- **R-hat 1.05-1.1:** Acceptable convergence ⚠️
- **R-hat > 1.1:** Poor convergence, increase draws/chains ❌

**Fix:** If R-hat > 1.1, increase `draws` and `tune` in `BGNBDConfig`:
```python
config = BGNBDConfig(method='mcmc', draws=2000, tune=2000, chains=4)
```

---

### ESS (Effective Sample Size)

**Purpose:** Measure number of independent samples (accounts for autocorrelation).

```python
print(f"Min ESS (bulk): {diagnostics.min_ess_bulk:.0f}")
print(f"Min ESS (tail): {diagnostics.min_ess_tail:.0f}")
```

**Interpretation:**
- **ESS > 400:** Sufficient for most applications ✅
- **ESS 100-400:** Marginal, consider more draws ⚠️
- **ESS < 100:** Insufficient, increase draws significantly ❌

**Fix:** If ESS < 400, increase `draws`:
```python
config = BGNBDConfig(method='mcmc', draws=3000, tune=2000)
```

---

### Posterior Predictive Checks

**Purpose:** Assess model fit by comparing observed data to model simulations.

```python
from customer_base_audit.validation.diagnostics import posterior_predictive_check

stats = posterior_predictive_check(observed_frequency, posterior_samples)
print(f"Coverage (95% CI): {stats.coverage_95:.2f}")
print(f"Observed mean: {stats.observed_mean:.2f}")
print(f"Predicted mean: {stats.predicted_mean:.2f}")
```

**Interpretation:**
- **Coverage ≈ 0.95:** Model well-calibrated ✅
- **Coverage < 0.80:** Model underfits (too confident) ❌
- **Coverage > 0.99:** Model overfits (too uncertain) ❌
- **Observed ≈ Predicted means:** No systematic bias ✅

---

## Practical Examples

### Example 1: Basic Validation with Texas CLV Data

```python
from datetime import datetime
from customer_base_audit.synthetic import generate_texas_clv_client
from customer_base_audit.models.model_prep import prepare_bg_nbd_inputs, prepare_gamma_gamma_inputs
from customer_base_audit.models.bg_nbd import BGNBDModelWrapper, BGNBDConfig
from customer_base_audit.models.gamma_gamma import GammaGammaModelWrapper, GammaGammaConfig
from customer_base_audit.models.clv_calculator import CLVCalculator
from customer_base_audit.validation import temporal_train_test_split, calculate_clv_metrics
import pandas as pd

# 1. Generate data
customers, transactions, city_map = generate_texas_clv_client(total_customers=1000, seed=42)

# Convert to DataFrame
txns_df = pd.DataFrame([{
    'customer_id': t.customer_id,
    'event_ts': t.transaction_date,
    'unit_price': t.unit_price,
    'quantity': t.quantity,
    'amount': t.unit_price * t.quantity
} for t in transactions])

# 2. Temporal split
train_txns, obs_txns, test_txns = temporal_train_test_split(
    txns_df,
    train_end_date=datetime(2024, 9, 1),
    observation_end_date=datetime(2024, 12, 31)
)

print(f"Training: {len(train_txns)} transactions")
print(f"Test: {len(test_txns)} transactions")

# 3. Build data mart from observation transactions
from customer_base_audit.foundation.data_mart import CustomerDataMartBuilder, PeriodGranularity
builder = CustomerDataMartBuilder([PeriodGranularity.MONTH])
mart = builder.build(obs_txns.to_dict('records'))

# 4. Prepare data for modeling (using period aggregations)
bgnbd_data = prepare_bg_nbd_inputs(
    period_aggregations=mart.periods[PeriodGranularity.MONTH],
    observation_start=datetime(2024, 1, 1),
    observation_end=datetime(2024, 12, 31, 23, 59, 59)
)

gg_data = prepare_gamma_gamma_inputs(
    period_aggregations=mart.periods[PeriodGranularity.MONTH],
    min_frequency=2
)
gg_data['monetary_value'] = gg_data['monetary_value'].astype(float)

# 5. Train BG/NBD model
bgnbd_model = BGNBDModelWrapper(BGNBDConfig(method='map'))
bgnbd_model.fit(bgnbd_data)

# 6. Train Gamma-Gamma model (repeat customers only)
gg_model = GammaGammaModelWrapper(GammaGammaConfig(method='map'))
gg_model.fit(gg_data)

# 7. Calculate CLV predictions
calculator = CLVCalculator(
    bg_nbd_model=bgnbd_model,
    gamma_gamma_model=gg_model,
    time_horizon_months=3  # Predict next 3 months (~90 days)
)
clv_df = calculator.calculate_clv(bgnbd_data, gg_data)

# 8. Calculate actual CLV from test period
actual_clv = test_txns.groupby('customer_id')['amount'].sum().reset_index()
actual_clv.columns = ['customer_id', 'actual_clv']

# 9. Merge predictions with actuals
comparison = clv_df[['customer_id', 'clv']].merge(actual_clv, on='customer_id', how='inner')
comparison.rename(columns={'clv': 'predicted_clv'}, inplace=True)

# 10. Calculate validation metrics
metrics = calculate_clv_metrics(
    actual=comparison['actual_clv'],
    predicted=comparison['predicted_clv']
)

print("\n📊 Validation Metrics:")
print(f"   MAE: ${metrics.mae}")
print(f"   MAPE: {metrics.mape}%")
print(f"   RMSE: ${metrics.rmse}")
print(f"   ARPE: {metrics.arpe}%")
print(f"   R²: {metrics.r_squared}")
print(f"   Sample Size: {metrics.sample_size} customers")

# Interpret results
if metrics.mape < 20 and metrics.arpe < 10:
    print("\n✅ Model performance: EXCELLENT")
elif metrics.mape < 30 and metrics.arpe < 20:
    print("\n✅ Model performance: GOOD")
else:
    print("\n⚠️  Model performance: NEEDS IMPROVEMENT")
```

---

### Example 2: Cross-Validation

```python
from customer_base_audit.validation import cross_validate_clv

# Define model pipeline
def clv_pipeline(transactions, observation_end):
    """Full CLV modeling pipeline."""
    # Build data mart
    builder = CustomerDataMartBuilder([PeriodGranularity.MONTH])
    mart = builder.build(transactions.to_dict('records'))

    # Prepare data
    observation_start = datetime(2024, 1, 1)
    bgnbd_data = prepare_bg_nbd_inputs(
        period_aggregations=mart.periods[PeriodGranularity.MONTH],
        observation_start=observation_start,
        observation_end=observation_end
    )

    gg_data = prepare_gamma_gamma_inputs(
        period_aggregations=mart.periods[PeriodGranularity.MONTH],
        min_frequency=2
    )
    gg_data['monetary_value'] = gg_data['monetary_value'].astype(float)

    # Train models
    bgnbd = BGNBDModelWrapper(BGNBDConfig(method='map'))
    bgnbd.fit(bgnbd_data)

    gg = GammaGammaModelWrapper(GammaGammaConfig(method='map'))
    gg.fit(gg_data)

    # Calculate CLV
    calc = CLVCalculator(bgnbd, gg, time_horizon_months=3)
    clv_df = calc.calculate_clv(bgnbd_data, gg_data)

    # Return predictions
    return clv_df[['customer_id', 'clv']]

# Run 5-fold cross-validation
metrics_list = cross_validate_clv(
    transactions=txns_df,
    model_pipeline=clv_pipeline,
    n_folds=5,
    time_increment_months=3,
    initial_train_months=12
)

# Analyze results
print("\n📊 Cross-Validation Results:")
print(f"{'Fold':<6} {'MAPE':<10} {'ARPE':<10} {'R²':<8}")
print("-" * 40)
for i, m in enumerate(metrics_list, 1):
    print(f"{i:<6} {m.mape:<10.2f} {m.arpe:<10.2f} {m.r_squared:<8.3f}")

mean_mape = sum(m.mape for m in metrics_list) / len(metrics_list)
mean_r2 = sum(m.r_squared for m in metrics_list) / len(metrics_list)

print(f"\n{'Mean':<6} {mean_mape:<10.2f} {'—':<10} {mean_r2:<8.3f}")
```

---

### Example 3: MCMC Convergence Diagnostics

```python
from customer_base_audit.validation.diagnostics import check_mcmc_convergence, plot_trace_diagnostics

# Train with MCMC
config_mcmc = BGNBDConfig(method='mcmc', draws=1000, tune=1000, chains=4)
bgnbd_mcmc = BGNBDModelWrapper(config_mcmc)
bgnbd_mcmc.fit(model_data)

# Check convergence
diagnostics = check_mcmc_convergence(bgnbd_mcmc.model.idata)

print("\n🔬 MCMC Convergence Diagnostics:")
print(f"   Converged: {diagnostics.converged}")
print(f"   Max R-hat: {diagnostics.max_r_hat:.4f}")
print(f"   Min ESS (bulk): {diagnostics.min_ess_bulk:.0f}")
print(f"   Min ESS (tail): {diagnostics.min_ess_tail:.0f}")

if not diagnostics.converged:
    print("\n⚠️  Convergence issues detected:")
    for param, r_hat in diagnostics.failed_parameters.items():
        print(f"   {param}: R-hat = {r_hat:.4f}")
    print("\nRecommendation: Increase draws/tune parameters")

# Generate trace plots
plot_trace_diagnostics(bgnbd_mcmc.model.idata, 'diagnostics/trace_plot.png')
print("\n✅ Trace plots saved to diagnostics/trace_plot.png")
```

---

## Best Practices

### 1. Always Use Temporal Splits

❌ **Wrong:** Random train/test split
```python
from sklearn.model_selection import train_test_split
train, test = train_test_split(transactions, test_size=0.2)  # DON'T DO THIS
```

✅ **Right:** Temporal split
```python
train, obs, test = temporal_train_test_split(
    transactions,
    train_end_date=datetime(2024, 9, 1),
    observation_end_date=datetime(2024, 12, 31)
)
```

**Why?** CLV is time-series data. Random splits leak future information.

---

### 2. Validate on Holdout Period

Never tune hyperparameters on test data. Use a separate validation set:

```
├── Training period (fit model)
├── Validation period (tune hyperparameters)
└── Test period (final evaluation)
```

---

### 3. Monitor Metrics Over Time

Set up automated monitoring:
```python
# Log metrics to database/dashboard
log_metrics_to_dashboard({
    'date': datetime.now(),
    'mae': metrics.mae,
    'mape': metrics.mape,
    'r_squared': metrics.r_squared
})

# Alert if metrics degrade
if metrics.mape > baseline_mape * 1.5:
    send_alert("Model drift detected! MAPE increased 50%")
```

---

### 4. Compare to Baselines

Always compare your model to simple baselines:
- **Historical average:** Mean of past CLV
- **Last observed value:** Most recent transaction amount
- **Linear extrapolation:** Trend-based forecast

If your model can't beat these, something is wrong.

---

### 5. Document Validation Results

Create a validation report for each model version:
```markdown
# Model v1.2.0 Validation Report
**Date:** 2024-10-12
**Validation Period:** 2024-09-01 to 2024-12-31
**Training Data:** 2023-01-01 to 2024-08-31

## Metrics
- MAPE: 18.5% ✅ (target: <20%)
- ARPE: 7.2% ✅ (target: <10%)
- R²: 0.73 ✅ (target: >0.5)

## Comparison to v1.1.0
- MAPE: 18.5% (v1.1.0: 22.3%) → 17% improvement
- ARPE: 7.2% (v1.1.0: 9.8%) → 27% improvement

## Recommendation
✅ Deploy v1.2.0 to production
```

---

### 6. Validate Before Deployment

**Pre-deployment checklist:**
- ✅ MAPE < 20% and ARPE < 10%
- ✅ Cross-validation shows stable performance
- ✅ Model beats baseline by >10%
- ✅ MCMC converged (if using MCMC)
- ✅ Predictions make business sense (no negative CLV, outliers investigated)

---

### 7. Use Multiple Metrics

Don't rely on a single metric. Use a dashboard:

| Metric | Value | Status |
|--------|-------|--------|
| MAPE | 18.5% | ✅ |
| ARPE | 7.2% | ✅ |
| R² | 0.73 | ✅ |
| MAE | $52 | ✅ |
| Max Error | $850 | ⚠️ |

**Why?** Different metrics capture different aspects of performance.

---

## Additional Resources

- **API Reference:** [`docs/api_reference.md`](api_reference.md) - Validation module functions
- **Example Notebooks:**
  - [`examples/03_model_comparison.ipynb`](../examples/03_model_comparison.ipynb) - Compare MAP, MCMC, baseline
  - [`examples/04_monitoring_drift.ipynb`](../examples/04_monitoring_drift.ipynb) - Drift detection and retraining

---

## Support

For questions or issues:
- **GitHub Issues:** https://github.com/datablogin/AutoCLV/issues
- **Discussions:** https://github.com/datablogin/AutoCLV/discussions

---

**Built with ❤️ for data-driven customer analytics**
