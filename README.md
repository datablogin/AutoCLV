# AutoCLV

**Customer Lifetime Value Analytics and Customer Base Audits Made Simple**

AutoCLV is a Python toolkit for analyzing customer behavior and predicting customer lifetime value (CLV). Built on proven probabilistic models (BG/NBD, Gamma-Gamma) and the Five Lenses framework from "The Customer-Base Audit", AutoCLV helps you understand customer retention, segment your customer base, and forecast future revenue.

[![CI Tests](https://img.shields.io/github/actions/workflow/status/datablogin/AutoCLV/ci.yml?branch=main&label=tests)](https://github.com/datablogin/AutoCLV/actions)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## ✨ Features

### 📊 Customer Analytics
- **RFM Analysis**: Recency, Frequency, Monetary segmentation
- **Five Lenses Framework**: Multi-dimensional customer base health assessment
- **Cohort Analysis**: Track customer behavior over time
- **Retention Metrics**: Measure and monitor customer churn

### 🎯 CLV Prediction
- **BG/NBD Model**: Predict purchase frequency and customer lifetime
- **Gamma-Gamma Model**: Estimate average transaction value
- **CLV Calculator**: Combine models for actionable CLV scores
- **Model Validation**: Built-in diagnostics and drift detection

### 🧪 Synthetic Data
- **Scenario Pack**: 7 pre-built business scenarios (baseline, high churn, product recall, etc.)
- **Texas CLV Client**: Realistic 1,000-customer dataset across 4 Texas cities
- **Validation Suite**: Ensure data quality for testing and demos

### 📚 Production-Ready
- **MAP & MCMC**: Fast parameter estimation or full Bayesian inference
- **Model Monitoring**: Detect drift and know when to retrain
- **Integration Tests**: End-to-end pipeline validation
- **Comprehensive Docs**: API reference, user guide, and example notebooks

---

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/datablogin/AutoCLV.git
cd AutoCLV

# Create virtual environment (recommended)
python3.12 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install package
pip install -e .

# Install dev dependencies (optional)
pip install -r requirements-dev.txt

# Verify installation
python -c "from customer_base_audit.foundation import rfm; print('✓ Installation successful')"
```

**Requirements:** Python 3.12+

### 30-Second Example

```python
from datetime import date, datetime
from customer_base_audit.synthetic import generate_customers, generate_transactions, BASELINE_SCENARIO
from customer_base_audit.foundation.data_mart import CustomerDataMartBuilder, PeriodGranularity
from customer_base_audit.foundation.rfm import calculate_rfm, calculate_rfm_scores
from customer_base_audit.analyses.lens1 import analyze_single_period

# 1. Generate synthetic data
customers = generate_customers(500, date(2024, 1, 1), date(2024, 12, 31), seed=42)
transactions = generate_transactions(customers, date(2024, 1, 1), date(2024, 12, 31), BASELINE_SCENARIO)

# 2. Build data mart
builder = CustomerDataMartBuilder([PeriodGranularity.MONTH])
mart = builder.build([t.__dict__ for t in transactions])

# 3. Calculate RFM metrics
rfm_metrics = calculate_rfm(mart.periods[PeriodGranularity.MONTH], datetime(2024, 12, 31, 23, 59, 59))
rfm_scores = calculate_rfm_scores(rfm_metrics)

# 4. Run Lens 1 analysis
lens1_results = analyze_single_period(rfm_metrics, rfm_scores)

print(f"📊 Customer Base Health Check")
print(f"   Total Customers: {lens1_results.total_customers:,}")
print(f"   One-Time Buyers: {lens1_results.one_time_buyer_pct:.1f}%")
print(f"   Top 10% Revenue: {lens1_results.top_10pct_revenue_contribution:.1f}%")
print(f"   Median CLV: ${lens1_results.median_customer_value:.2f}")
```

**Output:**
```
📊 Customer Base Health Check
   Total Customers: 500
   One-Time Buyers: 42.8%
   Top 10% Revenue: 35.2%
   Median CLV: $127.50
```

---

## 📖 Documentation

### Core Documentation
- **[API Reference](docs/api_reference.md)** - Complete API documentation with examples
- **[User Guide](docs/user_guide.md)** - Comprehensive tutorials and best practices
- **[Synthetic Data Toolkit](docs/TESTING_WITH_SYNTHETIC_DATA.md)** - Generate realistic test data

### Example Notebooks

Explore complete workflows in our Jupyter notebooks:

1. **[Texas CLV Walkthrough](examples/01_texas_clv_walkthrough.ipynb)**
   End-to-end CLV analysis with synthetic Texas customer data
   - Data generation and exploration
   - RFM analysis and customer segmentation
   - Five Lenses analyses (Lens 1-3)
   - BG/NBD and Gamma-Gamma model training
   - CLV predictions and visualizations

2. **[Custom Cohorts Analysis](examples/02_custom_cohorts.ipynb)**
   Advanced cohort analysis techniques
   - Monthly, quarterly, and geographic cohorts
   - Multi-cohort comparisons
   - Retention curves and heatmaps
   - Cohort health assessment

3. **[Model Comparison](examples/03_model_comparison.ipynb)**
   Compare different CLV modeling approaches
   - Historical Average (baseline)
   - BG/NBD + Gamma-Gamma (MAP)
   - BG/NBD + Gamma-Gamma (MCMC)
   - Performance benchmarks and tradeoffs

4. **[Monitoring Model Drift](examples/04_monitoring_drift.ipynb)**
   Production model monitoring and drift detection
   - Time-based train/test splits
   - Rolling window validation
   - Statistical drift tests (Kolmogorov-Smirnov)
   - Automated retraining decisions

---

## 🎓 Key Concepts

### RFM Analysis

RFM segments customers based on three dimensions:
- **Recency**: How recently did they purchase?
- **Frequency**: How often do they purchase?
- **Monetary**: How much do they spend?

```python
from customer_base_audit.foundation.rfm import calculate_rfm, calculate_rfm_scores

rfm_metrics = calculate_rfm(period_aggregations, observation_end)
rfm_scores = calculate_rfm_scores(rfm_metrics)

# Find best customers (RFM score 555)
best_customers = [s for s in rfm_scores if s.rfm_segment == "555"]
```

### Five Lenses Framework

The Five Lenses provide multiple perspectives on customer base health:

1. **Lens 1: Single Period** - Snapshot of current customer base
2. **Lens 2: Period Comparison** - Track retention and churn
3. **Lens 3: Cohort Evolution** - How cohorts age over time
4. **Lens 4: Multi-Cohort Comparison** *(coming soon)*
5. **Lens 5: Overall Health** *(coming soon)*

```python
from customer_base_audit.analyses import lens1, lens2, lens3

# Lens 1: Current snapshot
lens1_results = lens1.analyze_single_period(rfm_metrics, rfm_scores)

# Lens 2: Q3 vs Q4 comparison
lens2_results = lens2.analyze_period_comparison(q3_rfm, q4_rfm, all_customer_ids)

# Lens 3: January cohort evolution
lens3_results = lens3.analyze_cohort_evolution(
    cohort_name="2024-01",
    acquisition_date=datetime(2024, 1, 1),
    period_aggregations=period_aggs,
    cohort_customer_ids=cohort_customer_ids
)
```

### CLV Prediction Models

AutoCLV implements industry-standard probabilistic models:

- **BG/NBD**: Predicts purchase frequency and probability alive
- **Gamma-Gamma**: Predicts average transaction value
- **Combined**: Purchase frequency × Transaction value = CLV

```python
from customer_base_audit.models.bg_nbd import BGNBDModelWrapper, BGNBDConfig
from customer_base_audit.models.gamma_gamma import GammaGammaModelWrapper, GammaGammaConfig
from customer_base_audit.models.clv_calculator import CLVCalculator
from customer_base_audit.models.model_prep import prepare_clv_model_inputs

# Prepare data
model_data = prepare_clv_model_inputs(
    transactions=transactions,
    observation_start=datetime(2024, 1, 1),
    observation_end=datetime(2024, 12, 31, 23, 59, 59),
    customer_id_field='customer_id',
    timestamp_field='event_ts',
    monetary_field='unit_price'
)

# Train models
bgnbd_model = BGNBDModelWrapper(BGNBDConfig(method="map"))
bgnbd_model.fit(model_data)

gamma_gamma_model = GammaGammaModelWrapper(GammaGammaConfig(method="map"))
gamma_gamma_model.fit(model_data[model_data['frequency'] >= 2])

# Calculate CLV
calculator = CLVCalculator()
clv_scores = calculator.calculate_clv(
    bgnbd_model=bgnbd_model,
    gamma_gamma_model=gamma_gamma_model,
    customer_data=model_data,
    time_period_days=90
)

# Find top 10 customers by CLV
top_10 = sorted(clv_scores, key=lambda x: x.clv, reverse=True)[:10]
for score in top_10:
    print(f"{score.customer_id}: ${score.clv:.2f}")
```

---

## 📊 Pandas Integration

For users who prefer DataFrame workflows, the `customer_base_audit.pandas` module provides convenience adapters for all Track A components (RFM, Lens 1, Lens 2).

### Quick Start
```python
import pandas as pd
from datetime import datetime
from customer_base_audit.pandas import calculate_rfm_df, analyze_single_period_df

# Load period data
periods_df = pd.read_csv('customer_periods.csv')

# Calculate RFM metrics (one line!)
rfm_df = calculate_rfm_df(periods_df, observation_end=datetime(2023, 12, 31))

# Analyze single period
lens1_df = analyze_single_period_df(rfm_df)

# Export for Tableau/PowerBI
rfm_df.to_csv('rfm_scores.csv', index=False)
lens1_df.to_csv('lens1_metrics.csv', index=False)
```

### Period Comparison
```python
from customer_base_audit.pandas import analyze_period_comparison_df

# Calculate RFM for two periods
q1_rfm = calculate_rfm_df(q1_periods_df, datetime(2023, 3, 31))
q2_rfm = calculate_rfm_df(q2_periods_df, datetime(2023, 6, 30))

# Compare periods
comparison = analyze_period_comparison_df(q1_rfm, q2_rfm)

# Access results as DataFrames
print(f"Retention: {comparison['metrics']['retention_rate'].iloc[0]}%")

# Analyze churned customers
churned = comparison['migration'][comparison['migration']['status'] == 'churned']
churned_rfm = q1_rfm[q1_rfm['customer_id'].isin(churned['customer_id'])]
print(f"Avg churned customer value: ${churned_rfm['monetary'].mean():.2f}")
```

**Benefits:**
- ✅ One-line DataFrame conversions
- ✅ Native pandas workflows in Jupyter notebooks
- ✅ Easy export to BI tools (Tableau, PowerBI, etc.)
- ✅ 100% backward compatible with existing dataclass API

---

## 🧪 Synthetic Data for Testing

AutoCLV includes a powerful synthetic data generator with 7 pre-built scenarios:

```python
from customer_base_audit.synthetic import (
    generate_customers,
    generate_transactions,
    BASELINE_SCENARIO,
    HIGH_CHURN_SCENARIO,
    PRODUCT_RECALL_SCENARIO
)
from datetime import date

# Generate customers
customers = generate_customers(1000, date(2024, 1, 1), date(2024, 12, 31), seed=42)

# Generate transactions with different scenarios
baseline_txns = generate_transactions(customers, date(2024, 1, 1), date(2024, 12, 31), BASELINE_SCENARIO)
high_churn_txns = generate_transactions(customers, date(2024, 1, 1), date(2024, 12, 31), HIGH_CHURN_SCENARIO)
```

**Available Scenarios:**
- `BASELINE_SCENARIO`: Moderate, stable business
- `HIGH_CHURN_SCENARIO`: 30% monthly churn
- `PRODUCT_RECALL_SCENARIO`: 70% activity drop in June
- `HEAVY_PROMOTION_SCENARIO`: 5x activity spike
- `PRODUCT_LAUNCH_SCENARIO`: Gradual ramp-up after launch
- `SEASONAL_BUSINESS_SCENARIO`: Strong Q4 seasonality
- `STABLE_BUSINESS_SCENARIO`: Very low churn (2%)

### Texas CLV Client

Generate a realistic 1,000-customer dataset across 4 Texas cities:

```python
from customer_base_audit.synthetic.texas_clv_client import generate_texas_clv_client

customers, transactions, city_map = generate_texas_clv_client(total_customers=1000, seed=42)

print(f"Generated {len(customers):,} customers")
print(f"Generated {len(transactions):,} transactions")
print(f"Cities: {sorted(set(city_map.values()))}")
```

**Output:**
```
Generated 1,000 customers
Generated 30,025 transactions
Cities: ['Austin', 'Dallas', 'Houston', 'San Antonio']
```

Or generate CSV files:
```bash
python -m customer_base_audit.synthetic.texas_clv_client
# Creates:
#   data/texas_clv_client/customers.csv
#   data/texas_clv_client/transactions.csv
```

**Notes:**
- Generators are deterministic for a given `seed` and Python version
- `ScenarioConfig` enforces safe parameter ranges; invalid values raise `ValueError`
- CSV outputs are ignored by Git (`data/`), regenerate locally as needed

---

## 🏗️ Architecture

```
AutoCLV/
├── customer_base_audit/
│   ├── foundation/          # Core data structures
│   │   ├── rfm.py          # RFM calculation
│   │   ├── data_mart.py    # Data aggregation
│   │   └── cohorts.py      # Cohort definition
│   ├── analyses/            # Five Lenses framework
│   │   ├── lens1.py        # Single period analysis
│   │   ├── lens2.py        # Period comparison
│   │   └── lens3.py        # Cohort evolution
│   ├── models/              # CLV models
│   │   ├── bg_nbd.py       # BG/NBD model
│   │   ├── gamma_gamma.py  # Gamma-Gamma model
│   │   ├── clv_calculator.py
│   │   └── model_prep.py   # Data preparation
│   ├── synthetic/           # Data generation
│   │   ├── generator.py
│   │   ├── scenarios.py
│   │   └── texas_clv_client.py
│   └── validation/          # Model validation
├── docs/                    # Documentation
├── examples/                # Jupyter notebooks
└── tests/                   # Test suite
```

---

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=customer_base_audit --cov-report=html

# Run specific test suite
pytest tests/test_rfm.py -v

# Run integration tests
pytest tests/integration/ -v
```

---

## 🛠️ Development

### Code Quality

```bash
# Linting
ruff check .

# Formatting
ruff format .

# Type checking (if enabled)
mypy customer_base_audit
```

### CI/CD

All PRs are automatically tested with:
- ✅ Unit tests (pytest)
- ✅ Linting (ruff)
- ✅ Formatting checks
- ✅ Integration tests

---

## 📊 Performance

### Model Training Speed

| Model | Dataset Size | Method | Time | Use Case |
|-------|-------------|--------|------|----------|
| BG/NBD | 1,000 customers | MAP | ~5s | Production |
| BG/NBD | 1,000 customers | MCMC | ~2min | Research |
| BG/NBD | 10,000 customers | MAP | ~30s | Enterprise |
| Gamma-Gamma | 1,000 customers | MAP | ~3s | Production |

**Note:** Performance timings are approximate and measured on M1 MacBook Pro. Actual times may vary based on hardware, data characteristics, and system load.

**Recommendations:**
- Use **MAP** for production (fast, accurate parameter estimates)
- Use **MCMC** for research (full uncertainty quantification)
- For > 10K customers, consider distributed computing for MCMC

---

## 🤝 Contributing

We welcome contributions! Please see our contribution guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests (`pytest`)
5. Commit with conventional commits (`git commit -m "feat: add amazing feature"`)
6. Push to branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🙏 Acknowledgments

- **PyMC-Marketing**: Probabilistic models (BG/NBD, Gamma-Gamma)
- **"The Customer-Base Audit"**: Five Lenses framework inspiration
- **Lifetimes Library**: CLV modeling concepts

---

## 📬 Support

- **Issues**: [GitHub Issues](https://github.com/datablogin/AutoCLV/issues)
- **Discussions**: [GitHub Discussions](https://github.com/datablogin/AutoCLV/discussions)
- **Documentation**: [docs/](docs/)

---

## 🗺️ Roadmap

### Current Status
- ✅ RFM Analysis
- ✅ Lenses 1-3
- ✅ BG/NBD and Gamma-Gamma Models
- ✅ Synthetic Data Generation
- ✅ Example Notebooks
- ✅ Validation Framework

### Coming Soon
- 🔜 Lens 4: Multi-Cohort Comparison
- 🔜 Lens 5: Overall Customer Base Health
- 🔜 Advanced Drift Detection
- 🔜 Model Interpretability Tools
- 🔜 REST API for CLV Predictions
- 🔜 Dashboard UI

---

**Built with ❤️ for customer analytics professionals**
