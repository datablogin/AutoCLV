# AutoCLV

A lightweight toolkit for customer-base audits and CLV analysis, plus an analytics backend (in `analytics/`). This README highlights the new synthetic data features and how to use them for demos, tests, and CI.

## What’s New
- Synthetic data generation toolkit for CLV/audit workflows
- Validations to sanity-check generated datasets
- Turnkey Texas CLV client scenario (1,000 customers across major cities)

## Install

- Python 3.12+
- Optional: create and activate a virtualenv (e.g., `.venv`)
- Install dev tools if needed:

```bash
pip install -r requirements.txt -r requirements-dev.txt || true
```

The project also supports `ruff` and `pytest` for linting/testing.

## Synthetic Data Toolkit

Modules:
- `customer_base_audit/synthetic/generator.py`
  - `generate_customers(start, end, n, seed)`
  - `generate_transactions(customers, start, end, scenario)`
  - `ScenarioConfig` with validated parameters
- `customer_base_audit/synthetic/validation.py`
  - `check_non_negative_amounts`, `check_reasonable_order_density`, `check_promo_spike_signal`

### Quick Start (Python)
```python
from datetime import date
from customer_base_audit.synthetic import (
    ScenarioConfig,
    generate_customers,
    generate_transactions,
    check_promo_spike_signal,
)

# 1) Generate customers
customers = generate_customers(100, date(2024, 1, 1), date(2024, 12, 31), seed=7)

# 2) Configure scenario: May promo spike, reproducible seed
scenario = ScenarioConfig(promo_month=5, promo_uplift=2.0, seed=7)

# 3) Generate transactions for 2024
transactions = generate_transactions(customers, date(2024, 1, 1), date(2024, 12, 31), scenario=scenario)

# 4) Validate promo spike signal (expect True)
assert check_promo_spike_signal(transactions, promo_month=5, min_ratio=1.1).ok

print(len(customers), len(transactions))
```

Expected outcome with the snippet above:
- `len(customers) == 100`
- `len(transactions) == 1503`
- `check_promo_spike_signal(...)` returns `True`

## Texas CLV Client (CSV Output)
Create a 1,000‑customer dataset with city store openings and write CSVs.

```bash
python -m customer_base_audit.synthetic.texas_clv_client
# Outputs:
# data/texas_clv_client/customers.csv
# data/texas_clv_client/transactions.csv
```

Programmatic usage:
```python
from customer_base_audit.synthetic.texas_clv_client import generate_texas_clv_client

customers, txns, city_map = generate_texas_clv_client(total_customers=1000, seed=42)

print(len(customers))   # 1000
print(len(txns))        # 30025
print(sorted(set(city_map.values())))  # ['Austin', 'Dallas', 'Houston', 'San Antonio']
```

Expected outcome with seed=42:
- 1,000 customers
- 30,025 transaction lines
- City distribution: Houston 35%, Dallas 30%, San Antonio 20%, Austin 15%

## Build a Customer×Time Mart (Optional)
Use the foundational builder to aggregate orders and periods for CLV/Lens analyses.

```python
import pandas as pd
from customer_base_audit.foundation import CustomerDataMartBuilder, PeriodGranularity

tx = pd.read_csv("data/texas_clv_client/transactions.csv", parse_dates=["event_ts"]).to_dict(orient="records")
builder = CustomerDataMartBuilder(period_granularities=[PeriodGranularity.QUARTER, PeriodGranularity.YEAR])
mart = builder.build(tx)

print(len(mart.orders))                # e.g., number of aggregated orders
print({g: len(v) for g, v in mart.periods.items()})  # period aggregates
```

Notes
- Generators are deterministic for a given `seed` and Python version.
- `ScenarioConfig` enforces safe parameter ranges; invalid values raise `ValueError`.
- CSV outputs are ignored by Git (`data/`), regenerate locally as needed.

