# Phase 1 Testing Guide

## Overview

Phase 1 implements the **Foundation Services** for the Agentic Five Lenses Architecture:
- **Data Mart Service**: Builds customer data mart from transactions
- **RFM Service**: Calculates RFM (Recency, Frequency, Monetary) metrics
- **Cohort Service**: Creates and assigns customers to cohorts

## Testing Options

### Option 1: Run Automated Tests (Recommended)

Run the comprehensive test suite:

```bash
python -m pytest tests/services/mcp_server/test_foundation_tools.py -v
```

**Expected Output:**
```
✓ test_data_mart_build_workflow
✓ test_rfm_calculation_workflow
✓ test_cohort_creation_workflow
✓ test_full_foundation_pipeline
✓ test_rfm_without_data_mart_raises_error
✓ test_cohorts_without_data_mart_raises_error
```

### Option 2: Test with Synthetic Data (Realistic Scenarios)

Run the synthetic data test script to see Phase 1 work with realistic customer scenarios:

```bash
python test_phase1_with_synthetic_data.py
```

This tests three scenarios:
1. **Baseline Business**: Normal customer behavior
2. **High Churn**: Customers churning at higher rates
3. **Stable Business**: Strong retention and repeat purchases

**Sample Output:**
```
================================================================================
TESTING: Baseline Business Scenario
================================================================================

--- Step 1: Building Data Mart ---
✓ Data Mart Built:
  - Orders: 1486
  - Customers: 194
  - Periods: 658

--- Step 2: Calculating RFM Metrics ---
✓ RFM Metrics Calculated:
  - Customers Analyzed: 193
  - RFM Scores Generated: 193

  RFM Averages:
    - Recency: 25.8 days
    - Frequency: 7.69 orders
    - Monetary: $92.70 per order

--- Step 3: Creating Customer Cohorts ---
✓ Cohorts Created:
  - Cohort Count: 1
  - Customers Assigned: 193
```

### Option 3: Test with Your Own Data

#### Using JSON Files

Create a JSON file with your transaction data:

```json
[
  {
    "order_id": "O1",
    "customer_id": "C1",
    "event_ts": "2023-01-15T10:00:00",
    "unit_price": 100.0,
    "quantity": 2
  },
  ...
]
```

Then run:

```bash
python test_phase1_synthetic.py /path/to/your/transactions.json
```

#### Using Python Code

```python
import asyncio
from datetime import datetime
from test_phase1_synthetic import test_with_python_data

transactions = [
    {
        "order_id": "O1",
        "customer_id": "C1",
        "event_ts": datetime(2023, 1, 15, 10, 0, 0),
        "unit_price": 100.0,
        "quantity": 2,
    },
    # ... more transactions
]

asyncio.run(test_with_python_data(transactions))
```

## Understanding the Output

### Data Mart Metrics
- **order_count**: Number of unique orders
- **customer_count**: Number of unique customers
- **period_count**: Number of period aggregations (quarters/years)
- **date_range**: Span of the data

### RFM Metrics
- **metrics_count**: Number of customers analyzed
- **score_count**: Number of RFM scores (1-5 binning)
- **Recency**: Days since last purchase (lower = better)
- **Frequency**: Number of orders (higher = better)
- **Monetary**: Average order value (higher = better)

### RFM Score Interpretation
RFM scores are 3-digit codes (e.g., "555", "111"):
- First digit = Recency score (5 = most recent)
- Second digit = Frequency score (5 = most frequent)
- Third digit = Monetary score (5 = highest value)

**Best customers**: 555, 554, 545, 544
**At-risk customers**: 155, 154, 145 (high value but not purchasing)
**Lost customers**: 111, 112, 121 (haven't purchased in a long time)

### Cohort Metrics
- **cohort_count**: Number of cohorts created
- **customer_count**: Number of customers assigned
- **assignment_summary**: Distribution of customers across cohorts

## Data Requirements

Your transaction data must include:
- `order_id` (string): Unique order identifier
- `customer_id` (string): Customer identifier
- `event_ts` (datetime): Transaction timestamp
- `unit_price` (float): Price per unit
- `quantity` (int): Quantity purchased

Optional fields:
- `unit_cost` (float): Cost per unit (for margin calculations)
- `line_total` (float): Total line amount (calculated if not provided)
- `line_margin` (float): Line margin (calculated if not provided)

## Performance Notes

- **Parallel Processing**: RFM calculation supports parallel processing for large datasets
  - Enable with `enable_parallel=True` (default)
  - Recommended for >10,000 customers

- **Memory Usage**: Data is stored in MCP context between steps
  - Data mart, RFM metrics, and cohorts are all cached
  - Memory usage proportional to customer count

## Troubleshooting

### "Data mart not found" error
Make sure to run `build_customer_data_mart` before `calculate_rfm_metrics` or `create_customer_cohorts`.

### Timezone errors
Ensure all datetime fields use consistent timezone handling:
- Either all naive (no timezone)
- Or all timezone-aware (e.g., UTC)

### Performance issues
For datasets >100k customers:
1. Enable parallel processing
2. Consider reducing period granularities
3. Use year-level aggregations instead of month

## Next Steps

After validating Phase 1:
- **Phase 2**: Lens Services (Lenses 1-4 as MCP tools)
- **Phase 3**: LangGraph orchestration layer
- **Phase 4**: Production observability and resilience
- **Phase 5**: Natural language interface with Claude

## Files

- `analytics/services/mcp_server/tools/data_mart.py` - Data Mart service
- `analytics/services/mcp_server/tools/rfm.py` - RFM service
- `analytics/services/mcp_server/tools/cohorts.py` - Cohort service
- `tests/services/mcp_server/test_foundation_tools.py` - Automated tests
- `test_phase1_synthetic.py` - Simple testing script
- `test_phase1_with_synthetic_data.py` - Comprehensive scenario testing
