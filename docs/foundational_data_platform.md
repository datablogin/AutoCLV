# Foundational Customer Data Platform Elements

Issue [#1](https://github.com/datablogin/AutoCLV/issues/1) introduces the
minimum shared assets required to support consistent customer-base audits.

## Customer Contract

The customer contract lives in `customer_base_audit.foundation.customer_contract`.
Use `CustomerContract.validate_records()` to harden upstream payloads and
`CustomerContract.merge()` to combine the same customer across source systems.

```python
from datetime import datetime
from customer_base_audit.foundation import CustomerContract

contract = CustomerContract()
identifiers = contract.validate_records(
    [
        {"customer_id": "123", "acquisition_ts": datetime(2023, 1, 15)},
        {"customer_id": "123", "acquisition_ts": datetime(2023, 2, 2), "is_visible": False},
    ],
    source_system="crm",
)
canonical = contract.merge(identifiers)
```

## Customer × Time Data Mart

The `CustomerDataMartBuilder` converts raw transaction line items into both
order-level summaries and customer-period aggregations.

```python
from datetime import datetime
from customer_base_audit.foundation import CustomerDataMartBuilder

transactions = [
    {
        "order_id": "A-1",
        "customer_id": "123",
        "event_ts": datetime(2024, 1, 4, 10, 2),
        "product_id": "SKU-1",
        "quantity": 2,
        "unit_price": 30.0,
    },
    {
        "order_id": "A-1",
        "customer_id": "123",
        "event_ts": datetime(2024, 1, 4, 10, 5),
        "product_id": "SKU-2",
        "quantity": 1,
        "unit_price": 15.0,
    },
]

builder = CustomerDataMartBuilder()
mart = builder.build(transactions)
orders = mart.orders  # order-level aggregations
periods = mart.periods[builder.period_granularities[0]]
```

The CLI entry-point replicates this behaviour from the shell:

```bash
python -m customer_base_audit.cli transactions.json --output mart.json
```

This document will evolve with additional patterns (e.g., dbt staging
models, orchestrator templates) as the platform matures.
