from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Dict, List, Sequence, Tuple
import csv

from .generator import (
    Customer,
    ScenarioConfig,
    Transaction,
    generate_customers,
    generate_transactions,
)


@dataclass(frozen=True)
class CityPlan:
    city: str
    share: float
    launch_date: date
    promo_month: int


def _allocate_counts(total: int, shares: Sequence[float]) -> List[int]:
    # Largest-remainder method to ensure sum equals total
    raw = [s * total for s in shares]
    floors = [int(x) for x in raw]
    remainder = total - sum(floors)
    remainders = sorted(
        enumerate([x - f for x, f in zip(raw, floors)]),
        key=lambda t: t[1],
        reverse=True,
    )
    for i in range(remainder):
        floors[remainders[i][0]] += 1
    return floors


def generate_texas_clv_client(
    total_customers: int = 1000, *, seed: int = 42
) -> Tuple[List[Customer], List[Transaction], Dict[str, str]]:
    """Generate a 1k-customer CLV client across four Texas cities with store openings.

    Returns customers, transactions, and a mapping of customer_id -> city.
    """

    city_plans: List[CityPlan] = [
        CityPlan("Houston", 0.35, date(2024, 6, 1), 6),
        CityPlan("Dallas", 0.30, date(2024, 4, 1), 4),
        CityPlan("San Antonio", 0.20, date(2024, 8, 1), 8),
        CityPlan("Austin", 0.15, date(2024, 10, 1), 10),
    ]
    shares = [c.share for c in city_plans]
    counts = _allocate_counts(total_customers, shares)

    all_customers: List[Customer] = []
    city_by_customer: Dict[str, str] = {}

    # Acquisition dates over two years for spread
    acq_start = date(2023, 1, 1)
    acq_end = date(2024, 12, 31)

    # Generate per-city customers
    offset = 0
    for idx, (plan, n) in enumerate(zip(city_plans, counts)):
        customers = generate_customers(n, acq_start, acq_end, seed=seed + idx)
        # Re-label customer ids to ensure uniqueness across cities
        renamed: List[Customer] = []
        for i, c in enumerate(customers, start=1):
            cid = f"{plan.city[:2].upper()}-{offset + i}"
            renamed.append(
                Customer(customer_id=cid, acquisition_date=c.acquisition_date)
            )
            city_by_customer[cid] = plan.city
        offset += n
        all_customers.extend(renamed)

    # Generate per-city transactions with city-specific launch and promo
    txns_all: List[Transaction] = []
    start = date(2023, 1, 1)
    end = date(2024, 12, 31)
    catalog = [
        "SKU-DRY-GOODS",
        "SKU-BEVERAGE",
        "SKU-PREPARED",
        "SKU-HBA",
        "SKU-HOME",
    ]

    for idx, plan in enumerate(city_plans):
        cust_subset = [
            c for c in all_customers if city_by_customer[c.customer_id] == plan.city
        ]
        scenario = ScenarioConfig(
            promo_month=plan.promo_month,
            promo_uplift=1.8,
            launch_date=plan.launch_date,
            churn_hazard=0.06,
            base_orders_per_month=1.1,
            mean_unit_price=28.0,
            price_variability=0.35,
            quantity_mean=1.2,
            seed=seed + 100 + idx,
        )
        txns = generate_transactions(
            cust_subset, start, end, scenario=scenario, catalog=catalog
        )
        txns_all.extend(txns)

    return all_customers, txns_all, city_by_customer


def _write_csvs(
    customers: Sequence[Customer],
    transactions: Sequence[Transaction],
    city_by_customer: Dict[str, str],
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "customers.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["customer_id", "acquisition_date", "city"])
        for c in customers:
            w.writerow(
                [
                    c.customer_id,
                    c.acquisition_date.isoformat(),
                    city_by_customer[c.customer_id],
                ]
            )

    with (out_dir / "transactions.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "order_id",
                "customer_id",
                "event_ts",
                "product_id",
                "quantity",
                "unit_price",
            ]
        )
        for t in transactions:
            w.writerow(
                [
                    t.order_id,
                    t.customer_id,
                    t.event_ts.isoformat(),
                    t.product_id,
                    t.quantity,
                    f"{t.unit_price:.2f}",
                ]
            )


def main() -> None:
    customers, txns, city_map = generate_texas_clv_client()
    out = Path("data/texas_clv_client")
    _write_csvs(customers, txns, city_map, out)
    print(
        f"Wrote {len(customers)} customers and {len(txns)} transaction lines to {out}"
    )


if __name__ == "__main__":
    main()
