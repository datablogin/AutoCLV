from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
import math
import random
from typing import List, Optional, Sequence


@dataclass(frozen=True)
class Customer:
    customer_id: str
    acquisition_date: date


@dataclass(frozen=True)
class Transaction:
    order_id: str
    customer_id: str
    event_ts: datetime
    quantity: int
    unit_price: float
    product_id: str


@dataclass(frozen=True)
class ScenarioConfig:
    """Configuration for scenario-based generators.

    Attributes
    ----------
    promo_month: A specific month that should see higher purchase activity.
    promo_uplift: Multiplicative uplift for purchase propensity during promo month.
    launch_date: Date after which demand gradually increases.
    churn_hazard: Baseline monthly churn probability for existing customers.
    base_orders_per_month: Average orders per active customer per month.
    mean_unit_price: Average item price used to sample line items.
    price_variability: Coefficient in (0, 1] controlling price variance.
    quantity_mean: Average quantity per order line.
    seed: Optional RNG seed for reproducibility.
    """

    promo_month: Optional[int] = None
    promo_uplift: float = 1.5
    launch_date: Optional[date] = None
    churn_hazard: float = 0.08
    base_orders_per_month: float = 1.2
    mean_unit_price: float = 30.0
    price_variability: float = 0.4
    quantity_mean: float = 1.3
    seed: Optional[int] = None


def _month_range(start: date, end: date) -> List[date]:
    cur = date(start.year, start.month, 1)
    last = date(end.year, end.month, 1)
    out: List[date] = []
    while cur <= last:
        out.append(cur)
        if cur.month == 12:
            cur = date(cur.year + 1, 1, 1)
        else:
            cur = date(cur.year, cur.month + 1, 1)
    return out


def generate_customers(
    n: int,
    start: date,
    end: date,
    *,
    seed: Optional[int] = None,
) -> List[Customer]:
    """Generate ``n`` customers with acquisition dates uniformly between start/end."""

    if n <= 0:
        return []
    if start > end:
        raise ValueError("start date must be <= end date")

    rng = random.Random(seed)
    total_days = (end - start).days + 1

    customers: List[Customer] = []
    for i in range(n):
        offset = rng.randrange(total_days)
        acq = start + timedelta(days=offset)
        customers.append(Customer(customer_id=f"C-{i + 1}", acquisition_date=acq))
    return customers


def _orders_for_customer_month(
    rng: random.Random,
    base_orders_per_month: float,
    promo_multiplier: float,
    launch_uplift: float,
) -> int:
    # Poisson-like draw via Knuth's algorithm approximation for small lambdas
    lam = max(0.0, base_orders_per_month * promo_multiplier * launch_uplift)
    if lam <= 0:
        return 0
    L = math.exp(-lam)
    k = 0
    p = 1.0
    while p > L:
        k += 1
        p *= rng.random()
    return max(0, k - 1)


def _sample_price(rng: random.Random, mean: float, variability: float) -> float:
    variability = min(max(variability, 0.01), 1.0)
    # Log-normal-ish by exponentiating a normal draw for positivity
    sigma = variability
    mu = math.log(max(mean, 0.01)) - 0.5 * sigma * sigma
    price = math.exp(rng.normalvariate(mu, sigma))
    return round(max(price, 0.01), 2)


def _sample_quantity(rng: random.Random, mean_q: float) -> int:
    # Discretized log-normal for positive integer quantities
    q = max(1.0, rng.lognormvariate(mu=math.log(max(mean_q, 0.1)), sigma=0.5))
    return max(1, int(round(q)))


def generate_transactions(
    customers: Sequence[Customer],
    start: date,
    end: date,
    *,
    scenario: Optional[ScenarioConfig] = None,
    catalog: Optional[Sequence[str]] = None,
) -> List[Transaction]:
    """Generate transactions for customers between ``start`` and ``end``.

    The generator applies three optional scenario effects:
    - Promo spike in a given calendar month.
    - Product launch uplift after a specific date.
    - Baseline churn reducing the number of active customers over time.
    """

    if start > end:
        raise ValueError("start date must be <= end date")
    scenario = scenario or ScenarioConfig()
    rng = random.Random(scenario.seed)
    product_catalog = list(catalog) if catalog else [f"SKU-{i + 1}" for i in range(20)]

    months = _month_range(start, end)
    transactions: List[Transaction] = []
    order_seq = 1

    # Track active customers with simple per-month churn
    active = {c.customer_id: c for c in customers if c.acquisition_date <= end}

    for month_start in months:
        month_end = (
            date(month_start.year + 1, 1, 1)
            if month_start.month == 12
            else date(month_start.year, month_start.month + 1, 1)
        )
        # Scenario multipliers
        promo_multiplier = (
            scenario.promo_uplift
            if (scenario.promo_month and month_start.month == scenario.promo_month)
            else 1.0
        )
        launch_uplift = 1.0
        if scenario.launch_date and month_start >= scenario.launch_date:
            # Gradual ramp by months since launch
            months_since = (month_start.year - scenario.launch_date.year) * 12 + (
                month_start.month - scenario.launch_date.month
            )
            launch_uplift = 1.0 + min(0.75, 0.05 * months_since)

        # Per-month churn applied to currently active customers
        if scenario.churn_hazard > 0:
            to_remove = []
            for cid, cust in active.items():
                if cust.acquisition_date > month_end:
                    continue
                if rng.random() < scenario.churn_hazard:
                    to_remove.append(cid)
            for cid in to_remove:
                active.pop(cid, None)

        # Generate orders for currently active customers
        for cust in list(active.values()):
            if cust.acquisition_date > month_end:
                continue

            num_orders = _orders_for_customer_month(
                rng,
                scenario.base_orders_per_month,
                promo_multiplier,
                launch_uplift,
            )
            for _ in range(num_orders):
                # Event in the middle of the month +/- random jitter
                mid = datetime(
                    month_start.year,
                    month_start.month,
                    min(28, 15 + rng.randrange(0, 10)),
                    10 + rng.randrange(0, 9),
                    rng.randrange(0, 60),
                )
                order_id = f"O-{order_seq}"
                order_seq += 1

                # Sample 1-3 line items per order
                for _line in range(1 + rng.randrange(3)):
                    transactions.append(
                        Transaction(
                            order_id=order_id,
                            customer_id=cust.customer_id,
                            event_ts=mid,
                            quantity=_sample_quantity(rng, scenario.quantity_mean),
                            unit_price=_sample_price(
                                rng,
                                scenario.mean_unit_price,
                                scenario.price_variability,
                            ),
                            product_id=rng.choice(product_catalog),
                        )
                    )

    transactions.sort(key=lambda t: (t.customer_id, t.event_ts, t.order_id))
    return transactions
