"""Data mart construction utilities for customer × time analyses."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Iterable, Mapping, Sequence


@dataclass(slots=True)
class OrderAggregation:
    """Summary of an order after aggregating line items."""

    order_id: str
    customer_id: str
    order_ts: datetime
    first_item_ts: datetime
    last_item_ts: datetime
    total_spend: float
    total_margin: float
    total_quantity: int
    distinct_products: int


class PeriodGranularity(str, Enum):
    """Supported time granularities for customer aggregations."""

    MONTH = "month"
    QUARTER = "quarter"
    YEAR = "year"


@dataclass(slots=True)
class PeriodAggregation:
    """Aggregated metrics for a customer within a time period.

    Attributes
    ----------
    customer_id:
        Unique customer identifier
    period_start:
        Start of the time period (inclusive)
    period_end:
        End of the time period (exclusive)
    total_orders:
        Total number of orders in this period
    total_spend:
        Total spend amount in this period
    total_margin:
        Total margin in this period
    total_quantity:
        Total quantity of items purchased in this period
    last_transaction_ts:
        Timestamp of the last transaction in this period.
        This provides accurate recency calculation for RFM analysis.
        If None, RFM calculation will fall back to using period_end
        as a conservative approximation.

    Notes
    -----
    **Timezone Assumptions**: All datetime fields (period_start, period_end,
    last_transaction_ts) must use the same timezone (or all be timezone-naive).
    For production use, recommend using UTC timestamps consistently across the
    data pipeline. Mixing timezones will result in incorrect recency calculations.
    """

    customer_id: str
    period_start: datetime
    period_end: datetime
    total_orders: int
    total_spend: float
    total_margin: float
    total_quantity: int
    last_transaction_ts: datetime | None = None


@dataclass
class CustomerDataMart:
    """Container for aggregated order and period level data."""

    orders: list[OrderAggregation]
    periods: dict[PeriodGranularity, list[PeriodAggregation]] = field(
        default_factory=dict
    )

    def as_dict(self) -> dict[str, list[dict[str, object]]]:
        """Return JSON-serialisable representation of the mart."""

        def serialise_order(order: OrderAggregation) -> dict[str, object]:
            return {
                "order_id": order.order_id,
                "customer_id": order.customer_id,
                "order_ts": order.order_ts.isoformat(),
                "first_item_ts": order.first_item_ts.isoformat(),
                "last_item_ts": order.last_item_ts.isoformat(),
                "total_spend": order.total_spend,
                "total_margin": order.total_margin,
                "total_quantity": order.total_quantity,
                "distinct_products": order.distinct_products,
            }

        def serialise_period(period: PeriodAggregation) -> dict[str, object]:
            result = {
                "customer_id": period.customer_id,
                "period_start": period.period_start.isoformat(),
                "period_end": period.period_end.isoformat(),
                "total_orders": period.total_orders,
                "total_spend": period.total_spend,
                "total_margin": period.total_margin,
                "total_quantity": period.total_quantity,
            }
            if period.last_transaction_ts is not None:
                result["last_transaction_ts"] = period.last_transaction_ts.isoformat()
            return result

        payload = {"orders": [serialise_order(order) for order in self.orders]}
        for granularity, aggregates in self.periods.items():
            payload[f"periods_{granularity.value}"] = [
                serialise_period(aggregate) for aggregate in aggregates
            ]
        return payload


class CustomerDataMartBuilder:
    """Build reusable customer-level aggregations from raw transactions."""

    def __init__(self, period_granularities: Sequence[PeriodGranularity] | None = None):
        if period_granularities is None:
            period_granularities = (PeriodGranularity.QUARTER, PeriodGranularity.YEAR)
        self.period_granularities = tuple(dict.fromkeys(period_granularities))

    def build(self, transactions: Iterable[Mapping[str, object]]) -> CustomerDataMart:
        orders = self._aggregate_orders(transactions)
        periods: dict[PeriodGranularity, list[PeriodAggregation]] = {}
        for granularity in self.period_granularities:
            periods[granularity] = self._aggregate_periods(orders, granularity)
        return CustomerDataMart(orders=orders, periods=periods)

    @staticmethod
    def _aggregate_orders(
        transactions: Iterable[Mapping[str, object]],
    ) -> list[OrderAggregation]:
        grouped: dict[str, dict[str, object]] = {}
        for idx, txn in enumerate(transactions):
            try:
                order_id = str(txn["order_id"])
                customer_id = str(txn["customer_id"])
            except KeyError as exc:  # pragma: no cover - defensive
                raise KeyError(
                    f"Transaction at index {idx} missing key {exc.args[0]}"
                ) from exc

            line_ts = txn.get("event_ts") or txn.get("order_ts")
            if not isinstance(line_ts, datetime):
                raise TypeError(
                    "Transactions must provide an event_ts/order_ts datetime",
                    {"index": idx, "value": line_ts},
                )

            quantity = int(txn.get("quantity", 1) or 0)
            if quantity < 0:
                raise ValueError(
                    "Transaction quantity cannot be negative",
                    {"index": idx, "quantity": quantity},
                )

            unit_price = Decimal(str(txn.get("unit_price", 0.0)))
            if unit_price < 0:
                raise ValueError(
                    "Unit price cannot be negative",
                    {"index": idx, "unit_price": unit_price},
                )

            if "line_total" in txn:
                line_total = Decimal(str(txn["line_total"]))
            else:
                line_total = unit_price * quantity
            if line_total < 0:
                raise ValueError(
                    "Line total cannot be negative",
                    {"index": idx, "line_total": line_total},
                )

            if "line_margin" in txn:
                line_margin = Decimal(str(txn["line_margin"]))
            else:
                unit_cost = Decimal(str(txn.get("unit_cost", 0.0)))
                if unit_cost < 0:
                    raise ValueError(
                        "Unit cost cannot be negative",
                        {"index": idx, "unit_cost": unit_cost},
                    )
                if "unit_margin" in txn:
                    unit_margin = Decimal(str(txn["unit_margin"]))
                else:
                    unit_margin = unit_price - unit_cost
                line_margin = unit_margin * quantity

            product_id = str(txn.get("product_id", ""))

            bucket = grouped.setdefault(
                order_id,
                {
                    "customer_id": customer_id,
                    "order_ts": line_ts,
                    "first_item_ts": line_ts,
                    "last_item_ts": line_ts,
                    "total_spend": Decimal("0"),
                    "total_margin": Decimal("0"),
                    "total_quantity": 0,
                    "products": set(),
                },
            )

            bucket["order_ts"] = min(bucket["order_ts"], line_ts)
            bucket["first_item_ts"] = min(bucket["first_item_ts"], line_ts)
            bucket["last_item_ts"] = max(bucket["last_item_ts"], line_ts)
            bucket["total_spend"] += line_total
            bucket["total_margin"] += line_margin
            bucket["total_quantity"] += quantity
            if product_id:
                bucket["products"].add(product_id)

        aggregations: list[OrderAggregation] = []
        for order_id, payload in grouped.items():
            aggregations.append(
                OrderAggregation(
                    order_id=order_id,
                    customer_id=str(payload["customer_id"]),
                    order_ts=payload["order_ts"],
                    first_item_ts=payload["first_item_ts"],
                    last_item_ts=payload["last_item_ts"],
                    total_spend=float(
                        payload["total_spend"].quantize(
                            Decimal("0.01"), rounding=ROUND_HALF_UP
                        )
                    ),
                    total_margin=float(
                        payload["total_margin"].quantize(
                            Decimal("0.01"), rounding=ROUND_HALF_UP
                        )
                    ),
                    total_quantity=int(payload["total_quantity"]),
                    distinct_products=len(payload["products"]),
                )
            )

        aggregations.sort(key=lambda order: (order.customer_id, order.order_ts))
        return aggregations

    def _aggregate_periods(
        self, orders: Iterable[OrderAggregation], granularity: PeriodGranularity
    ) -> list[PeriodAggregation]:
        buckets: dict[tuple[str, datetime], dict[str, object]] = {}
        for order in orders:
            period_start, period_end = _normalise_period(order.order_ts, granularity)
            key = (order.customer_id, period_start)
            if key not in buckets:
                buckets[key] = {
                    "customer_id": order.customer_id,
                    "period_start": period_start,
                    "period_end": period_end,
                    "total_orders": 0,
                    "total_spend": Decimal("0"),
                    "total_margin": Decimal("0"),
                    "total_quantity": 0,
                    "last_transaction_ts": order.order_ts,
                }

            bucket = buckets[key]
            bucket["total_orders"] += 1
            bucket["total_spend"] += Decimal(str(order.total_spend))
            bucket["total_margin"] += Decimal(str(order.total_margin))
            bucket["total_quantity"] += order.total_quantity
            # Track the latest transaction timestamp in this period
            if order.order_ts > bucket["last_transaction_ts"]:
                bucket["last_transaction_ts"] = order.order_ts

        aggregates: list[PeriodAggregation] = []
        for payload in buckets.values():
            aggregates.append(
                PeriodAggregation(
                    customer_id=payload["customer_id"],
                    period_start=payload["period_start"],
                    period_end=payload["period_end"],
                    total_orders=payload["total_orders"],
                    total_spend=float(
                        payload["total_spend"].quantize(
                            Decimal("0.01"), rounding=ROUND_HALF_UP
                        )
                    ),
                    total_margin=float(
                        payload["total_margin"].quantize(
                            Decimal("0.01"), rounding=ROUND_HALF_UP
                        )
                    ),
                    total_quantity=payload["total_quantity"],
                    last_transaction_ts=payload["last_transaction_ts"],
                )
            )

        return sorted(
            aggregates,
            key=lambda aggregation: (aggregation.customer_id, aggregation.period_start),
        )


def _normalise_period(
    dt: datetime, granularity: PeriodGranularity
) -> tuple[datetime, datetime]:
    """Return period start/end datetimes for the provided granularity."""

    if granularity is PeriodGranularity.MONTH:
        period_start = dt.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        next_month = (period_start.replace(day=28) + timedelta(days=4)).replace(day=1)
        period_end = next_month
    elif granularity is PeriodGranularity.QUARTER:
        quarter = (dt.month - 1) // 3
        start_month = quarter * 3 + 1
        period_start = dt.replace(
            month=start_month, day=1, hour=0, minute=0, second=0, microsecond=0
        )
        next_quarter_month = start_month + 3
        if next_quarter_month > 12:
            period_end = period_start.replace(
                year=period_start.year + 1, month=next_quarter_month - 12
            )
        else:
            period_end = period_start.replace(month=next_quarter_month)
    elif granularity is PeriodGranularity.YEAR:
        period_start = dt.replace(
            month=1, day=1, hour=0, minute=0, second=0, microsecond=0
        )
        period_end = period_start.replace(year=period_start.year + 1)
    else:  # pragma: no cover - defensive
        raise ValueError(f"Unsupported granularity: {granularity}")
    return period_start, period_end
