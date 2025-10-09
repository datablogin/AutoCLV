from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Sequence, Tuple

from .generator import Transaction


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    message: str = ""


def check_non_negative_amounts(transactions: Sequence[Transaction]) -> ValidationResult:
    for idx, t in enumerate(transactions):
        if t.quantity <= 0:
            return ValidationResult(False, f"quantity must be positive at index {idx}")
        if t.unit_price < 0:
            return ValidationResult(False, f"unit_price must be >= 0 at index {idx}")
    return ValidationResult(True, "amounts are non-negative")


def _month_key(ts: datetime) -> Tuple[int, int]:
    return (ts.year, ts.month)


def check_reasonable_order_density(
    transactions: Sequence[Transaction], *, min_avg_lines_per_order: float = 1.0
) -> ValidationResult:
    if not transactions:
        return ValidationResult(True, "no transactions to validate")

    lines_per_order: dict[str, int] = defaultdict(int)
    for t in transactions:
        lines_per_order[t.order_id] += 1

    avg = sum(lines_per_order.values()) / max(1, len(lines_per_order))
    if avg < min_avg_lines_per_order:
        return ValidationResult(
            False, f"average lines/order too low: {avg:.2f} < {min_avg_lines_per_order}"
        )
    return ValidationResult(True, f"average lines/order ok: {avg:.2f}")


def check_promo_spike_signal(
    transactions: Sequence[Transaction], promo_month: int, *, min_ratio: float = 1.2
) -> ValidationResult:
    if not transactions:
        return ValidationResult(False, "no data to assess promo spike")

    spend_by_month: dict[tuple[int, int], float] = defaultdict(float)
    for t in transactions:
        spend_by_month[_month_key(t.event_ts)] += float(t.quantity) * float(
            t.unit_price
        )

    # Average spend in promo month vs other months
    promo_spend = 0.0
    promo_count = 0
    other_spend = 0.0
    other_count = 0
    for (y, m), total in spend_by_month.items():
        if m == promo_month:
            promo_spend += total
            promo_count += 1
        else:
            other_spend += total
            other_count += 1

    if promo_count == 0 or other_count == 0:
        return ValidationResult(False, "insufficient months to compare promo signal")

    promo_avg = promo_spend / promo_count
    other_avg = other_spend / other_count
    ratio = (promo_avg / other_avg) if other_avg > 0 else float("inf")
    if ratio < min_ratio:
        return ValidationResult(
            False,
            f"promo spike too weak: ratio={ratio:.2f} min={min_ratio}",
        )
    return ValidationResult(True, f"promo spike detected: ratio={ratio:.2f}")
