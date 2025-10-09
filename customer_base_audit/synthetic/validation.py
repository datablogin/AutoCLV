from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Sequence, Tuple

from .generator import Transaction, Customer


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


# ========== Statistical Validation Functions ==========


def check_spend_distribution_is_realistic(
    transactions: Sequence[Transaction],
    *,
    expected_mean: float | None = None,
    expected_std: float | None = None,
    tolerance: float = 0.3,
) -> ValidationResult:
    """Validate that transaction spend follows expected distribution characteristics.

    Parameters
    ----------
    transactions:
        Transaction data to validate.
    expected_mean:
        Expected mean transaction value. If None, uses general heuristics.
    expected_std:
        Expected standard deviation. If None, uses general heuristics.
    tolerance:
        Acceptable deviation from expected values (as fraction, e.g., 0.3 = 30%).

    Returns
    -------
    ValidationResult
        Validation result with distribution statistics.
    """
    if not transactions:
        return ValidationResult(False, "no transactions to validate")

    # Calculate transaction values
    values = [t.quantity * t.unit_price for t in transactions]
    if not values:
        return ValidationResult(False, "no transaction values found")

    mean_value = sum(values) / len(values)
    variance = sum((v - mean_value) ** 2 for v in values) / len(values)
    std_value = variance**0.5

    # Basic statistical checks
    if mean_value <= 0:
        return ValidationResult(
            False, f"mean transaction value is non-positive: {mean_value:.2f}"
        )

    # Coefficient of variation (CV) should be reasonable for retail data
    # Typical retail CV is 0.5 to 2.0
    cv = std_value / mean_value if mean_value > 0 else 0
    if cv < 0.1 or cv > 5.0:
        return ValidationResult(
            False,
            f"coefficient of variation unrealistic: CV={cv:.2f} (expected 0.1-5.0)",
        )

    # Check against expected values if provided
    if expected_mean is not None:
        mean_diff = abs(mean_value - expected_mean) / expected_mean
        if mean_diff > tolerance:
            return ValidationResult(
                False,
                f"mean differs from expected: {mean_value:.2f} vs {expected_mean:.2f} "
                f"(diff={mean_diff:.1%} > {tolerance:.1%})",
            )

    if expected_std is not None:
        std_diff = abs(std_value - expected_std) / expected_std
        if std_diff > tolerance:
            return ValidationResult(
                False,
                f"std differs from expected: {std_value:.2f} vs {expected_std:.2f} "
                f"(diff={std_diff:.1%} > {tolerance:.1%})",
            )

    return ValidationResult(
        True,
        f"spend distribution realistic: mean={mean_value:.2f}, std={std_value:.2f}, CV={cv:.2f}",
    )


def check_cohort_decay_pattern(
    transactions: Sequence[Transaction],
    customers: Sequence[Customer],
    *,
    max_expected_churn_rate: float = 0.5,
) -> ValidationResult:
    """Validate that customer cohorts show realistic decay patterns over time.

    Parameters
    ----------
    transactions:
        Transaction data to analyze.
    customers:
        Customer data with acquisition dates.
    max_expected_churn_rate:
        Maximum acceptable monthly churn rate (e.g., 0.5 = 50% per month).

    Returns
    -------
    ValidationResult
        Validation result with cohort decay statistics.
    """
    if not transactions or not customers:
        return ValidationResult(False, "insufficient data for cohort decay analysis")

    # Group customers by acquisition month
    from collections import defaultdict

    cohorts: dict[tuple[int, int], list[str]] = defaultdict(list)
    for c in customers:
        month_key = (c.acquisition_date.year, c.acquisition_date.month)
        cohorts[month_key].append(c.customer_id)

    # Count active customers by cohort and month
    cohort_activity: dict[tuple[tuple[int, int], tuple[int, int]], set[str]] = (
        defaultdict(set)
    )
    for t in transactions:
        txn_month = (t.event_ts.year, t.event_ts.month)
        # Find which cohort this customer belongs to
        for cohort_month, cohort_customers in cohorts.items():
            if t.customer_id in cohort_customers:
                cohort_activity[(cohort_month, txn_month)].add(t.customer_id)
                break

    # Check decay rates for each cohort
    for cohort_month, cohort_ids in cohorts.items():
        cohort_size = len(cohort_ids)
        if cohort_size == 0:
            continue

        # Track retention over months following acquisition
        months_after_acquisition = []
        retention_rates = []

        for activity_key, active_ids in cohort_activity.items():
            coh_month, txn_month = activity_key
            if coh_month != cohort_month:
                continue

            # Calculate months since acquisition
            months_diff = (txn_month[0] - cohort_month[0]) * 12 + (
                txn_month[1] - cohort_month[1]
            )
            if months_diff >= 0:
                retention = len(active_ids) / cohort_size
                months_after_acquisition.append(months_diff)
                retention_rates.append(retention)

        # Check that retention doesn't increase dramatically (unrealistic recovery)
        # Note: Small increases are acceptable (reactivations, seasonality)
        # We only flag increases > 2x from one period to next
        if len(retention_rates) >= 2:
            for i in range(1, len(retention_rates)):
                if retention_rates[i] > retention_rates[i - 1] * 2.0:
                    return ValidationResult(
                        False,
                        f"unrealistic retention increase in cohort {cohort_month}: "
                        f"{retention_rates[i - 1]:.2%} → {retention_rates[i]:.2%}",
                    )

    return ValidationResult(True, "cohort decay patterns are realistic")


def check_no_duplicate_transactions(
    transactions: Sequence[Transaction],
) -> ValidationResult:
    """Validate that there are no exact duplicate transactions.

    Note: Multiple transaction records can share the same order_id (representing
    different line items within the same order). This function checks for exact
    duplicates across all fields.

    Parameters
    ----------
    transactions:
        Transaction data to validate.

    Returns
    -------
    ValidationResult
        Validation result indicating if duplicates were found.
    """
    if not transactions:
        return ValidationResult(True, "no transactions to check")

    # Check for exact duplicates (same order_id, event_ts, customer_id, product_id, etc.)
    seen = set()
    duplicates = 0

    for t in transactions:
        # Create a hashable tuple of all transaction fields
        txn_tuple = (
            t.order_id,
            t.customer_id,
            t.event_ts,
            t.quantity,
            t.unit_price,
            t.product_id,
        )
        if txn_tuple in seen:
            duplicates += 1
        else:
            seen.add(txn_tuple)

    if duplicates > 0:
        return ValidationResult(
            False,
            f"found {duplicates} exact duplicate transactions out of {len(transactions)} total",
        )

    return ValidationResult(
        True,
        f"all {len(transactions)} transactions are unique (multi-line orders allowed)",
    )


def check_temporal_coverage(
    transactions: Sequence[Transaction],
    customers: Sequence[Customer],
    *,
    min_months_with_activity: int = 1,
) -> ValidationResult:
    """Validate that transactions span expected time periods.

    Parameters
    ----------
    transactions:
        Transaction data to validate.
    customers:
        Customer data to check against.
    min_months_with_activity:
        Minimum number of months that should have at least one transaction.

    Returns
    -------
    ValidationResult
        Validation result with temporal coverage statistics.
    """
    if not transactions:
        return ValidationResult(False, "no transactions to validate")

    # Get unique months with activity
    months_with_activity = set(
        (t.event_ts.year, t.event_ts.month) for t in transactions
    )

    if len(months_with_activity) < min_months_with_activity:
        return ValidationResult(
            False,
            f"insufficient temporal coverage: {len(months_with_activity)} months < "
            f"{min_months_with_activity} required",
        )

    # Check that transactions don't precede any customer acquisitions
    if customers:
        earliest_acquisition = min(c.acquisition_date for c in customers)
        earliest_txn = min(t.event_ts.date() for t in transactions)

        if earliest_txn < earliest_acquisition:
            return ValidationResult(
                False,
                f"transactions precede earliest acquisition: {earliest_txn} < {earliest_acquisition}",
            )

    return ValidationResult(
        True,
        f"temporal coverage adequate: {len(months_with_activity)} months with activity",
    )
