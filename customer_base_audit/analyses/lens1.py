"""Lens 1: Single Period Analysis.

Provides a snapshot view of the customer base within a single time period,
answering questions like:
- How many customers are there?
- What percentage are one-time buyers?
- How concentrated is revenue? (Pareto analysis)
- What is the distribution across RFM segments?

This is the foundational lens from "The Customer-Base Audit" framework.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from typing import Sequence

from customer_base_audit.foundation.rfm import RFMMetrics, RFMScore

# Standard percentage precision: 2 decimal places (e.g., 45.67%)
# Note: 2 decimal places may not be granular enough for highly concentrated
# customer bases where top 0.1% or 0.01% might be critical for business decisions.
# For such cases, consider using calculate_revenue_concentration() directly with
# custom precision if needed.
PERCENTAGE_PRECISION = Decimal("0.01")

# Default Pareto analysis percentiles
# The Pareto principle suggests ~80% of revenue comes from ~20% of customers
# These are the standard percentiles used in customer base audits
DEFAULT_PARETO_PERCENTILES = (10, 20)

# Minimum customer count for revenue concentration calculations
# Even for very small percentiles, we need at least 1 customer
MIN_CUSTOMERS_IN_SEGMENT = 1


@dataclass(frozen=True)
class Lens1Metrics:
    """Lens 1: Single period analysis results.

    Attributes
    ----------
    total_customers:
        Total unique customers in the period
    one_time_buyers:
        Number of customers with only 1 purchase
    one_time_buyer_pct:
        Percentage of customers who are one-time buyers
    total_revenue:
        Total revenue across all customers
    top_10pct_revenue_contribution:
        Percentage of revenue from top 10% of customers
    top_20pct_revenue_contribution:
        Percentage of revenue from top 20% of customers
    avg_orders_per_customer:
        Average number of orders per customer
    median_customer_value:
        Median total spend per customer
    rfm_distribution:
        Count of customers in each RFM score category
    """

    total_customers: int
    one_time_buyers: int
    one_time_buyer_pct: Decimal
    total_revenue: Decimal
    top_10pct_revenue_contribution: Decimal
    top_20pct_revenue_contribution: Decimal
    avg_orders_per_customer: Decimal
    median_customer_value: Decimal
    rfm_distribution: dict[str, int]

    def __post_init__(self) -> None:
        """Validate Lens 1 metrics."""
        if self.total_customers < 0:
            raise ValueError(
                f"Total customers cannot be negative: {self.total_customers}"
            )
        if self.one_time_buyers < 0:
            raise ValueError(
                f"One-time buyers cannot be negative: {self.one_time_buyers}"
            )
        if self.one_time_buyers > self.total_customers:
            raise ValueError(
                f"One-time buyers ({self.one_time_buyers}) cannot exceed total customers ({self.total_customers})"
            )
        if self.total_revenue < 0:
            raise ValueError(f"Total revenue cannot be negative: {self.total_revenue}")
        if not 0 <= self.one_time_buyer_pct <= 100:
            raise ValueError(
                f"One-time buyer percentage must be 0-100: {self.one_time_buyer_pct}"
            )
        if not 0 <= self.top_10pct_revenue_contribution <= 100:
            raise ValueError(
                f"Top 10% revenue contribution must be 0-100: {self.top_10pct_revenue_contribution}"
            )
        if not 0 <= self.top_20pct_revenue_contribution <= 100:
            raise ValueError(
                f"Top 20% revenue contribution must be 0-100: {self.top_20pct_revenue_contribution}"
            )


def analyze_single_period(
    rfm_metrics: Sequence[RFMMetrics],
    rfm_scores: Sequence[RFMScore] | None = None,
) -> Lens1Metrics:
    """Perform Lens 1 analysis on a single period.

    Parameters
    ----------
    rfm_metrics:
        RFM metrics for all customers in the period
    rfm_scores:
        Optional RFM scores for segmentation analysis. If not provided,
        rfm_distribution will be empty.

    Returns
    -------
    Lens1Metrics
        Comprehensive single-period analysis results

    Examples
    --------
    >>> from decimal import Decimal
    >>> from datetime import datetime
    >>> from customer_base_audit.foundation.rfm import RFMMetrics
    >>> metrics = [
    ...     RFMMetrics("C1", 10, 5, Decimal("50"), datetime(2023,1,1), datetime(2023,12,31), Decimal("250")),
    ...     RFMMetrics("C2", 30, 1, Decimal("100"), datetime(2023,1,1), datetime(2023,12,31), Decimal("100")),
    ...     RFMMetrics("C3", 5, 10, Decimal("75"), datetime(2023,1,1), datetime(2023,12,31), Decimal("750")),
    ... ]
    >>> lens1 = analyze_single_period(metrics)
    >>> lens1.total_customers
    3
    >>> lens1.one_time_buyers
    1
    >>> float(lens1.one_time_buyer_pct)
    33.33
    """
    if not rfm_metrics:
        return Lens1Metrics(
            total_customers=0,
            one_time_buyers=0,
            one_time_buyer_pct=Decimal("0"),
            total_revenue=Decimal("0"),
            top_10pct_revenue_contribution=Decimal("0"),
            top_20pct_revenue_contribution=Decimal("0"),
            avg_orders_per_customer=Decimal("0"),
            median_customer_value=Decimal("0"),
            rfm_distribution={},
        )

    # Basic counts
    total_customers = len(rfm_metrics)
    one_time_buyers = sum(1 for m in rfm_metrics if m.frequency == 1)
    one_time_buyer_pct = (
        Decimal(one_time_buyers) / Decimal(total_customers) * 100
    ).quantize(PERCENTAGE_PRECISION, rounding=ROUND_HALF_UP)

    # Revenue metrics
    total_revenue = sum(m.total_spend for m in rfm_metrics).quantize(
        Decimal("0.01"), rounding=ROUND_HALF_UP
    )

    # Revenue concentration
    revenue_concentration = calculate_revenue_concentration(
        rfm_metrics, percentiles=DEFAULT_PARETO_PERCENTILES
    )
    top_10pct_revenue_contribution = revenue_concentration[
        DEFAULT_PARETO_PERCENTILES[0]
    ]
    top_20pct_revenue_contribution = revenue_concentration[
        DEFAULT_PARETO_PERCENTILES[1]
    ]

    # Order statistics
    total_orders = sum(m.frequency for m in rfm_metrics)
    avg_orders_per_customer = (
        Decimal(total_orders) / Decimal(total_customers)
    ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

    # Median customer value
    customer_values = sorted([m.total_spend for m in rfm_metrics])
    median_idx = len(customer_values) // 2
    if len(customer_values) % 2 == 0:
        # Average of two middle values for even count
        median_customer_value = (
            (customer_values[median_idx - 1] + customer_values[median_idx]) / 2
        ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    else:
        median_customer_value = customer_values[median_idx]

    # RFM distribution
    rfm_distribution: dict[str, int] = {}
    if rfm_scores:
        for score in rfm_scores:
            rfm_distribution[score.rfm_score] = (
                rfm_distribution.get(score.rfm_score, 0) + 1
            )

    return Lens1Metrics(
        total_customers=total_customers,
        one_time_buyers=one_time_buyers,
        one_time_buyer_pct=one_time_buyer_pct,
        total_revenue=total_revenue,
        top_10pct_revenue_contribution=top_10pct_revenue_contribution,
        top_20pct_revenue_contribution=top_20pct_revenue_contribution,
        avg_orders_per_customer=avg_orders_per_customer,
        median_customer_value=median_customer_value,
        rfm_distribution=rfm_distribution,
    )


def calculate_revenue_concentration(
    rfm_metrics: Sequence[RFMMetrics],
    percentiles: Sequence[int] = DEFAULT_PARETO_PERCENTILES,
) -> dict[int, Decimal]:
    """Calculate what percentage of revenue comes from top N% of customers.

    This implements a Lorenz curve / Pareto analysis to understand
    revenue concentration. The Pareto principle suggests that ~80%
    of revenue comes from ~20% of customers.

    Parameters
    ----------
    rfm_metrics:
        RFM metrics for all customers
    percentiles:
        Top percentiles to calculate (e.g., [10, 20] for top 10% and 20%)

    Returns
    -------
    dict[int, Decimal]
        Mapping of percentile to revenue contribution percentage.
        Example: {10: Decimal('45.20'), 20: Decimal('62.80')}
        means top 10% of customers drive 45.20% of revenue,
        and top 20% drive 62.80% of revenue.

    Examples
    --------
    >>> from decimal import Decimal
    >>> from datetime import datetime
    >>> from customer_base_audit.foundation.rfm import RFMMetrics
    >>> metrics = [
    ...     RFMMetrics("C1", 10, 1, Decimal("100"), datetime(2023,1,1), datetime(2023,12,31), Decimal("100")),
    ...     RFMMetrics("C2", 20, 1, Decimal("200"), datetime(2023,1,1), datetime(2023,12,31), Decimal("200")),
    ...     RFMMetrics("C3", 30, 1, Decimal("700"), datetime(2023,1,1), datetime(2023,12,31), Decimal("700")),
    ... ]
    >>> concentration = calculate_revenue_concentration(metrics, percentiles=[33])
    >>> float(concentration[33])  # Top 33% (1 customer) has 70% of revenue
    70.0
    """
    if not rfm_metrics:
        return {p: Decimal("0") for p in percentiles}

    # Sort customers by total spend (descending)
    sorted_metrics = sorted(rfm_metrics, key=lambda m: m.total_spend, reverse=True)

    total_revenue = sum(m.total_spend for m in rfm_metrics)
    if total_revenue == 0:
        return {p: Decimal("0") for p in percentiles}

    concentration: dict[int, Decimal] = {}
    for percentile in percentiles:
        # Calculate number of customers in top N%
        # Use MIN_CUSTOMERS_IN_SEGMENT to ensure at least 1 customer is included
        # (e.g., 0.5% of 100 customers = 0.5 -> rounds to 0, but we need >= 1)
        top_n_count = max(
            MIN_CUSTOMERS_IN_SEGMENT, int(len(sorted_metrics) * percentile / 100)
        )

        # Sum revenue from top N% customers
        top_n_revenue = sum(m.total_spend for m in sorted_metrics[:top_n_count])

        # Calculate percentage contribution
        contribution_pct = (top_n_revenue / total_revenue * 100).quantize(
            PERCENTAGE_PRECISION, rounding=ROUND_HALF_UP
        )
        concentration[percentile] = contribution_pct

    return concentration
