"""Model input preparation for BG/NBD and Gamma-Gamma models.

This module transforms CustomerDataMart period aggregations into the specific
input formats required by probabilistic CLV models (BG/NBD for purchase frequency
and Gamma-Gamma for monetary value prediction).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
from typing import Sequence

import pandas as pd

from customer_base_audit.foundation.data_mart import PeriodAggregation


@dataclass(frozen=True)
class BGNBDInput:
    """Input for BG/NBD model (Beta-Geometric/Negative Binomial Distribution).

    The BG/NBD model predicts customer purchase frequency by modeling two processes:
    1. Transaction process (while alive): Poisson with rate λ
    2. Dropout process: Geometric with probability p after each transaction

    Attributes
    ----------
    customer_id:
        Unique customer identifier
    frequency:
        Number of repeat purchases (total_orders - 1). BG/NBD models repeat
        behavior, so the first purchase is excluded.
    recency:
        Time of last purchase relative to first purchase (in observation units).
        For a customer who made purchases at T=0 and T=30 days, recency=30.
    T:
        Total observation period length (time from first purchase to observation end).
        For a customer acquired on Day 0 observed through Day 90, T=90.
    """

    customer_id: str
    frequency: int
    recency: float
    T: float

    def __post_init__(self) -> None:
        """Validate BG/NBD inputs."""
        if self.frequency < 0:
            raise ValueError(
                f"Frequency cannot be negative: {self.frequency} (customer_id={self.customer_id})"
            )
        if self.recency < 0:
            raise ValueError(
                f"Recency cannot be negative: {self.recency} (customer_id={self.customer_id})"
            )
        if self.T <= 0:
            raise ValueError(
                f"T must be positive: {self.T} (customer_id={self.customer_id})"
            )
        # With period-level aggregations, recency might slightly exceed T due to period boundaries.
        # Cap recency at T instead of erroring to handle this approximation gracefully.
        if self.recency > self.T:
            object.__setattr__(self, 'recency', self.T)


@dataclass(frozen=True)
class GammaGammaInput:
    """Input for Gamma-Gamma model (monetary value prediction).

    The Gamma-Gamma model predicts average transaction value by assuming
    transaction values vary randomly around each customer's latent mean spend.

    Attributes
    ----------
    customer_id:
        Unique customer identifier
    frequency:
        Number of transactions. Must be >= 2 (excludes one-time buyers).
        The Gamma-Gamma model requires multiple observations to estimate
        a customer's average spend.
    monetary_value:
        Average transaction value (total_spend / frequency)
    """

    customer_id: str
    frequency: int
    monetary_value: Decimal

    def __post_init__(self) -> None:
        """Validate Gamma-Gamma inputs."""
        if self.frequency < 2:
            raise ValueError(
                f"Frequency must be >= 2 for Gamma-Gamma model: {self.frequency} (customer_id={self.customer_id})"
            )
        if self.monetary_value <= 0:
            raise ValueError(
                f"Monetary value must be positive (>0) for Gamma-Gamma model: {self.monetary_value} (customer_id={self.customer_id})"
            )


def prepare_bg_nbd_inputs(
    period_aggregations: Sequence[PeriodAggregation],
    observation_start: datetime,
    observation_end: datetime,
) -> pd.DataFrame:
    """Convert period aggregations to BG/NBD input format.

    The BG/NBD model requires:
    - frequency: Number of repeat purchases (total_orders - 1)
    - recency: Time from first purchase to last purchase
    - T: Time from first purchase to observation_end

    **⚠️ Time Calculation Approximation Warning**:
    Since PeriodAggregation doesn't include exact transaction timestamps,
    we use period boundaries as conservative estimates:
    - First purchase time: period_start of earliest period
    - Last purchase time: period_end of most recent period

    **Known Limitations**:
    - `recency` will be OVERESTIMATED (using period_end instead of actual last transaction)
    - `T` will be OVERESTIMATED (using period_start instead of actual first transaction)
    - This introduces systematic bias in BG/NBD parameter estimation (λ, p)
    - May underestimate churn probability and overestimate future purchases

    For most accurate CLV predictions, use transaction-level data with exact timestamps.
    For period-aggregated data, this approximation provides reasonable estimates but
    should be validated against holdout data

    Parameters
    ----------
    period_aggregations:
        List of period-level customer aggregations covering the observation period
    observation_start:
        Start date of observation period (used for validation)
    observation_end:
        End date of observation period. T is calculated as time from each
        customer's first purchase to this date.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: customer_id, frequency, recency, T
        One row per customer. Includes customers with frequency=0 (no repeat purchases).

    Examples
    --------
    >>> from datetime import datetime
    >>> from customer_base_audit.foundation.data_mart import PeriodAggregation
    >>> periods = [
    ...     PeriodAggregation("C1", datetime(2023,1,1), datetime(2023,2,1), 2, 100.0, 30.0, 5),
    ...     PeriodAggregation("C1", datetime(2023,3,1), datetime(2023,4,1), 1, 50.0, 15.0, 3),
    ... ]
    >>> df = prepare_bg_nbd_inputs(periods, datetime(2023,1,1), datetime(2023,6,1))
    >>> df.loc[0, 'customer_id']
    'C1'
    >>> df.loc[0, 'frequency']  # total_orders (3) - 1 = 2
    2
    """
    if not period_aggregations:
        return pd.DataFrame(columns=["customer_id", "frequency", "recency", "T"])

    # Validate that periods fall within observation window
    for period in period_aggregations:
        if period.period_start < observation_start:
            raise ValueError(
                f"Period start ({period.period_start.isoformat()}) is before "
                f"observation_start ({observation_start.isoformat()}) "
                f"for customer {period.customer_id}"
            )
        if period.period_end > observation_end:
            raise ValueError(
                f"Period end ({period.period_end.isoformat()}) is after "
                f"observation_end ({observation_end.isoformat()}) "
                f"for customer {period.customer_id}"
            )

    # Group by customer to calculate metrics
    customer_data: dict[str, dict] = {}
    for period in period_aggregations:
        customer_id = period.customer_id

        # Validate period data quality
        if period.total_orders < 0:
            raise ValueError(
                f"Invalid total_orders: {period.total_orders} for customer {period.customer_id} "
                f"in period [{period.period_start.isoformat()}, {period.period_end.isoformat()}]. "
                f"total_orders must be non-negative."
            )
        if period.total_spend < 0:
            raise ValueError(
                f"Invalid total_spend: {period.total_spend} for customer {period.customer_id} "
                f"in period [{period.period_start.isoformat()}, {period.period_end.isoformat()}]. "
                f"total_spend must be non-negative."
            )

        if customer_id not in customer_data:
            customer_data[customer_id] = {
                "first_period_start": period.period_start,
                "last_period_end": period.period_end,
                "total_orders": 0,
            }

        data = customer_data[customer_id]
        # Track earliest period start (proxy for first purchase)
        if period.period_start < data["first_period_start"]:
            data["first_period_start"] = period.period_start
        # Track latest period end (proxy for last purchase)
        if period.period_end > data["last_period_end"]:
            data["last_period_end"] = period.period_end

        data["total_orders"] += period.total_orders

    # Calculate BG/NBD metrics for each customer
    rows: list[dict] = []
    for customer_id, data in customer_data.items():
        # Frequency: repeat purchases only (total - 1)
        # BG/NBD models the repeat purchase process, so first purchase is excluded.
        # This differs from Gamma-Gamma which uses total purchases to estimate
        # average monetary value across all transactions.
        frequency = max(0, data["total_orders"] - 1)

        # Recency: time from first purchase to last purchase (in days)
        # For single-purchase customers, recency = 0
        if frequency > 0:
            recency_delta = data["last_period_end"] - data["first_period_start"]
            recency = recency_delta.total_seconds() / 86400.0  # Convert to days
        else:
            recency = 0.0

        # T: total observation time from first purchase to observation_end (in days)
        T_delta = observation_end - data["first_period_start"]
        T = T_delta.total_seconds() / 86400.0  # Convert to days

        rows.append(
            {
                "customer_id": customer_id,
                "frequency": frequency,
                "recency": recency,
                "T": T,
            }
        )

    df = pd.DataFrame(rows)
    # Sort by customer_id for consistency
    df = df.sort_values("customer_id").reset_index(drop=True)
    return df


def prepare_gamma_gamma_inputs(
    period_aggregations: Sequence[PeriodAggregation],
    min_frequency: int = 2,
) -> pd.DataFrame:
    """Convert period aggregations to Gamma-Gamma input format.

    The Gamma-Gamma model requires customers with multiple transactions to
    estimate average monetary value. One-time buyers are excluded.

    **Monetary Value Calculation**:
    - monetary_value = total_spend / total_orders
    - Uses Decimal arithmetic with ROUND_HALF_UP for financial precision

    Parameters
    ----------
    period_aggregations:
        List of period-level customer aggregations
    min_frequency:
        Minimum number of transactions required (default: 2).
        Customers with fewer transactions are excluded.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: customer_id, frequency, monetary_value
        One row per customer with frequency >= min_frequency.
        Sorted by customer_id.
        Note: monetary_value column contains Decimal objects for financial precision.
        Use float(value) or .astype(float) if float conversion is needed.

    Examples
    --------
    >>> from datetime import datetime
    >>> from customer_base_audit.foundation.data_mart import PeriodAggregation
    >>> periods = [
    ...     PeriodAggregation("C1", datetime(2023,1,1), datetime(2023,2,1), 3, 150.0, 45.0, 10),
    ...     PeriodAggregation("C2", datetime(2023,1,1), datetime(2023,2,1), 1, 50.0, 15.0, 3),
    ... ]
    >>> df = prepare_gamma_gamma_inputs(periods, min_frequency=2)
    >>> len(df)  # Only C1 included (3 orders >= 2)
    1
    >>> df.loc[0, 'customer_id']
    'C1'
    >>> df.loc[0, 'frequency']
    3
    >>> df.loc[0, 'monetary_value']
    Decimal('50.00')
    """
    if not period_aggregations:
        return pd.DataFrame(columns=["customer_id", "frequency", "monetary_value"])

    # Group by customer to calculate total orders and spend
    customer_data: dict[str, dict] = {}
    for period in period_aggregations:
        customer_id = period.customer_id

        # Validate period data quality
        if period.total_orders < 0:
            raise ValueError(
                f"Invalid total_orders: {period.total_orders} for customer {period.customer_id} "
                f"in period [{period.period_start.isoformat()}, {period.period_end.isoformat()}]. "
                f"total_orders must be non-negative."
            )
        if period.total_spend < 0:
            raise ValueError(
                f"Invalid total_spend: {period.total_spend} for customer {period.customer_id} "
                f"in period [{period.period_start.isoformat()}, {period.period_end.isoformat()}]. "
                f"total_spend must be non-negative."
            )

        if customer_id not in customer_data:
            customer_data[customer_id] = {
                "total_orders": 0,
                "total_spend": Decimal("0"),
            }

        data = customer_data[customer_id]
        data["total_orders"] += period.total_orders
        data["total_spend"] += Decimal(str(period.total_spend))

    # Calculate Gamma-Gamma metrics, filtering by min_frequency
    rows: list[dict] = []
    for customer_id, data in customer_data.items():
        # Frequency: total purchases (NOT repeat purchases like BG/NBD)
        # Gamma-Gamma uses all transactions to estimate average spend per transaction.
        # This differs from BG/NBD which uses (total - 1) to model repeat behavior.
        frequency = data["total_orders"]

        # Exclude customers below min_frequency threshold
        if frequency < min_frequency:
            continue

        # Defensive check: frequency should never be 0 here due to min_frequency filter,
        # but add explicit validation for robustness
        if frequency == 0:
            raise ValueError(
                f"Cannot calculate monetary value with zero frequency "
                f"(customer_id={customer_id}). This should not occur if min_frequency >= 1."
            )

        # Monetary value: average transaction value
        monetary_value = (data["total_spend"] / frequency).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )

        rows.append(
            {
                "customer_id": customer_id,
                "frequency": frequency,
                "monetary_value": monetary_value,  # Keep as Decimal for precision
            }
        )

    df = pd.DataFrame(rows)
    # Sort by customer_id for consistency
    if not df.empty:
        df = df.sort_values("customer_id").reset_index(drop=True)
    return df
