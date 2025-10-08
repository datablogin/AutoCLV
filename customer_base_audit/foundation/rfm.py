"""RFM (Recency-Frequency-Monetary) calculation utilities.

RFM analysis segments customers based on three dimensions:
- Recency: How recently did the customer make a purchase?
- Frequency: How often do they purchase?
- Monetary: How much do they spend?

These metrics are foundational for customer segmentation, CLV modeling,
and the Five Lenses framework from "The Customer-Base Audit".
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
from typing import Sequence

import pandas as pd  # Used for quantile-based RFM scoring (qcut)

from customer_base_audit.foundation.data_mart import PeriodAggregation


@dataclass(frozen=True)
class RFMMetrics:
    """RFM metrics for a single customer.

    Attributes
    ----------
    customer_id:
        Unique customer identifier
    recency_days:
        Days since last purchase (from observation_end)
    frequency:
        Total number of purchases in observation period
    monetary:
        Average transaction value (total_spend / frequency)
    observation_start:
        Start date of observation period
    observation_end:
        End date of observation period
    total_spend:
        Total spend in observation period
    """

    customer_id: str
    recency_days: int
    frequency: int
    monetary: Decimal
    observation_start: datetime
    observation_end: datetime
    total_spend: Decimal

    def __post_init__(self) -> None:
        """Validate RFM metrics."""
        if self.recency_days < 0:
            raise ValueError(
                f"Recency cannot be negative: {self.recency_days} (customer_id={self.customer_id})"
            )
        if self.frequency <= 0:
            raise ValueError(
                f"Frequency must be positive: {self.frequency} (customer_id={self.customer_id})"
            )
        if self.monetary < 0:
            raise ValueError(
                f"Monetary value cannot be negative: {self.monetary} (customer_id={self.customer_id})"
            )
        if self.total_spend < 0:
            raise ValueError(
                f"Total spend cannot be negative: {self.total_spend} (customer_id={self.customer_id})"
            )
        # Validate monetary = total_spend / frequency (within rounding tolerance)
        expected_monetary = self.total_spend / self.frequency
        tolerance = Decimal("0.01")
        if abs(self.monetary - expected_monetary) > tolerance:
            raise ValueError(
                f"Monetary ({self.monetary}) != total_spend / frequency ({expected_monetary}) (customer_id={self.customer_id})"
            )


def calculate_rfm(
    period_aggregations: Sequence[PeriodAggregation],
    observation_end: datetime,
) -> list[RFMMetrics]:
    """Calculate RFM metrics from period aggregations.

    **Note on Recency Calculation**: Since PeriodAggregation does not include
    the exact transaction timestamp, recency is approximated using the end date
    of the last active period. This provides a conservative estimate (slightly
    understating recency) compared to using the actual last transaction date.

    For example, if a customer made a purchase on Dec 28 in a monthly period
    (Dec 1-31), recency will be calculated from Dec 31, not Dec 28. This is
    more accurate than using period_start (Dec 1) which would overstate recency
    by ~27 days.

    Parameters
    ----------
    period_aggregations:
        List of period-level customer aggregations. These should cover
        the full observation period for accurate RFM calculations.
    observation_end:
        End date of observation period. Recency is calculated as days
        from the end of the last active period to this date.

    Returns
    -------
    list[RFMMetrics]
        One RFMMetrics per customer, sorted by customer_id

    Examples
    --------
    >>> from datetime import datetime
    >>> from customer_base_audit.foundation.data_mart import PeriodAggregation
    >>> periods = [
    ...     PeriodAggregation(
    ...         customer_id="C1",
    ...         period_start=datetime(2023, 1, 1),
    ...         period_end=datetime(2023, 2, 1),
    ...         total_orders=3,
    ...         total_spend=150.0,
    ...         total_margin=50.0,
    ...         total_quantity=10,
    ...     ),
    ...     PeriodAggregation(
    ...         customer_id="C1",
    ...         period_start=datetime(2023, 3, 1),
    ...         period_end=datetime(2023, 4, 1),
    ...         total_orders=2,
    ...         total_spend=100.0,
    ...         total_margin=30.0,
    ...         total_quantity=5,
    ...     ),
    ... ]
    >>> obs_end = datetime(2023, 4, 15)
    >>> rfm = calculate_rfm(periods, obs_end)
    >>> rfm[0].customer_id
    'C1'
    >>> rfm[0].frequency
    5
    >>> rfm[0].total_spend
    Decimal('250.00')
    """
    if not period_aggregations:
        return []

    # Group by customer_id
    customer_data: dict[str, dict] = {}
    for period in period_aggregations:
        customer_id = period.customer_id
        if customer_id not in customer_data:
            customer_data[customer_id] = {
                "last_period_end": period.period_end,
                "first_period_start": period.period_start,
                "total_orders": 0,
                "total_spend": Decimal("0"),
            }

        data = customer_data[customer_id]
        # Track the most recent period end (conservative proxy for last purchase)
        # Using period_end instead of period_start provides a more accurate
        # recency estimate since purchases could occur anywhere in the period
        if period.period_end > data["last_period_end"]:
            data["last_period_end"] = period.period_end
        # Track the earliest period start for observation_start
        if period.period_start < data["first_period_start"]:
            data["first_period_start"] = period.period_start

        data["total_orders"] += period.total_orders
        data["total_spend"] += Decimal(str(period.total_spend))

    # Calculate RFM metrics for each customer
    rfm_metrics: list[RFMMetrics] = []
    for customer_id, data in customer_data.items():
        # Recency: days from last period end to observation_end
        # This is a conservative estimate - actual recency may be slightly higher
        recency_delta = observation_end - data["last_period_end"]
        recency_days = recency_delta.days

        # Frequency: total number of orders
        frequency = data["total_orders"]

        # Total spend
        total_spend = data["total_spend"].quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )

        # Monetary: average transaction value
        monetary = (total_spend / frequency).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )

        rfm_metrics.append(
            RFMMetrics(
                customer_id=customer_id,
                recency_days=recency_days,
                frequency=frequency,
                monetary=monetary,
                observation_start=data["first_period_start"],
                observation_end=observation_end,
                total_spend=total_spend,
            )
        )

    # Sort by customer_id for consistency
    rfm_metrics.sort(key=lambda m: m.customer_id)
    return rfm_metrics


@dataclass(frozen=True)
class RFMScore:
    """RFM scores (1-5 quintiles) for a single customer.

    Attributes
    ----------
    customer_id:
        Unique customer identifier
    r_score:
        Recency score (1-5, where 5 = most recent)
    f_score:
        Frequency score (1-5, where 5 = most frequent)
    m_score:
        Monetary score (1-5, where 5 = highest spend)
    rfm_score:
        Combined RFM score string (e.g., "555" for best customers)
    """

    customer_id: str
    r_score: int
    f_score: int
    m_score: int
    rfm_score: str

    def __post_init__(self) -> None:
        """Validate RFM scores."""
        for score_name, score_value in [
            ("r_score", self.r_score),
            ("f_score", self.f_score),
            ("m_score", self.m_score),
        ]:
            if not 1 <= score_value <= 5:
                raise ValueError(
                    f"{score_name} must be between 1 and 5: {score_value} (customer_id={self.customer_id})"
                )
        expected_rfm = f"{self.r_score}{self.f_score}{self.m_score}"
        if self.rfm_score != expected_rfm:
            raise ValueError(
                f"rfm_score ({self.rfm_score}) does not match r/f/m scores ({expected_rfm}) (customer_id={self.customer_id})"
            )


def calculate_rfm_scores(
    rfm_metrics: Sequence[RFMMetrics],
    recency_bins: int = 5,
    frequency_bins: int = 5,
    monetary_bins: int = 5,
) -> list[RFMScore]:
    """Score RFM metrics into quintiles (1-5).

    Customers are ranked and divided into bins for each dimension.
    For recency, lower values (more recent) get higher scores.
    For frequency and monetary, higher values get higher scores.

    **Note on Small Datasets**: For small datasets (< 10 customers) or datasets
    with many duplicate values, pandas qcut with `duplicates="drop"` may produce
    fewer than 5 bins. This can result in score ranges like 1-3 instead of 1-5.
    The algorithm handles this gracefully, but be aware that bin distribution
    depends on data variety. For production use with CLV models, recommend
    datasets of 100+ customers for stable quintile binning.

    Parameters
    ----------
    rfm_metrics:
        List of RFM metrics to score. Recommend 100+ customers for stable
        quintile binning, though smaller datasets are supported.
    recency_bins:
        Number of bins for recency (default: 5 for quintiles)
    frequency_bins:
        Number of bins for frequency (default: 5 for quintiles)
    monetary_bins:
        Number of bins for monetary (default: 5 for quintiles)

    Returns
    -------
    list[RFMScore]
        RFM scores for each customer, sorted by customer_id. Scores will be
        in the range 1-5, but small datasets may not use all values.

    Examples
    --------
    >>> from decimal import Decimal
    >>> from datetime import datetime
    >>> metrics = [
    ...     RFMMetrics("C1", 10, 5, Decimal("50"), datetime(2023,1,1), datetime(2023,12,31), Decimal("250")),
    ...     RFMMetrics("C2", 30, 2, Decimal("75"), datetime(2023,1,1), datetime(2023,12,31), Decimal("150")),
    ... ]
    >>> scores = calculate_rfm_scores(metrics)
    >>> scores[0].customer_id
    'C1'
    >>> scores[0].r_score >= scores[1].r_score  # C1 more recent (lower recency_days)
    True
    """
    if not rfm_metrics:
        return []

    # Convert to DataFrame for easier quantile calculations
    data = {
        "customer_id": [m.customer_id for m in rfm_metrics],
        "recency_days": [m.recency_days for m in rfm_metrics],
        "frequency": [m.frequency for m in rfm_metrics],
        "monetary": [float(m.monetary) for m in rfm_metrics],
    }
    df = pd.DataFrame(data)

    # Calculate quintile scores
    # Note: Using duplicates="drop" means qcut may produce fewer bins than requested
    # when there are many duplicate values. We handle this by using labels=False
    # and manually mapping to 1-5 scale. For completely uniform values, assign score 3.

    # Helper function to safely score a dimension
    def safe_qcut_score(
        values: pd.Series, bins: int, reverse: bool = False
    ) -> pd.Series:
        """Score values using qcut, handling edge cases like uniform values."""
        unique_count = values.nunique()

        if unique_count == 1:
            # All values identical - assign middle score (3)
            return pd.Series([3] * len(values), index=values.index)

        # Use qcut with labels=False to get bin indices
        categories = pd.qcut(
            values, q=min(bins, unique_count), labels=False, duplicates="drop"
        )

        if reverse:
            # For recency: reverse mapping so lower values get higher scores
            max_bin = categories.max()
            return (max_bin - categories + 1).astype(int)
        else:
            # For frequency/monetary: map 0->1, 1->2, etc.
            return (categories + 1).astype(int)

    # Recency: lower is better, so we reverse the scoring (5 = most recent)
    df["r_score"] = safe_qcut_score(df["recency_days"], recency_bins, reverse=True)

    # Frequency: higher is better
    df["f_score"] = safe_qcut_score(df["frequency"], frequency_bins, reverse=False)

    # Monetary: higher is better
    df["m_score"] = safe_qcut_score(df["monetary"], monetary_bins, reverse=False)

    # Create combined RFM score string
    df["rfm_score"] = (
        df["r_score"].astype(str)
        + df["f_score"].astype(str)
        + df["m_score"].astype(str)
    )

    # Convert to RFMScore objects
    rfm_scores: list[RFMScore] = []
    for _, row in df.iterrows():
        rfm_scores.append(
            RFMScore(
                customer_id=row["customer_id"],
                r_score=row["r_score"],
                f_score=row["f_score"],
                m_score=row["m_score"],
                rfm_score=row["rfm_score"],
            )
        )

    # Sort by customer_id for consistency
    rfm_scores.sort(key=lambda s: s.customer_id)
    return rfm_scores
