"""RFM (Recency-Frequency-Monetary) calculation utilities.

RFM analysis segments customers based on three dimensions:
- Recency: How recently did the customer make a purchase?
- Frequency: How often do they purchase?
- Monetary: How much do they spend?

These metrics are foundational for customer segmentation, CLV modeling,
and the Five Lenses framework from "The Customer-Base Audit".
"""

from __future__ import annotations

import multiprocessing
import os
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
from typing import Optional, Sequence

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


def _calculate_rfm_for_customers(
    customer_data_chunk: dict[str, dict], observation_end: datetime
) -> list[RFMMetrics]:
    """Helper function to calculate RFM metrics for a chunk of customers.

    This function is designed to be called by multiprocessing workers.
    It processes a subset of customers independently.

    Parameters
    ----------
    customer_data_chunk:
        Dictionary mapping customer_id to aggregated customer data
        (last_transaction_ts, last_period_end, first_period_start, total_orders, total_spend)
    observation_end:
        End date for recency calculation

    Returns
    -------
    list[RFMMetrics]
        RFM metrics for customers in this chunk (unsorted)
    """
    rfm_metrics: list[RFMMetrics] = []

    for customer_id, data in customer_data_chunk.items():
        # Recency: days from last transaction to observation_end
        # Use actual last_transaction_ts if available, otherwise fall back to period_end
        if data["last_transaction_ts"] is not None:
            recency_reference = data["last_transaction_ts"]
        else:
            # Backward compatibility: use period_end approximation
            recency_reference = data["last_period_end"]

        recency_delta = observation_end - recency_reference
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

    return rfm_metrics


def calculate_rfm(
    period_aggregations: Sequence[PeriodAggregation],
    observation_end: datetime,
    parallel: bool = True,
    parallel_threshold: int = 10_000_000,
    n_workers: Optional[int] = None,
) -> list[RFMMetrics]:
    """Calculate RFM metrics from period aggregations.

    **Note on Recency Calculation**: This function uses the actual last transaction
    timestamp from PeriodAggregation.last_transaction_ts when available, providing
    accurate recency calculations for CLV modeling. If last_transaction_ts is None
    (backward compatibility with older data), it falls back to using period_end as
    a conservative approximation.

    For example, if a customer made a purchase on Dec 28 in a monthly period
    (Dec 1-31):
    - With last_transaction_ts: recency calculated from Dec 28 (accurate)
    - Without last_transaction_ts: recency calculated from Dec 31 (conservative,
      understates recency by 3 days)

    **Timezone Assumptions**: All datetime values must use the same timezone
    (or all be timezone-naive). For production use, UTC timestamps are recommended.

    **Data Quality**: This function validates that transaction timestamps do not
    occur after observation_end. Future-dated transactions raise a ValueError.

    **Parallel Processing**: For large datasets (>10M customers by default), this
    function automatically enables parallel processing using multiprocessing to
    improve performance. The parallelization is done at the customer level, so
    each worker processes a subset of customers independently. You can control this
    behavior using the parallel, parallel_threshold, and n_workers parameters.

    Parameters
    ----------
    period_aggregations:
        List of period-level customer aggregations. These should cover
        the full observation period for accurate RFM calculations.
    observation_end:
        End date of observation period. Recency is calculated as days
        from the end of the last active period to this date.
    parallel:
        Enable parallel processing (default: True). When enabled, uses
        multiprocessing for datasets exceeding parallel_threshold.
    parallel_threshold:
        Number of customers above which to enable parallel processing
        (default: 10,000,000). Lower values enable parallelization for
        smaller datasets; higher values delay it for larger datasets.
    n_workers:
        Number of worker processes for parallel processing. If None
        (default), uses CPU count. Ignored if parallel=False.

    Returns
    -------
    list[RFMMetrics]
        One RFMMetrics per customer, sorted by customer_id

    Examples
    --------
    Basic usage (serial processing for small datasets):

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

    Force parallel processing for smaller datasets:

    >>> # Process 100k customers in parallel with 4 workers
    >>> rfm = calculate_rfm(periods, obs_end, parallel=True,
    ...                      parallel_threshold=100_000, n_workers=4)

    Disable parallel processing entirely:

    >>> # Always use serial processing (useful for debugging)
    >>> rfm = calculate_rfm(periods, obs_end, parallel=False)
    """
    if not period_aggregations:
        return []

    # Group by customer_id
    customer_data: dict[str, dict] = {}
    for period in period_aggregations:
        customer_id = period.customer_id
        if customer_id not in customer_data:
            customer_data[customer_id] = {
                "last_transaction_ts": None,
                "last_period_end": period.period_end,
                "first_period_start": period.period_start,
                "total_orders": 0,
                "total_spend": Decimal("0"),
            }

        data = customer_data[customer_id]

        # Track the most recent transaction timestamp (accurate recency)
        # Fall back to period_end if last_transaction_ts is not available
        if period.last_transaction_ts is not None:
            # Validate transaction timestamp is not in the future
            if period.last_transaction_ts > observation_end:
                raise ValueError(
                    f"Transaction timestamp ({period.last_transaction_ts}) cannot be after "
                    f"observation_end ({observation_end}) for customer {customer_id}"
                )

            if (
                data["last_transaction_ts"] is None
                or period.last_transaction_ts > data["last_transaction_ts"]
            ):
                data["last_transaction_ts"] = period.last_transaction_ts

        # Track the most recent period end (fallback for recency calculation)
        if period.period_end > data["last_period_end"]:
            data["last_period_end"] = period.period_end

        # Track the earliest period start for observation_start
        if period.period_start < data["first_period_start"]:
            data["first_period_start"] = period.period_start

        data["total_orders"] += period.total_orders
        data["total_spend"] += Decimal(str(period.total_spend))

    # Determine if we should use parallel processing
    num_customers = len(customer_data)
    use_parallel = parallel and num_customers >= parallel_threshold

    # Calculate RFM metrics for each customer
    if use_parallel:
        # Parallel processing for large datasets
        # Determine number of workers
        if n_workers is None:
            workers = os.cpu_count() or 1
        else:
            workers = max(1, n_workers)

        # Chunk customer data for parallel processing
        customer_items = list(customer_data.items())
        chunk_size = max(1, num_customers // workers)
        chunks = []
        for i in range(0, num_customers, chunk_size):
            chunk_dict = dict(customer_items[i : i + chunk_size])
            chunks.append((chunk_dict, observation_end))

        # Process chunks in parallel
        with multiprocessing.Pool(processes=workers) as pool:
            chunk_results = pool.starmap(_calculate_rfm_for_customers, chunks)

        # Merge results from all workers
        rfm_metrics: list[RFMMetrics] = []
        for chunk_result in chunk_results:
            rfm_metrics.extend(chunk_result)
    else:
        # Serial processing for small datasets (original implementation)
        rfm_metrics = _calculate_rfm_for_customers(customer_data, observation_end)

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
