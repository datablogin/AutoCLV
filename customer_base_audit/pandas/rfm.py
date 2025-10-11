"""Pandas DataFrame adapters for RFM calculations."""

from typing import List, Optional, Sequence
from datetime import datetime
import pandas as pd  # type: ignore

from customer_base_audit.foundation.rfm import RFMMetrics, calculate_rfm
from customer_base_audit.foundation.data_mart import PeriodAggregation
from ._utils import decimal_to_float, float_to_decimal


def rfm_to_dataframe(rfm_metrics: Sequence[RFMMetrics]) -> pd.DataFrame:
    """Convert RFM metrics to pandas DataFrame.

    Args:
        rfm_metrics: Sequence of RFMMetrics objects

    Returns:
        DataFrame with columns: customer_id, recency_days, frequency,
        monetary, total_spend, observation_start, observation_end

    Example:
        >>> rfm_metrics = calculate_rfm(periods, datetime(2023, 12, 31))
        >>> rfm_df = rfm_to_dataframe(rfm_metrics)
        >>> rfm_df.head()
    """
    if not rfm_metrics:
        return pd.DataFrame(
            columns=[
                "customer_id",
                "recency_days",
                "frequency",
                "monetary",
                "total_spend",
                "observation_start",
                "observation_end",
            ]
        )

    rows = [
        {
            "customer_id": m.customer_id,
            "recency_days": m.recency_days,
            "frequency": m.frequency,
            "monetary": decimal_to_float(m.monetary),
            "total_spend": decimal_to_float(m.total_spend),
            "observation_start": m.observation_start,
            "observation_end": m.observation_end,
        }
        for m in rfm_metrics
    ]

    df = pd.DataFrame(rows)
    df = df.sort_values("customer_id").reset_index(drop=True)
    return df


def dataframe_to_rfm(rfm_df: pd.DataFrame) -> List[RFMMetrics]:
    """Convert pandas DataFrame to RFM metrics.

    Args:
        rfm_df: DataFrame with RFM columns

    Returns:
        List of validated RFMMetrics objects with schema:
        - customer_id: str
        - recency_days: int
        - frequency: int
        - monetary: float (converted to Decimal)
        - total_spend: float (converted to Decimal)
        - observation_start: datetime64[ns] (converted to datetime)
        - observation_end: datetime64[ns] (converted to datetime)

    Raises:
        ValueError: If DataFrame missing required columns, has null values, or invalid data

    Example:
        >>> rfm_metrics = dataframe_to_rfm(rfm_df)
        >>> lens1 = analyze_single_period(rfm_metrics)

    Note:
        Output is sorted by customer_id (lexicographic order). For numeric IDs,
        "C2" < "C10" in the sort order.
    """
    required_cols = [
        "customer_id",
        "recency_days",
        "frequency",
        "monetary",
        "total_spend",
        "observation_start",
        "observation_end",
    ]

    missing_cols = set(required_cols) - set(rfm_df.columns)
    if missing_cols:
        raise ValueError(f"DataFrame missing required columns: {missing_cols}")

    if rfm_df.empty:
        return []

    # Validate for null/NaN values
    null_cols = rfm_df[required_cols].isnull().any()
    if null_cols.any():
        null_col_names = null_cols[null_cols].index.tolist()
        raise ValueError(
            f"Null/NaN values found in columns: {null_col_names}. "
            "RFM calculations require complete data."
        )

    rfm_metrics = []
    for record in rfm_df.to_dict("records"):
        rfm_metrics.append(
            RFMMetrics(
                customer_id=str(record["customer_id"]),
                recency_days=int(record["recency_days"]),
                frequency=int(record["frequency"]),
                monetary=float_to_decimal(record["monetary"]),
                observation_start=pd.to_datetime(
                    record["observation_start"]
                ).to_pydatetime(),
                observation_end=pd.to_datetime(
                    record["observation_end"]
                ).to_pydatetime(),
                total_spend=float_to_decimal(record["total_spend"]),
            )
        )

    return rfm_metrics


def dataframe_to_period_aggregations(
    periods_df: pd.DataFrame,
    customer_id_col: str = "customer_id",
    period_start_col: str = "period_start",
    period_end_col: str = "period_end",
    total_orders_col: str = "total_orders",
    total_spend_col: str = "total_spend",
    total_margin_col: str = "total_margin",
    total_quantity_col: str = "total_quantity",
) -> List[PeriodAggregation]:
    """Convert pandas DataFrame to PeriodAggregation list.

    Args:
        periods_df: DataFrame with period aggregation data
        *_col: Column name mappings for flexibility

    Returns:
        List of PeriodAggregation objects with mapped schema:
        - customer_id: str (from customer_id_col)
        - period_start: datetime64[ns] (from period_start_col, converted to datetime)
        - period_end: datetime64[ns] (from period_end_col, converted to datetime)
        - total_orders: int (from total_orders_col)
        - total_spend: float (from total_spend_col)
        - total_margin: float (from total_margin_col)
        - total_quantity: int (from total_quantity_col)

    Raises:
        ValueError: If DataFrame missing required columns, has null values, or invalid data

    Example:
        >>> periods_df = pd.read_csv('customer_periods.csv')
        >>> periods = dataframe_to_period_aggregations(periods_df)
        >>> rfm = calculate_rfm(periods, datetime(2023, 12, 31))

    Example with custom column names:
        >>> periods = dataframe_to_period_aggregations(
        ...     df,
        ...     customer_id_col='client_id',
        ...     total_spend_col='revenue'
        ... )
    """
    required_mapping = {
        "customer_id": customer_id_col,
        "period_start": period_start_col,
        "period_end": period_end_col,
        "total_orders": total_orders_col,
        "total_spend": total_spend_col,
        "total_margin": total_margin_col,
        "total_quantity": total_quantity_col,
    }

    missing_cols = set(required_mapping.values()) - set(periods_df.columns)
    if missing_cols:
        raise ValueError(f"DataFrame missing required columns: {missing_cols}")

    if periods_df.empty:
        return []

    # Validate for null/NaN values
    required_cols = list(required_mapping.values())
    null_cols = periods_df[required_cols].isnull().any()
    if null_cols.any():
        null_col_names = null_cols[null_cols].index.tolist()
        raise ValueError(
            f"Null/NaN values found in columns: {null_col_names}. "
            "Period aggregations require complete data."
        )

    periods = []
    for record in periods_df.to_dict("records"):
        periods.append(
            PeriodAggregation(
                customer_id=str(record[customer_id_col]),
                period_start=pd.to_datetime(record[period_start_col]).to_pydatetime(),
                period_end=pd.to_datetime(record[period_end_col]).to_pydatetime(),
                total_orders=int(record[total_orders_col]),
                total_spend=float(record[total_spend_col]),
                total_margin=float(record[total_margin_col]),
                total_quantity=int(record[total_quantity_col]),
            )
        )

    return periods


def calculate_rfm_df(
    periods_df: pd.DataFrame,
    observation_end: datetime,
    customer_id_col: str = "customer_id",
    period_start_col: str = "period_start",
    period_end_col: str = "period_end",
    total_orders_col: str = "total_orders",
    total_spend_col: str = "total_spend",
    total_margin_col: str = "total_margin",
    total_quantity_col: str = "total_quantity",
    parallel: bool = True,
    parallel_threshold: int = 10_000_000,
    n_workers: Optional[int] = None,
) -> pd.DataFrame:
    """Calculate RFM metrics from a pandas DataFrame.

    Convenience function that combines conversion and calculation.

    Args:
        periods_df: DataFrame with period aggregations
        observation_end: End date for recency calculation
        *_col: Column name mappings for flexibility
        parallel: Enable parallel processing (default: True)
        parallel_threshold: Customer count threshold for parallel processing (default: 10M)
        n_workers: Number of worker processes (default: CPU count)

    Returns:
        DataFrame with RFM metrics

    Example:
        >>> periods_df = pd.read_parquet('periods.parquet')
        >>> rfm_df = calculate_rfm_df(periods_df, datetime(2023, 12, 31))
        >>> high_value = rfm_df[rfm_df['monetary'] > 100]

    Example with parallel processing:
        >>> # Force parallel for 100k+ customers with 8 workers
        >>> rfm_df = calculate_rfm_df(periods_df, datetime(2023, 12, 31),
        ...                           parallel=True, parallel_threshold=100_000, n_workers=8)
    """
    # Convert DataFrame → List[PeriodAggregation]
    periods = dataframe_to_period_aggregations(
        periods_df,
        customer_id_col=customer_id_col,
        period_start_col=period_start_col,
        period_end_col=period_end_col,
        total_orders_col=total_orders_col,
        total_spend_col=total_spend_col,
        total_margin_col=total_margin_col,
        total_quantity_col=total_quantity_col,
    )

    # Calculate RFM using core API with parallel processing support
    rfm_metrics = calculate_rfm(
        periods,
        observation_end,
        parallel=parallel,
        parallel_threshold=parallel_threshold,
        n_workers=n_workers,
    )

    # Convert List[RFMMetrics] → DataFrame
    return rfm_to_dataframe(rfm_metrics)
