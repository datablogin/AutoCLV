"""Pandas DataFrame adapters for RFM calculations."""

from typing import List, Sequence
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
        List of validated RFMMetrics objects

    Raises:
        ValueError: If DataFrame missing required columns or has invalid data

    Example:
        >>> rfm_metrics = dataframe_to_rfm(rfm_df)
        >>> lens1 = analyze_single_period(rfm_metrics)
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

    rfm_metrics = []
    for _, row in rfm_df.iterrows():
        rfm_metrics.append(
            RFMMetrics(
                customer_id=str(row["customer_id"]),
                recency_days=int(row["recency_days"]),
                frequency=int(row["frequency"]),
                monetary=float_to_decimal(row["monetary"]),
                observation_start=pd.to_datetime(
                    row["observation_start"]
                ).to_pydatetime(),
                observation_end=pd.to_datetime(row["observation_end"]).to_pydatetime(),
                total_spend=float_to_decimal(row["total_spend"]),
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
        List of PeriodAggregation objects

    Raises:
        ValueError: If DataFrame missing required columns

    Example:
        >>> periods_df = pd.read_csv('customer_periods.csv')
        >>> periods = dataframe_to_period_aggregations(periods_df)
        >>> rfm = calculate_rfm(periods, datetime(2023, 12, 31))
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

    periods = []
    for _, row in periods_df.iterrows():
        periods.append(
            PeriodAggregation(
                customer_id=str(row[customer_id_col]),
                period_start=pd.to_datetime(row[period_start_col]).to_pydatetime(),
                period_end=pd.to_datetime(row[period_end_col]).to_pydatetime(),
                total_orders=int(row[total_orders_col]),
                total_spend=float(row[total_spend_col]),
                total_margin=float(row[total_margin_col]),
                total_quantity=int(row[total_quantity_col]),
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
) -> pd.DataFrame:
    """Calculate RFM metrics from a pandas DataFrame.

    Convenience function that combines conversion and calculation.

    Args:
        periods_df: DataFrame with period aggregations
        observation_end: End date for recency calculation
        *_col: Column name mappings for flexibility

    Returns:
        DataFrame with RFM metrics

    Example:
        >>> periods_df = pd.read_parquet('periods.parquet')
        >>> rfm_df = calculate_rfm_df(periods_df, datetime(2023, 12, 31))
        >>> high_value = rfm_df[rfm_df['monetary'] > 100]
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

    # Calculate RFM using core API
    rfm_metrics = calculate_rfm(periods, observation_end)

    # Convert List[RFMMetrics] → DataFrame
    return rfm_to_dataframe(rfm_metrics)
