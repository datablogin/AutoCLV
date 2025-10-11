"""Pandas DataFrame adapters for Lens 1 analysis."""

from typing import Optional
import pandas as pd  # type: ignore

from customer_base_audit.analyses.lens1 import Lens1Metrics, analyze_single_period
from .rfm import dataframe_to_rfm
from ._utils import decimal_to_float


def lens1_to_dataframe(lens1: Lens1Metrics) -> pd.DataFrame:
    """Convert Lens1Metrics to single-row DataFrame.

    Args:
        lens1: Lens1Metrics object

    Returns:
        Single-row DataFrame with Lens 1 metrics

    Example:
        >>> lens1 = analyze_single_period(rfm_metrics)
        >>> lens1_df = lens1_to_dataframe(lens1)
        >>> print(lens1_df['one_time_buyer_pct'].iloc[0])
    """
    return pd.DataFrame(
        [
            {
                "total_customers": lens1.total_customers,
                "one_time_buyers": lens1.one_time_buyers,
                "one_time_buyer_pct": decimal_to_float(lens1.one_time_buyer_pct),
                "total_revenue": decimal_to_float(lens1.total_revenue),
                "top_10pct_revenue_contribution": decimal_to_float(
                    lens1.top_10pct_revenue_contribution
                ),
                "top_20pct_revenue_contribution": decimal_to_float(
                    lens1.top_20pct_revenue_contribution
                ),
                "avg_orders_per_customer": decimal_to_float(
                    lens1.avg_orders_per_customer
                ),
                "median_customer_value": decimal_to_float(lens1.median_customer_value),
                "rfm_distribution": str(
                    lens1.rfm_distribution
                ),  # Dict as string for CSV export
            }
        ]
    )


def analyze_single_period_df(
    rfm_df: pd.DataFrame,
    rfm_scores_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Perform Lens 1 analysis on DataFrame.

    Convenience function combining conversion and analysis.

    Args:
        rfm_df: DataFrame with RFM metrics
        rfm_scores_df: Optional DataFrame with RFM scores

    Returns:
        Single-row DataFrame with Lens 1 metrics

    Example:
        >>> rfm_df = calculate_rfm_df(periods_df, datetime(2023, 12, 31))
        >>> lens1_df = analyze_single_period_df(rfm_df)
        >>> lens1_df.to_csv('lens1_metrics.csv', index=False)
    """
    # Convert DataFrame → List[RFMMetrics]
    rfm_metrics = dataframe_to_rfm(rfm_df)

    # Convert RFM scores if provided
    rfm_scores = None
    if rfm_scores_df is not None:
        # Note: RFMScore conversion not in scope for Issue #78
        # For now, rfm_scores must be None or passed as list
        raise NotImplementedError("RFMScore DataFrame conversion not yet supported")

    # Use core API
    lens1 = analyze_single_period(rfm_metrics, rfm_scores=rfm_scores)

    # Convert Lens1Metrics → DataFrame
    return lens1_to_dataframe(lens1)
