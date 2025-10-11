"""Pandas DataFrame adapters for Lens 2 analysis."""

from typing import Dict
import pandas as pd  # type: ignore

from customer_base_audit.analyses.lens2 import Lens2Metrics, analyze_period_comparison
from .rfm import dataframe_to_rfm
from .lens1 import lens1_to_dataframe
from ._utils import decimal_to_float


def lens2_to_dataframes(lens2: Lens2Metrics) -> Dict[str, pd.DataFrame]:
    """Convert Lens2Metrics to multiple DataFrames.

    Args:
        lens2: Lens2Metrics object with nested structures

    Returns:
        Dictionary with keys:
        - 'metrics': Single-row DataFrame with Lens 2 scalar metrics
        - 'migration': Multi-row DataFrame with customer migration (customer_id, status)
        - 'period1_summary': Single-row DataFrame with period 1 Lens 1 metrics
        - 'period2_summary': Single-row DataFrame with period 2 Lens 1 metrics

    Example:
        >>> lens2 = analyze_period_comparison(period1_rfm, period2_rfm)
        >>> dfs = lens2_to_dataframes(lens2)
        >>> dfs['metrics'].to_csv('lens2_metrics.csv', index=False)
        >>> dfs['migration'].to_csv('customer_migration.csv', index=False)
    """
    # Scalar metrics (single row)
    metrics_df = pd.DataFrame(
        [
            {
                "retention_rate": decimal_to_float(lens2.retention_rate),
                "churn_rate": decimal_to_float(lens2.churn_rate),
                "reactivation_rate": decimal_to_float(lens2.reactivation_rate),
                "customer_count_change": lens2.customer_count_change,
                "revenue_change_pct": decimal_to_float(lens2.revenue_change_pct),
                "avg_order_value_change_pct": decimal_to_float(
                    lens2.avg_order_value_change_pct
                ),
            }
        ]
    )

    # Customer migration (multi-row with status column)
    migration_rows = []
    for customer_id in lens2.migration.retained:
        migration_rows.append({"customer_id": customer_id, "status": "retained"})
    for customer_id in lens2.migration.churned:
        migration_rows.append({"customer_id": customer_id, "status": "churned"})
    for customer_id in lens2.migration.new:
        migration_rows.append({"customer_id": customer_id, "status": "new"})
    for customer_id in lens2.migration.reactivated:
        migration_rows.append({"customer_id": customer_id, "status": "reactivated"})

    migration_df = (
        pd.DataFrame(migration_rows)
        if migration_rows
        else pd.DataFrame(columns=["customer_id", "status"])
    )
    migration_df = (
        migration_df.sort_values("customer_id").reset_index(drop=True)
        if not migration_df.empty
        else migration_df
    )

    # Period summaries (convert nested Lens1Metrics)
    period1_summary = lens1_to_dataframe(lens2.period1_metrics)
    period2_summary = lens1_to_dataframe(lens2.period2_metrics)

    return {
        "metrics": metrics_df,
        "migration": migration_df,
        "period1_summary": period1_summary,
        "period2_summary": period2_summary,
    }


def analyze_period_comparison_df(
    period1_rfm_df: pd.DataFrame,
    period2_rfm_df: pd.DataFrame,
) -> Dict[str, pd.DataFrame]:
    """Compare two periods using DataFrames.

    Convenience function combining conversion and analysis.

    Args:
        period1_rfm_df: DataFrame with period 1 RFM metrics
        period2_rfm_df: DataFrame with period 2 RFM metrics

    Returns:
        Dictionary of DataFrames (same as lens2_to_dataframes)

    Example:
        >>> q1_rfm = calculate_rfm_df(q1_periods, datetime(2023, 3, 31))
        >>> q2_rfm = calculate_rfm_df(q2_periods, datetime(2023, 6, 30))
        >>> comparison = analyze_period_comparison_df(q1_rfm, q2_rfm)
        >>> print(f"Retention: {comparison['metrics']['retention_rate'].iloc[0]}%")
        >>> churned = comparison['migration'][comparison['migration']['status'] == 'churned']
    """
    # Convert DataFrames → List[RFMMetrics]
    period1_rfm = dataframe_to_rfm(period1_rfm_df)
    period2_rfm = dataframe_to_rfm(period2_rfm_df)

    # Use core API
    lens2 = analyze_period_comparison(period1_rfm, period2_rfm)

    # Convert Lens2Metrics → Dict[str, DataFrame]
    return lens2_to_dataframes(lens2)
