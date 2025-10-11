"""Tests for Lens 1 pandas adapters."""

from datetime import datetime
from decimal import Decimal
import pandas as pd  # type: ignore

from customer_base_audit.foundation.rfm import RFMMetrics
from customer_base_audit.analyses.lens1 import analyze_single_period
from customer_base_audit.pandas import (
    lens1_to_dataframe,
    analyze_single_period_df,
    rfm_to_dataframe,
)


class TestLens1ToDataFrame:
    """Test lens1_to_dataframe conversion."""

    def test_converts_all_fields(self):
        """All Lens1Metrics fields are converted to DataFrame columns."""
        metrics = [
            RFMMetrics(
                "C1",
                10,
                5,
                Decimal("50.00"),
                datetime(2023, 1, 1),
                datetime(2023, 12, 31),
                Decimal("250.00"),
            ),
            RFMMetrics(
                "C2",
                20,
                3,
                Decimal("40.00"),
                datetime(2023, 1, 1),
                datetime(2023, 12, 31),
                Decimal("120.00"),
            ),
        ]
        lens1 = analyze_single_period(metrics)

        df = lens1_to_dataframe(lens1)

        assert len(df) == 1  # Single row
        assert df.iloc[0]["total_customers"] == 2
        assert df.iloc[0]["one_time_buyers"] == 0
        assert "one_time_buyer_pct" in df.columns
        assert "total_revenue" in df.columns
        assert "top_10pct_revenue_contribution" in df.columns


class TestAnalyzeSinglePeriodDF:
    """Test analyze_single_period_df convenience function."""

    def test_end_to_end_lens1_workflow(self):
        """End-to-end Lens 1 DataFrame workflow."""
        metrics = [
            RFMMetrics(
                "C1",
                10,
                1,
                Decimal("100.00"),
                datetime(2023, 1, 1),
                datetime(2023, 12, 31),
                Decimal("100.00"),
            ),
            RFMMetrics(
                "C2",
                20,
                5,
                Decimal("50.00"),
                datetime(2023, 1, 1),
                datetime(2023, 12, 31),
                Decimal("250.00"),
            ),
        ]
        rfm_df = rfm_to_dataframe(metrics)

        lens1_df = analyze_single_period_df(rfm_df)

        assert len(lens1_df) == 1
        assert lens1_df.iloc[0]["total_customers"] == 2
        assert lens1_df.iloc[0]["one_time_buyers"] == 1  # C1 has frequency=1

    def test_empty_dataframe_returns_zero_metrics(self):
        """Empty RFM DataFrame returns zero-valued metrics."""
        rfm_df = pd.DataFrame(
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

        lens1_df = analyze_single_period_df(rfm_df)

        assert len(lens1_df) == 1
        assert lens1_df.iloc[0]["total_customers"] == 0
        assert lens1_df.iloc[0]["total_revenue"] == 0.0
