"""Tests for Lens 2 pandas adapters."""

from datetime import datetime
from decimal import Decimal
import pandas as pd  # type: ignore

from customer_base_audit.foundation.rfm import RFMMetrics
from customer_base_audit.analyses.lens2 import analyze_period_comparison
from customer_base_audit.pandas import (
    lens2_to_dataframes,
    analyze_period_comparison_df,
    rfm_to_dataframe,
)


class TestLens2ToDataFrames:
    """Test lens2_to_dataframes conversion."""

    def create_rfm(
        self, customer_id: str, frequency: int, spend: Decimal
    ) -> RFMMetrics:
        """Helper to create RFMMetrics for testing."""
        return RFMMetrics(
            customer_id=customer_id,
            recency_days=10,
            frequency=frequency,
            monetary=spend / frequency,
            observation_start=datetime(2023, 1, 1),
            observation_end=datetime(2023, 12, 31),
            total_spend=spend,
        )

    def test_returns_four_dataframes(self):
        """Returns dict with 4 DataFrames: metrics, migration, period1_summary, period2_summary."""
        period1 = [self.create_rfm("C1", 5, Decimal("250"))]
        period2 = [self.create_rfm("C1", 3, Decimal("180"))]

        lens2 = analyze_period_comparison(period1, period2)
        dfs = lens2_to_dataframes(lens2)

        assert set(dfs.keys()) == {
            "metrics",
            "migration",
            "period1_summary",
            "period2_summary",
        }
        assert isinstance(dfs["metrics"], pd.DataFrame)
        assert isinstance(dfs["migration"], pd.DataFrame)

    def test_migration_dataframe_structure(self):
        """Migration DataFrame has customer_id and status columns."""
        period1 = [
            self.create_rfm("C1", 5, Decimal("250")),
            self.create_rfm("C2", 3, Decimal("150")),
        ]
        period2 = [
            self.create_rfm("C1", 3, Decimal("180")),
            self.create_rfm("C3", 2, Decimal("100")),
        ]

        lens2 = analyze_period_comparison(period1, period2)
        dfs = lens2_to_dataframes(lens2)
        migration = dfs["migration"]

        assert list(migration.columns) == ["customer_id", "status"]
        assert "C1" in migration["customer_id"].values  # Retained
        assert "C2" in migration["customer_id"].values  # Churned
        assert "C3" in migration["customer_id"].values  # New

    def test_migration_status_values(self):
        """Migration status column contains correct values."""
        period1 = [self.create_rfm("C1", 5, Decimal("250"))]
        period2 = [self.create_rfm("C2", 3, Decimal("180"))]

        lens2 = analyze_period_comparison(period1, period2)
        dfs = lens2_to_dataframes(lens2)
        migration = dfs["migration"]

        c1_status = migration[migration["customer_id"] == "C1"]["status"].iloc[0]
        c2_status = migration[migration["customer_id"] == "C2"]["status"].iloc[0]

        assert c1_status == "churned"
        assert c2_status == "new"


class TestAnalyzePeriodComparisonDF:
    """Test analyze_period_comparison_df convenience function."""

    def test_end_to_end_lens2_workflow(self):
        """End-to-end Lens 2 DataFrame workflow."""
        period1_rfm = rfm_to_dataframe(
            [
                RFMMetrics(
                    "C1",
                    10,
                    5,
                    Decimal("50"),
                    datetime(2023, 1, 1),
                    datetime(2023, 6, 30),
                    Decimal("250"),
                ),
                RFMMetrics(
                    "C2",
                    20,
                    3,
                    Decimal("50"),
                    datetime(2023, 1, 1),
                    datetime(2023, 6, 30),
                    Decimal("150"),
                ),
            ]
        )
        period2_rfm = rfm_to_dataframe(
            [
                RFMMetrics(
                    "C1",
                    15,
                    3,
                    Decimal("60"),
                    datetime(2023, 7, 1),
                    datetime(2023, 12, 31),
                    Decimal("180"),
                ),
                RFMMetrics(
                    "C3",
                    10,
                    2,
                    Decimal("50"),
                    datetime(2023, 7, 1),
                    datetime(2023, 12, 31),
                    Decimal("100"),
                ),
            ]
        )

        dfs = analyze_period_comparison_df(period1_rfm, period2_rfm)

        assert dfs["metrics"].iloc[0]["retention_rate"] == 50.0  # 1 of 2 retained
        assert len(dfs["migration"]) == 3  # C1, C2, C3
