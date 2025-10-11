"""Tests for RFM pandas adapters."""

import pytest
from datetime import datetime
from decimal import Decimal
import pandas as pd

from customer_base_audit.foundation.rfm import RFMMetrics
from customer_base_audit.pandas import (
    rfm_to_dataframe,
    dataframe_to_rfm,
    calculate_rfm_df,
)


class TestRFMToDataFrame:
    """Test rfm_to_dataframe conversion."""

    def test_single_rfm_metric(self):
        """Single RFM metric converts to DataFrame correctly."""
        metrics = [
            RFMMetrics(
                customer_id="C1",
                recency_days=10,
                frequency=5,
                monetary=Decimal("50.00"),
                observation_start=datetime(2023, 1, 1),
                observation_end=datetime(2023, 12, 31),
                total_spend=Decimal("250.00"),
            )
        ]

        df = rfm_to_dataframe(metrics)

        assert len(df) == 1
        assert df.iloc[0]["customer_id"] == "C1"
        assert df.iloc[0]["recency_days"] == 10
        assert df.iloc[0]["frequency"] == 5
        assert df.iloc[0]["monetary"] == 50.0  # Decimal converted to float
        assert df.iloc[0]["total_spend"] == 250.0

    def test_empty_input_returns_empty_dataframe(self):
        """Empty metrics list returns empty DataFrame with correct columns."""
        df = rfm_to_dataframe([])

        assert df.empty
        assert list(df.columns) == [
            "customer_id",
            "recency_days",
            "frequency",
            "monetary",
            "total_spend",
            "observation_start",
            "observation_end",
        ]

    def test_sorted_by_customer_id(self):
        """Output DataFrame is sorted by customer_id."""
        metrics = [
            RFMMetrics(
                "C3",
                10,
                5,
                Decimal("50"),
                datetime(2023, 1, 1),
                datetime(2023, 12, 31),
                Decimal("250"),
            ),
            RFMMetrics(
                "C1",
                20,
                3,
                Decimal("40"),
                datetime(2023, 1, 1),
                datetime(2023, 12, 31),
                Decimal("120"),
            ),
            RFMMetrics(
                "C2",
                15,
                4,
                Decimal("45"),
                datetime(2023, 1, 1),
                datetime(2023, 12, 31),
                Decimal("180"),
            ),
        ]

        df = rfm_to_dataframe(metrics)

        assert list(df["customer_id"]) == ["C1", "C2", "C3"]


class TestDataFrameToRFM:
    """Test dataframe_to_rfm conversion."""

    def test_round_trip_conversion(self):
        """Round-trip conversion preserves data exactly."""
        original = [
            RFMMetrics(
                "C1",
                10,
                5,
                Decimal("50.00"),
                datetime(2023, 1, 1),
                datetime(2023, 12, 31),
                Decimal("250.00"),
            )
        ]

        df = rfm_to_dataframe(original)
        restored = dataframe_to_rfm(df)

        assert len(restored) == 1
        assert restored[0].customer_id == original[0].customer_id
        assert restored[0].recency_days == original[0].recency_days
        assert restored[0].monetary == original[0].monetary

    def test_missing_columns_raises_error(self):
        """DataFrame missing required columns raises ValueError."""
        df = pd.DataFrame(
            {
                "customer_id": ["C1"],
                "recency_days": [10],
                # Missing other columns
            }
        )

        with pytest.raises(ValueError, match="missing required columns"):
            dataframe_to_rfm(df)

    def test_empty_dataframe_returns_empty_list(self):
        """Empty DataFrame returns empty list."""
        df = pd.DataFrame(
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

        result = dataframe_to_rfm(df)

        assert result == []

    def test_nan_values_raise_error(self):
        """DataFrame with NaN values raises ValueError."""
        import numpy as np

        df = pd.DataFrame(
            {
                "customer_id": ["C1"],
                "recency_days": [10],
                "frequency": [np.nan],  # NaN value
                "monetary": [50.0],
                "total_spend": [250.0],
                "observation_start": [datetime(2023, 1, 1)],
                "observation_end": [datetime(2023, 12, 31)],
            }
        )

        with pytest.raises(ValueError, match="Null/NaN values found"):
            dataframe_to_rfm(df)

    def test_null_values_raise_error(self):
        """DataFrame with null values raises ValueError."""
        df = pd.DataFrame(
            {
                "customer_id": ["C1"],
                "recency_days": [10],
                "frequency": [5],
                "monetary": [None],  # Null value
                "total_spend": [250.0],
                "observation_start": [datetime(2023, 1, 1)],
                "observation_end": [datetime(2023, 12, 31)],
            }
        )

        with pytest.raises(ValueError, match="Null/NaN values found"):
            dataframe_to_rfm(df)


class TestCalculateRFMDF:
    """Test calculate_rfm_df convenience function."""

    def test_end_to_end_dataframe_workflow(self):
        """End-to-end DataFrame workflow produces correct results."""
        periods_df = pd.DataFrame(
            {
                "customer_id": ["C1", "C1", "C2"],
                "period_start": [
                    datetime(2023, 1, 1),
                    datetime(2023, 2, 1),
                    datetime(2023, 1, 1),
                ],
                "period_end": [
                    datetime(2023, 2, 1),
                    datetime(2023, 3, 1),
                    datetime(2023, 2, 1),
                ],
                "total_orders": [2, 1, 3],
                "total_spend": [100.0, 50.0, 200.0],
                "total_margin": [30.0, 15.0, 60.0],
                "total_quantity": [5, 2, 8],
            }
        )

        rfm_df = calculate_rfm_df(periods_df, datetime(2023, 4, 15))

        assert len(rfm_df) == 2  # C1 and C2
        assert "customer_id" in rfm_df.columns
        assert "frequency" in rfm_df.columns
        assert "monetary" in rfm_df.columns

    def test_custom_column_names(self):
        """Custom column name mapping works correctly."""
        periods_df = pd.DataFrame(
            {
                "client_id": ["C1", "C2"],
                "start_date": [datetime(2023, 1, 1), datetime(2023, 1, 1)],
                "end_date": [datetime(2023, 2, 1), datetime(2023, 2, 1)],
                "order_count": [2, 3],
                "revenue": [100.0, 200.0],
                "profit": [30.0, 60.0],
                "item_count": [5, 8],
            }
        )

        rfm_df = calculate_rfm_df(
            periods_df,
            datetime(2023, 4, 15),
            customer_id_col="client_id",
            period_start_col="start_date",
            period_end_col="end_date",
            total_orders_col="order_count",
            total_spend_col="revenue",
            total_margin_col="profit",
            total_quantity_col="item_count",
        )

        assert len(rfm_df) == 2
        assert "customer_id" in rfm_df.columns
        assert set(rfm_df["customer_id"]) == {"C1", "C2"}

    @pytest.mark.slow
    def test_performance_10k_customers(self):
        """Performance test with 10k customers."""
        import time

        # Create 10k customer periods
        periods_df = pd.DataFrame(
            {
                "customer_id": [f"C{i}" for i in range(10000)],
                "period_start": [datetime(2023, 1, 1)] * 10000,
                "period_end": [datetime(2023, 2, 1)] * 10000,
                "total_orders": [5] * 10000,
                "total_spend": [250.0] * 10000,
                "total_margin": [75.0] * 10000,
                "total_quantity": [10] * 10000,
            }
        )

        start = time.time()
        rfm_df = calculate_rfm_df(periods_df, datetime(2023, 12, 31))
        duration = time.time() - start

        assert duration < 2.0, f"Took {duration:.2f}s (expected < 2.0s)"
        assert len(rfm_df) == 10000
