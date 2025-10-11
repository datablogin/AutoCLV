"""Pandas DataFrame adapters for customer base audit components."""

from .rfm import (
    rfm_to_dataframe,
    dataframe_to_rfm,
    dataframe_to_period_aggregations,
    calculate_rfm_df,
)
from .lens1 import (
    lens1_to_dataframe,
    analyze_single_period_df,
)
from .lens2 import (
    lens2_to_dataframes,
    analyze_period_comparison_df,
)

__all__ = [
    # RFM adapters
    "rfm_to_dataframe",
    "dataframe_to_rfm",
    "dataframe_to_period_aggregations",
    "calculate_rfm_df",
    # Lens 1 adapters
    "lens1_to_dataframe",
    "analyze_single_period_df",
    # Lens 2 adapters
    "lens2_to_dataframes",
    "analyze_period_comparison_df",
]
