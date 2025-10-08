"""Foundational building blocks for the customer data platform.

This package exposes the customer contract definition as well as
utilities for building reusable customer × time data marts and
RFM (Recency-Frequency-Monetary) analysis.
"""

from .customer_contract import CustomerContract, CustomerIdentifier
from .data_mart import (
    CustomerDataMart,
    CustomerDataMartBuilder,
    OrderAggregation,
    PeriodAggregation,
    PeriodGranularity,
)
from .rfm import RFMMetrics, RFMScore, calculate_rfm, calculate_rfm_scores

__all__ = [
    "CustomerContract",
    "CustomerIdentifier",
    "CustomerDataMart",
    "CustomerDataMartBuilder",
    "OrderAggregation",
    "PeriodAggregation",
    "PeriodGranularity",
    "RFMMetrics",
    "RFMScore",
    "calculate_rfm",
    "calculate_rfm_scores",
]
