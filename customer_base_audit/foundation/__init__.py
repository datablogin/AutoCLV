"""Foundational building blocks for the customer data platform.

This package exposes the customer contract definition as well as
utilities for building reusable customer × time data marts.
"""

from .customer_contract import CustomerContract, CustomerIdentifier
from .data_mart import (
    CustomerDataMart,
    CustomerDataMartBuilder,
    OrderAggregation,
    PeriodAggregation,
    PeriodGranularity,
)

__all__ = [
    "CustomerContract",
    "CustomerIdentifier",
    "CustomerDataMart",
    "CustomerDataMartBuilder",
    "OrderAggregation",
    "PeriodAggregation",
    "PeriodGranularity",
]
