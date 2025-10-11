"""Shared utilities for pandas conversion operations."""

from decimal import Decimal
from datetime import datetime
import pandas as pd  # type: ignore


def decimal_to_float(value: Decimal) -> float:
    """Convert Decimal to float for pandas compatibility."""
    return float(value)


def float_to_decimal(value: float) -> Decimal:
    """Convert float to Decimal, avoiding precision issues."""
    return Decimal(str(value))


def datetime_to_pandas(dt: datetime) -> pd.Timestamp:
    """Convert Python datetime to pandas Timestamp."""
    return pd.Timestamp(dt)


def pandas_to_datetime(timestamp: pd.Timestamp) -> datetime:
    """Convert pandas Timestamp to Python datetime."""
    return timestamp.to_pydatetime()
