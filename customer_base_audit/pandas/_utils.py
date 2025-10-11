"""Shared utilities for pandas conversion operations."""

from decimal import Decimal


def decimal_to_float(value: Decimal) -> float:
    """Convert Decimal to float for pandas compatibility."""
    return float(value)


def float_to_decimal(value: float) -> Decimal:
    """Convert float to Decimal, avoiding precision issues.

    Warning:
        Floats with >15 significant digits may lose precision due to
        float representation limits. For financial calculations requiring
        exact precision, use Decimal inputs from the start.

    Args:
        value: Float value to convert

    Returns:
        Decimal representation of the float

    Raises:
        TypeError: If value is not numeric

    Example:
        >>> float_to_decimal(123.45)
        Decimal('123.45')
        >>> float_to_decimal(123.456789012345678)  # 17 digits
        Decimal('123.45678901234567')  # Last digit lost
    """
    if not isinstance(value, (int, float)):
        raise TypeError(f"Expected numeric type, got {type(value)}")
    return Decimal(str(value))
