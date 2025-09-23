"""Synthetic data generation and validation utilities.

This package helps produce realistic-but-fake datasets to exercise
customer audit and CLV pipelines without accessing production data.
"""

from .generator import (
    Customer,
    Transaction,
    ScenarioConfig,
    generate_customers,
    generate_transactions,
)
from .validation import (
    check_non_negative_amounts,
    check_reasonable_order_density,
    check_promo_spike_signal,
)

__all__ = [
    "Customer",
    "Transaction",
    "ScenarioConfig",
    "generate_customers",
    "generate_transactions",
    "check_non_negative_amounts",
    "check_reasonable_order_density",
    "check_promo_spike_signal",
]
