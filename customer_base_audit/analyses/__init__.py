"""Five Lenses customer-base audit analyses.

This package implements the Five Lenses framework from
"The Customer-Base Audit" by Fader, Hardie, and Ross (2022).

The five lenses provide progressively deeper insights into
customer behavior and business health:

1. Single Period Analysis (Lens 1) - snapshot of current state
2. Period-to-Period Comparison (Lens 2) - customer migration
3. Single Cohort Evolution (Lens 3) - retention curves
4. Multi-Cohort Comparison (Lens 4) - cohort quality trends
5. Overall Customer Base Health (Lens 5) - integrative view
"""

from .lens1 import Lens1Metrics, analyze_single_period, calculate_revenue_concentration
from .lens2 import (
    CustomerMigration,
    Lens2Metrics,
    analyze_period_comparison,
)
from .lens3 import (
    CohortPeriodMetrics,
    Lens3Metrics,
    analyze_cohort_evolution,
    calculate_retention_curve,
)

__all__ = [
    # Lens 1
    "Lens1Metrics",
    "analyze_single_period",
    "calculate_revenue_concentration",
    # Lens 2
    "CustomerMigration",
    "Lens2Metrics",
    "analyze_period_comparison",
    # Lens 3
    "CohortPeriodMetrics",
    "Lens3Metrics",
    "analyze_cohort_evolution",
    "calculate_retention_curve",
]
