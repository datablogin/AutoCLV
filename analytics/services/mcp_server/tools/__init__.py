"""MCP Tools for Five Lenses Analytics.

This module exports all MCP tools for foundation services and lens analyses.
"""

# Foundation services
from .data_mart import build_customer_data_mart
from .rfm import calculate_rfm_metrics
from .cohorts import create_customer_cohorts

# Lens analyses
from .lens1 import analyze_single_period_snapshot
from .lens2 import analyze_period_to_period_comparison
from .lens3 import analyze_cohort_lifecycle
from .lens4 import compare_multiple_cohorts
from .lens5 import assess_overall_customer_base_health

__all__ = [
    # Foundation
    "build_customer_data_mart",
    "calculate_rfm_metrics",
    "create_customer_cohorts",
    # Lenses
    "analyze_single_period_snapshot",
    "analyze_period_to_period_comparison",
    "analyze_cohort_lifecycle",
    "compare_multiple_cohorts",
    "assess_overall_customer_base_health",
]
