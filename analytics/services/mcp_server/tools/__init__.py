"""MCP Tools for Five Lenses Analytics.

This module exports all MCP tools for foundation services and lens analyses.
"""

# Foundation services
from .cohorts import create_customer_cohorts
from .data_loader import load_transactions
from .data_mart import build_customer_data_mart

# Lens analyses
from .lens1 import analyze_single_period_snapshot
from .lens2 import analyze_period_to_period_comparison
from .lens3 import analyze_cohort_lifecycle
from .lens4 import compare_multiple_cohorts
from .lens5 import assess_overall_customer_base_health
from .orchestrated_analysis import run_orchestrated_analysis
from .rfm import calculate_rfm_metrics

# Phase 4A: Observability & Resilience
from .execution_metrics import get_execution_metrics, reset_execution_metrics
from .health_check import health_check

__all__ = [
    # Foundation
    "build_customer_data_mart",
    "calculate_rfm_metrics",
    "create_customer_cohorts",
    "load_transactions",
    # Lenses
    "analyze_single_period_snapshot",
    "analyze_period_to_period_comparison",
    "analyze_cohort_lifecycle",
    "compare_multiple_cohorts",
    "assess_overall_customer_base_health",
    # Orchestration (Phase 3)
    "run_orchestrated_analysis",
    # Observability (Phase 4A)
    "health_check",
    "get_execution_metrics",
    "reset_execution_metrics",
]
