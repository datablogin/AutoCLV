"""Rich formatting for Five Lenses customer analytics.

This package provides formatters for converting Five Lenses analysis results
into presentation-ready formats:

- Markdown tables for readable text output
- Plotly charts for interactive visualizations
- Executive summaries with actionable insights

Designed for integration with Claude Desktop MCP server and other
visualization platforms.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from customer_base_audit.mcp.formatters.executive_summaries import (
    generate_cohort_comparison,
    generate_health_summary,
    generate_retention_insights,
)
from customer_base_audit.mcp.formatters.markdown_tables import (
    format_lens1_table,
    format_lens2_table,
    format_lens4_decomposition_table,
    format_lens5_health_summary_table,
)
from customer_base_audit.mcp.formatters.plotly_charts import (
    create_cohort_heatmap,
    create_executive_dashboard,
    create_health_score_gauge,
    create_retention_trend_chart,
    create_revenue_concentration_pie,
    create_sankey_diagram,
)


@dataclass(frozen=True)
class ChartConfig:
    """Configuration for chart size and quality optimization.

    Balances visualization quality with token efficiency for Claude Desktop.
    Smaller dimensions reduce token usage but may impact readability.

    Attributes
    ----------
    width:
        Chart width in pixels (default: 800, original: 1200)
    height:
        Chart height in pixels (default: 400, original: 600)
    quality:
        Quality preset: 'high' (original size), 'medium' (balanced), 'low' (compact)
    """

    width: int = 800
    height: int = 400
    quality: Literal["high", "medium", "low"] = "medium"

    @classmethod
    def from_quality(cls, quality: Literal["high", "medium", "low"]) -> ChartConfig:
        """Create config from quality preset.

        Parameters
        ----------
        quality:
            Quality level preset

        Returns
        -------
        ChartConfig:
            Configuration with preset dimensions

        Examples
        --------
        >>> config = ChartConfig.from_quality("high")
        >>> config.width
        1200
        >>> config.height
        600
        """
        if quality == "high":
            return cls(width=1200, height=600, quality="high")
        elif quality == "low":
            return cls(width=600, height=300, quality="low")
        else:  # medium (default)
            return cls(width=800, height=400, quality="medium")


# Global default configuration (medium quality for token efficiency)
_DEFAULT_CHART_CONFIG = ChartConfig.from_quality("medium")


def get_chart_config() -> ChartConfig:
    """Get current chart configuration.

    Returns
    -------
    ChartConfig:
        Current global chart configuration
    """
    return _DEFAULT_CHART_CONFIG


def set_chart_config(config: ChartConfig) -> None:
    """Set global chart configuration.

    Parameters
    ----------
    config:
        New chart configuration to use

    Examples
    --------
    >>> from customer_base_audit.mcp.formatters import ChartConfig, set_chart_config
    >>> set_chart_config(ChartConfig.from_quality("high"))
    """
    global _DEFAULT_CHART_CONFIG
    _DEFAULT_CHART_CONFIG = config


__all__ = [
    # Configuration
    "ChartConfig",
    "get_chart_config",
    "set_chart_config",
    # Markdown tables
    "format_lens1_table",
    "format_lens2_table",
    "format_lens4_decomposition_table",
    "format_lens5_health_summary_table",
    # Plotly charts
    "create_retention_trend_chart",
    "create_revenue_concentration_pie",
    "create_health_score_gauge",
    "create_executive_dashboard",
    "create_cohort_heatmap",
    "create_sankey_diagram",
    # Executive summaries
    "generate_health_summary",
    "generate_retention_insights",
    "generate_cohort_comparison",
]
