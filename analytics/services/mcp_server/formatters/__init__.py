"""Rich formatting for Five Lenses customer analytics.

This package provides formatters for converting Five Lenses analysis results
into presentation-ready formats:

- Markdown tables for readable text output
- Plotly charts for interactive visualizations
- Executive summaries with actionable insights

Designed for integration with Claude Desktop MCP server and other
visualization platforms.
"""

from analytics.services.mcp_server.formatters.executive_summaries import (
    generate_cohort_comparison,
    generate_health_summary,
    generate_retention_insights,
)
from analytics.services.mcp_server.formatters.markdown_tables import (
    format_lens1_table,
    format_lens2_table,
    format_lens4_decomposition_table,
    format_lens5_health_summary_table,
)
from analytics.services.mcp_server.formatters.plotly_charts import (
    create_executive_dashboard,
    create_health_score_gauge,
    create_retention_trend_chart,
    create_revenue_concentration_pie,
)

__all__ = [
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
    # Executive summaries
    "generate_health_summary",
    "generate_retention_insights",
    "generate_cohort_comparison",
]
